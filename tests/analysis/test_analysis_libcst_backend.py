import importlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

from einf.analysis.engine import analyze_module
from einf.analysis.lsp.service import LspService
from einf.analysis.model import TextPosition, TextSpan
from einf.analysis.parser import (
    AstParserBackend,
    LibCstParserBackend,
    ParserSyntaxError,
    ParserUnavailableError,
)
from einf.analysis.validator.cli import main

_INVALID_DSL_SOURCE = """from einf import ax, axes, reduce
b, n, z = axes("b", "n", "z")
reduce(ax[b, n], ax[b, z])
"""
_MALFORMED_SOURCE = "from einf import rearrange\nrearrange(\n"


def test_libcst_backend_requires_optional_dependency() -> None:
    if importlib.util.find_spec("libcst") is not None:
        pytest.skip("libcst is installed in this environment")

    backend = LibCstParserBackend()
    with pytest.raises(ParserUnavailableError, match=r"einf\[analysis\]"):
        backend.validate_available()


def test_libcst_backend_reports_runtime_error_when_import_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import einf.analysis.parser.libcst_backend as libcst_backend_module

    import_module = importlib.import_module

    def _fake_import_module(name: str):
        if name in {"libcst", "libcst.metadata"}:
            raise ModuleNotFoundError(name)
        return import_module(name)

    monkeypatch.setattr(
        libcst_backend_module.importlib,
        "import_module",
        _fake_import_module,
    )
    backend = LibCstParserBackend()
    with pytest.raises(ParserUnavailableError, match=r"einf\[analysis\]"):
        backend.validate_available()


def test_libcst_backend_rejects_incomplete_optional_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import einf.analysis.parser.libcst_backend as libcst_backend_module

    libcst = ModuleType("libcst")
    libcst.ParserSyntaxError = SyntaxError
    libcst_metadata = ModuleType("libcst.metadata")
    libcst_metadata.MetadataWrapper = type("MetadataWrapper", (), {})
    libcst_metadata.PositionProvider = type("PositionProvider", (), {})

    def _fake_import_module(name: str):
        if name == "libcst":
            return libcst
        if name == "libcst.metadata":
            return libcst_metadata
        return importlib.import_module(name)

    monkeypatch.setattr(
        libcst_backend_module.importlib,
        "import_module",
        _fake_import_module,
    )

    with pytest.raises(ParserUnavailableError, match="dependencies are incomplete"):
        LibCstParserBackend().validate_available()


def test_libcst_validator_cli_reports_unavailable_parser(
    tmp_path: Path,
) -> None:
    target = tmp_path / "sample.py"
    target.write_text("x = 1\n", encoding="utf-8")
    script = """
import importlib
import sys

real_import_module = importlib.import_module

def blocked_import_module(name, package=None):
    if name == "libcst" or name.startswith("libcst."):
        raise ModuleNotFoundError(name)
    return real_import_module(name, package)

importlib.import_module = blocked_import_module

from einf.analysis.validator.cli import main

raise SystemExit(main(["--parser", "libcst", sys.argv[1]]))
"""

    result = subprocess.run(
        [sys.executable, "-c", script, str(target)],
        check=False,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)

    assert result.returncode == 1
    assert result.stderr == ""
    assert payload["files"][0]["failures"] == [
        {
            "kind": "parser_unavailable",
            "message": (
                "libcst parser backend requires libcst; "
                "install einf[analysis] to enable it"
            ),
            "span": None,
        }
    ]


def test_libcst_lsp_reports_unavailable_parser_before_source_filter(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import einf.analysis.parser.libcst_backend as libcst_backend_module

    import_module = importlib.import_module

    def _fake_import_module(name: str):
        if name in {"libcst", "libcst.metadata"}:
            raise ModuleNotFoundError(name)
        return import_module(name)

    monkeypatch.setattr(
        libcst_backend_module.importlib,
        "import_module",
        _fake_import_module,
    )

    state = LspService("libcst").open_document(
        uri=(tmp_path / "sample.py").as_uri(),
        source="x = 1\n",
        version=1,
    )

    assert tuple(failure.kind for failure in state.report.failures) == (
        "parser_unavailable",
    )


def test_libcst_backend_parse_graph_invariants_when_installed() -> None:
    if importlib.util.find_spec("libcst") is None:
        pytest.skip("libcst is not installed in this environment")

    backend = LibCstParserBackend()
    module = backend.parse(source="x = 1\nprint(x)\n", path=Path("sample.py"))
    root = module.root()
    children = module.children(root)
    assert root.node_id == module.root_id
    assert root.kind == "Module"
    assert tuple(child.node_id for child in children) == root.child_ids
    assert all(module.node(child.node_id) == child for child in children)


def test_libcst_backend_node_spans_are_non_inverted_when_installed() -> None:
    if importlib.util.find_spec("libcst") is None:
        pytest.skip("libcst is not installed in this environment")

    backend = LibCstParserBackend()
    module = backend.parse(source="f(\n    x,\n)\n", path=Path("sample.py"))
    for node in module.nodes:
        if node.span is None:
            continue
        starts_after_end = node.span.start.line > node.span.end.line or (
            node.span.start.line == node.span.end.line
            and node.span.start.column > node.span.end.column
        )
        assert not starts_after_end


def test_libcst_backend_call_spans_use_wrapper_owned_tree_when_installed() -> None:
    if importlib.util.find_spec("libcst") is None:
        pytest.skip("libcst is not installed in this environment")

    module = LibCstParserBackend().parse(
        source=_INVALID_DSL_SOURCE,
        path=Path("sample.py"),
    )
    call_nodes = tuple(node for node in module.nodes if node.kind == "Call")

    assert call_nodes
    assert all(node.span is not None for node in call_nodes)


def test_libcst_backend_matches_ast_diagnostics_and_tokens_when_installed() -> None:
    if importlib.util.find_spec("libcst") is None:
        pytest.skip("libcst is not installed in this environment")

    ast_output = analyze_module(
        source=_INVALID_DSL_SOURCE,
        path=Path("sample.py"),
        parser_backend=AstParserBackend(),
    )
    libcst_output = analyze_module(
        source=_INVALID_DSL_SOURCE,
        path=Path("sample.py"),
        parser_backend=LibCstParserBackend(),
    )

    assert libcst_output.diagnostics == ast_output.diagnostics
    assert libcst_output.axis_tokens == ast_output.axis_tokens


def test_libcst_backend_normalizes_syntax_error_when_installed() -> None:
    if importlib.util.find_spec("libcst") is None:
        pytest.skip("libcst is not installed in this environment")

    with pytest.raises(ParserSyntaxError) as error_info:
        LibCstParserBackend().parse(
            source=_MALFORMED_SOURCE,
            path=Path("sample.py"),
        )

    assert error_info.value.span == TextSpan(
        start=TextPosition(line=2, column=0),
        end=TextPosition(line=2, column=1),
    )


def test_libcst_validator_cli_reports_malformed_source_when_installed(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    if importlib.util.find_spec("libcst") is None:
        pytest.skip("libcst is not installed in this environment")

    target = tmp_path / "malformed.py"
    target.write_text(_MALFORMED_SOURCE, encoding="utf-8")

    exit_code = main(["--parser", "libcst", str(target)])
    payload = json.loads(capsys.readouterr().out)
    failure = payload["files"][0]["failures"][0]

    assert exit_code == 1
    assert failure["kind"] == "parse_error"
    assert failure["span"]["start"] == {"line": 2, "column": 0}


def test_libcst_lsp_reports_malformed_source_when_installed(tmp_path: Path) -> None:
    if importlib.util.find_spec("libcst") is None:
        pytest.skip("libcst is not installed in this environment")

    state = LspService("libcst").open_document(
        uri=(tmp_path / "malformed.py").as_uri(),
        source=_MALFORMED_SOURCE,
        version=1,
    )

    assert tuple(failure.kind for failure in state.report.failures) == ("parse_error",)
    assert state.report.failures[0].span == TextSpan(
        start=TextPosition(line=2, column=0),
        end=TextPosition(line=2, column=1),
    )


def test_libcst_validator_cli_reports_invalid_dsl_when_installed(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    if importlib.util.find_spec("libcst") is None:
        pytest.skip("libcst is not installed in this environment")

    target = tmp_path / "invalid_dsl.py"
    target.write_text(_INVALID_DSL_SOURCE, encoding="utf-8")

    exit_code = main(["--parser", "libcst", str(target)])
    payload = json.loads(capsys.readouterr().out)
    file_payload = payload["files"][0]

    assert exit_code == 1
    assert tuple(item["code"] for item in file_payload["diagnostics"]) == (
        "ANALYSIS_AXIS_NOT_IN_INPUT",
    )
    assert len(file_payload["axis_tokens"]) == 4


def test_libcst_backend_reparse_uses_new_source_and_previous_path_when_installed() -> (
    None
):
    if importlib.util.find_spec("libcst") is None:
        pytest.skip("libcst is not installed in this environment")

    backend = LibCstParserBackend()
    previous = backend.parse(source="x = 1\n", path=Path("sample.py"))
    reparsed = backend.reparse(
        previous=previous,
        edits=(),
        new_source="x = 2\ny = 3\n",
    )
    assert reparsed.path == previous.path
    assert reparsed.source == "x = 2\ny = 3\n"
    assert reparsed.nodes is not previous.nodes
