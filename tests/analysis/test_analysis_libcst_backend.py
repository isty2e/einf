import importlib
import importlib.util
from pathlib import Path

import pytest

from einf.analysis.parser import LibCstParserBackend


def test_libcst_backend_requires_optional_dependency() -> None:
    if importlib.util.find_spec("libcst") is not None:
        pytest.skip("libcst is installed in this environment")

    backend = LibCstParserBackend()
    with pytest.raises(RuntimeError, match=r"einf\[analysis\]"):
        backend.parse(source="x = 1", path=Path("sample.py"))


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
    with pytest.raises(RuntimeError, match=r"einf\[analysis\]"):
        backend.parse(source="x = 1\n", path=Path("sample.py"))


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
