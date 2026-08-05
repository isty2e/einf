import ast
from dataclasses import dataclass
from pathlib import Path

import pytest

from einf.analysis.engine import analyze_module, analyze_source
from einf.analysis.model import TextPosition, TextSpan
from einf.analysis.parser import (
    AstParserBackend,
    ParsedModule,
    ParsedNode,
    ParserSyntaxError,
    ParserUnavailableError,
    TextEdit,
)


@dataclass(frozen=True, slots=True)
class _StubParserBackend:
    name: str = "stub"

    def validate_available(self) -> None:
        pass

    def parse(self, source: str, path: Path) -> ParsedModule:
        root_node = ParsedNode(
            node_id=0,
            kind="Module",
            span=TextSpan(
                start=TextPosition(line=1, column=0),
                end=TextPosition(line=1, column=len(source)),
            ),
            value=None,
            child_ids=(),
        )
        return ParsedModule(
            path=path,
            source=source,
            nodes=(root_node,),
            root_id=0,
        )

    def reparse(
        self,
        previous: ParsedModule,
        edits: tuple[TextEdit, ...],
        new_source: str,
    ) -> ParsedModule:
        _ = edits
        return self.parse(source=new_source, path=previous.path)


def test_analyze_module_uses_parser_backend() -> None:
    backend = _StubParserBackend()
    result = analyze_module(
        source="x = 1",
        path=Path("sample.py"),
        parser_backend=backend,
    )
    assert result.module.root().kind == "Module"
    assert result.module.path == Path("sample.py")
    assert result.diagnostics == ()
    assert result.axis_tokens == ()


def test_analyze_module_propagates_parser_failure() -> None:
    @dataclass(frozen=True, slots=True)
    class _FailingParserBackend:
        name: str = "failing"

        def validate_available(self) -> None:
            pass

        def parse(self, source: str, path: Path) -> ParsedModule:
            _ = source
            _ = path
            raise ParserSyntaxError(message="bad source", span=None)

        def reparse(
            self,
            previous: ParsedModule,
            edits: tuple[TextEdit, ...],
            new_source: str,
        ) -> ParsedModule:
            _ = previous
            _ = edits
            _ = new_source
            raise NotImplementedError

    with pytest.raises(ParserSyntaxError, match="bad source"):
        analyze_module(
            source="x =",
            path=Path("sample.py"),
            parser_backend=_FailingParserBackend(),
        )


def test_analyze_module_emits_diagnostics_and_tokens_from_real_parser() -> None:
    result = analyze_module(
        source="from einf import reduce\nreduce(ax[b, n], ax[z])\n",
        path=Path("sample.py"),
        parser_backend=AstParserBackend(),
    )
    assert len(result.diagnostics) == 1
    assert result.diagnostics[0].code == "ANALYSIS_AXIS_NOT_IN_INPUT"
    assert len(result.axis_tokens) == 3


def test_analyze_module_reuses_backend_parse_for_call_bindings(monkeypatch) -> None:
    source = (
        "from einf import rearrange\n"
        + "\n".join("rearrange(ax[b, n], ax[n, b])" for _ in range(4))
        + "\n"
    )
    parse_modes: list[str | None] = []
    real_parse = ast.parse

    def counting_parse(*args, **kwargs) -> ast.Module:
        parse_modes.append(kwargs.get("mode"))
        return real_parse(*args, **kwargs)

    monkeypatch.setattr("ast.parse", counting_parse)
    analyze_module(
        source=source,
        path=Path("sample.py"),
        parser_backend=AstParserBackend(),
    )

    full_parses = [mode for mode in parse_modes if mode is None]
    assert len(full_parses) == 1
    assert full_parses == [None]


def test_analyze_source_keeps_clean_report_without_einf_calls() -> None:
    report = analyze_source(
        source="value = 1\n",
        path=Path("plain.py"),
        parser_backend=AstParserBackend(),
    )

    assert report.path == "plain.py"
    assert report.diagnostics == ()
    assert report.axis_tokens == ()
    assert report.failures == ()
    assert not report.has_errors()


def test_analyze_source_accepts_empty_source() -> None:
    report = analyze_source(
        source="",
        path=Path("empty.py"),
        parser_backend=AstParserBackend(),
    )

    assert report.diagnostics == ()
    assert report.axis_tokens == ()
    assert report.failures == ()


def test_analyze_module_is_repeatable_on_shared_backend() -> None:
    backend = AstParserBackend()
    source = "from einf import reduce\nreduce(ax[b, n], ax[z])\n"

    first = analyze_module(
        source=source,
        path=Path("sample.py"),
        parser_backend=backend,
    )
    second = analyze_module(
        source=source,
        path=Path("sample.py"),
        parser_backend=backend,
    )

    assert second.diagnostics == first.diagnostics
    assert second.axis_tokens == first.axis_tokens
    assert second.module.stdlib_module is not None


def test_analyze_source_reports_unavailable_parser_as_failure(monkeypatch) -> None:
    @dataclass(frozen=True, slots=True)
    class _UnavailableParserBackend:
        name: str = "unavailable"

        def validate_available(self) -> None:
            pass

        def parse(self, source: str, path: Path) -> ParsedModule:
            _ = source
            _ = path
            raise ParserUnavailableError(
                backend="unavailable",
                message="unavailable parser",
            )

        def reparse(
            self,
            previous: ParsedModule,
            edits: tuple[TextEdit, ...],
            new_source: str,
        ) -> ParsedModule:
            _ = previous
            _ = edits
            _ = new_source
            raise NotImplementedError

    report = analyze_source(
        source="x = 1\n",
        path=Path("sample.py"),
        parser_backend=_UnavailableParserBackend(),
    )

    assert report.diagnostics == ()
    assert len(report.failures) == 1
    assert report.failures[0].kind == "parser_unavailable"
    assert report.failures[0].message == "unavailable parser"
