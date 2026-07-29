from dataclasses import dataclass
from pathlib import Path

import pytest

from einf.analysis.engine import analyze_module
from einf.analysis.model import TextPosition, TextSpan
from einf.analysis.parser import (
    AstParserBackend,
    ParsedModule,
    ParsedNode,
    ParserSyntaxError,
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
