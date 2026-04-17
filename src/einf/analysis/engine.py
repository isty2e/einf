from dataclasses import dataclass
from pathlib import Path

from einf.analysis.model import AnalysisDiagnostic, AxisToken
from einf.analysis.parser import ParsedModule, ParserBackend
from einf.analysis.passes import analyze_einf_calls


@dataclass(frozen=True, slots=True)
class AnalysisOutput:
    """Static-analysis output bundle for validator/LSP consumers."""

    module: ParsedModule
    diagnostics: tuple[AnalysisDiagnostic, ...]
    axis_tokens: tuple[AxisToken, ...]


def analyze_module(
    *,
    source: str,
    path: Path,
    parser_backend: ParserBackend,
) -> AnalysisOutput:
    """Parse and return one baseline analysis output."""
    parsed_module = parser_backend.parse(source=source, path=path)
    diagnostics, axis_tokens = analyze_einf_calls(parsed_module)
    return AnalysisOutput(
        module=parsed_module,
        diagnostics=diagnostics,
        axis_tokens=axis_tokens,
    )
