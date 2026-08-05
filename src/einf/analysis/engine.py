from dataclasses import dataclass
from pathlib import Path

from einf.analysis.model import AnalysisDiagnostic, AxisToken
from einf.analysis.parser import (
    ParsedModule,
    ParserBackend,
    ParserSyntaxError,
    ParserUnavailableError,
)
from einf.analysis.passes import analyze_einf_calls
from einf.analysis.report import (
    AnalysisFileReport,
    parse_error_report,
    parser_unavailable_report,
)


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


def analyze_source(
    *,
    source: str,
    path: Path,
    parser_backend: ParserBackend,
) -> AnalysisFileReport:
    """Analyze one in-memory source string into a per-file analysis report."""
    try:
        output = analyze_module(source=source, path=path, parser_backend=parser_backend)
    except ParserSyntaxError as error:
        return parse_error_report(path=path, error=error)
    except ParserUnavailableError as error:
        return parser_unavailable_report(path=path, error=error)

    return AnalysisFileReport(
        path=str(path),
        diagnostics=output.diagnostics,
        checker_diagnostics=(),
        axis_tokens=output.axis_tokens,
        failures=(),
    )
