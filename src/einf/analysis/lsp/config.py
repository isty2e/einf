import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Self

from einf.analysis.checkers import SUPPORTED_CHECKER_NAMES, CheckerExecutionPolicy
from einf.analysis.parser import SUPPORTED_PARSER_NAMES

from .analysis_limits import LspAnalysisLimits

InitializeOptionValue = str | int | float | Sequence[str] | None
InitializeOptions = Mapping[str, InitializeOptionValue]


@dataclass(frozen=True, slots=True)
class LspConfig:
    """Static LSP session configuration for the `einf-lsp` sidecar.

    The recommended editor model runs `einf-lsp` alongside a primary Python
    language server and leaves ``checkers`` empty. Configured checkers act as a
    fallback single-server mode for editors that cannot comfortably host a
    separate Python checker server.

    Parameters
    ----------
    parser : str, optional
        Semantic parser name, normally ``ast`` or ``libcst``.
    checkers : tuple[str, ...], optional
        External checkers enabled on saved-file boundaries.
    checker_execution_policy : CheckerExecutionPolicy, optional
        Independent process and output bounds for external checkers.
    analysis_limits : LspAnalysisLimits, optional
        Opt-in limits on logical semantic-analysis source captures.
    """

    parser: str = "ast"
    checkers: tuple[str, ...] = ()
    checker_execution_policy: CheckerExecutionPolicy = field(
        default_factory=CheckerExecutionPolicy
    )
    analysis_limits: LspAnalysisLimits = field(default_factory=LspAnalysisLimits)

    @classmethod
    def from_initialize_options(
        cls,
        initialize_options: InitializeOptions | None,
    ) -> Self:
        """Build a session configuration from LSP initialize options.

        Parameters
        ----------
        initialize_options : InitializeOptions or None
            Editor options. Missing or null analysis limits remain unrestricted.

        Returns
        -------
        Self
            Normalized parser, checker and analysis policies.

        Raises
        ------
        ValueError
            An explicit analysis limit is not a positive built-in integer or null.
        """
        if initialize_options is None:
            return cls()

        parser_value = initialize_options.get("parser")
        parser = parser_value if parser_value in SUPPORTED_PARSER_NAMES else "ast"

        checker_values = initialize_options.get("checkers")
        ordered_checkers: list[str] = []
        if isinstance(checker_values, Sequence) and not isinstance(checker_values, str):
            for checker_value in checker_values:
                if checker_value not in SUPPORTED_CHECKER_NAMES:
                    continue
                if checker_value in ordered_checkers:
                    continue
                ordered_checkers.append(checker_value)

        defaults = CheckerExecutionPolicy()
        timeout_seconds = _positive_float(
            initialize_options.get("checkerTimeoutSeconds"),
            default=defaults.timeout_seconds,
        )
        cleanup_timeout_seconds = _positive_float(
            initialize_options.get("checkerCleanupTimeoutSeconds"),
            default=defaults.cleanup_timeout_seconds,
        )
        max_concurrency = _positive_int(
            initialize_options.get("checkerMaxConcurrency"),
            default=defaults.max_concurrency,
        )
        max_output_bytes = _positive_int(
            initialize_options.get("checkerMaxOutputBytes"),
            default=defaults.max_output_bytes,
        )
        max_diagnostics = _positive_int(
            initialize_options.get("checkerMaxDiagnostics"),
            default=defaults.max_diagnostics,
        )
        max_field_length = _positive_int(
            initialize_options.get("checkerMaxFieldLength"),
            default=defaults.max_field_length,
        )
        return cls(
            parser=parser,
            checkers=tuple(ordered_checkers),
            checker_execution_policy=CheckerExecutionPolicy(
                timeout_seconds=timeout_seconds,
                cleanup_timeout_seconds=cleanup_timeout_seconds,
                max_concurrency=max_concurrency,
                max_output_bytes=max_output_bytes,
                max_diagnostics=max_diagnostics,
                max_field_length=max_field_length,
            ),
            analysis_limits=LspAnalysisLimits(
                max_documents=_optional_limit(
                    initialize_options, "analysisMaxDocuments"
                ),
                max_document_bytes=_optional_limit(
                    initialize_options, "analysisMaxDocumentBytes"
                ),
                max_source_bytes=_optional_limit(
                    initialize_options, "analysisMaxSourceBytes"
                ),
            ),
        )


def _optional_limit(options: InitializeOptions, name: str) -> int | None:
    value = options.get(name)
    if value is None:
        return None
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer or null")
    return value


def _positive_float(value: InitializeOptionValue, *, default: float) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        return default
    return float(value)


def _positive_int(value: InitializeOptionValue, *, default: int) -> int:
    if type(value) is not int or value < 1:
        return default
    return value


__all__ = ["InitializeOptions", "LspConfig"]
