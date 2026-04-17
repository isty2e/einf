from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from typing_extensions import Self

from einf.analysis.checkers import SUPPORTED_CHECKER_NAMES
from einf.analysis.validator.run import SUPPORTED_PARSER_NAMES

InitializeOptions = Mapping[str, str | Sequence[str] | None]


@dataclass(frozen=True, slots=True)
class LspConfig:
    """Static LSP session configuration for the `einf-lsp` sidecar.

    The recommended editor model runs `einf-lsp` alongside a primary Python
    language server and leaves ``checkers`` empty. Configured checkers act as a
    fallback single-server mode for editors that cannot comfortably host a
    separate Python checker server.
    """

    parser: str = "ast"
    checkers: tuple[str, ...] = ()

    @classmethod
    def from_initialize_options(
        cls,
        initialize_options: InitializeOptions | None,
    ) -> Self:
        """Build one canonical sidecar config from LSP initialize options."""
        if initialize_options is None:
            return cls()

        parser_value = initialize_options.get("parser")
        parser = parser_value if parser_value in SUPPORTED_PARSER_NAMES else "ast"

        checker_values = initialize_options.get("checkers")
        if not isinstance(checker_values, Sequence) or isinstance(checker_values, str):
            return cls(parser=parser)

        ordered_checkers: list[str] = []
        for checker_value in checker_values:
            if checker_value not in SUPPORTED_CHECKER_NAMES:
                continue
            if checker_value in ordered_checkers:
                continue
            ordered_checkers.append(checker_value)
        return cls(parser=parser, checkers=tuple(ordered_checkers))


__all__ = ["InitializeOptions", "LspConfig"]
