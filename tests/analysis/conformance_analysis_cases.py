from dataclasses import dataclass

VALID_SOURCE = """from einf import ax, axes, rearrange\nb = axes(\"b\")[0]\nrearrange(ax[b], ax[b])\n"""
INVALID_SOURCE = """from einf import ax, axes, reduce\nb, n, z = axes(\"b\", \"n\", \"z\")\nreduce(ax[b, n], ax[b, z])\n"""
SYNTAX_ERROR_SOURCE = "from einf import rearrange\nrearrange(\n"
CONTRACT_SOURCE = """from einf import ax, axes, contract
i, k, j = axes("i", "k", "j")
contract((ax[i, k], ax[k, j]), ax[i, j])
"""


@dataclass(frozen=True, slots=True)
class ValidatorConformanceCase:
    """Language-neutral validator contract fixture."""

    name: str
    source: str
    parser: str
    expected_diagnostic_codes: tuple[str, ...]
    expected_failure_kinds: tuple[str, ...]
    expected_axis_token_count: int
    expected_exit_code: int


@dataclass(frozen=True, slots=True)
class LspConformanceCase:
    """Language-neutral LSP service contract fixture."""

    name: str
    source: str
    expected_diagnostic_codes: tuple[str, ...]
    expected_failure_kinds: tuple[str, ...]
    expect_checker_refresh_on_save: bool
    expected_axis_token_count: int
    expected_semantic_token_int_count: int
    expected_inlay_labels: tuple[str, ...] = ()
    hover_line: int | None = None
    hover_column: int | None = None
    expected_hover_contains: tuple[str, ...] = ()


VALIDATOR_CASES = (
    ValidatorConformanceCase(
        name="validator_valid_source",
        source=VALID_SOURCE,
        parser="ast",
        expected_diagnostic_codes=(),
        expected_failure_kinds=(),
        expected_axis_token_count=2,
        expected_exit_code=0,
    ),
    ValidatorConformanceCase(
        name="validator_semantic_diagnostic",
        source=INVALID_SOURCE,
        parser="ast",
        expected_diagnostic_codes=("ANALYSIS_AXIS_NOT_IN_INPUT",),
        expected_failure_kinds=(),
        expected_axis_token_count=4,
        expected_exit_code=1,
    ),
    ValidatorConformanceCase(
        name="validator_parse_failure",
        source=SYNTAX_ERROR_SOURCE,
        parser="ast",
        expected_diagnostic_codes=(),
        expected_failure_kinds=("parse_error",),
        expected_axis_token_count=0,
        expected_exit_code=1,
    ),
)

LSP_CASES = (
    LspConformanceCase(
        name="lsp_valid_document",
        source=VALID_SOURCE,
        expected_diagnostic_codes=(),
        expected_failure_kinds=(),
        expect_checker_refresh_on_save=False,
        expected_axis_token_count=2,
        expected_semantic_token_int_count=10,
    ),
    LspConformanceCase(
        name="lsp_semantic_diagnostic",
        source=INVALID_SOURCE,
        expected_diagnostic_codes=("ANALYSIS_AXIS_NOT_IN_INPUT",),
        expected_failure_kinds=(),
        expect_checker_refresh_on_save=False,
        expected_axis_token_count=4,
        expected_semantic_token_int_count=20,
    ),
    LspConformanceCase(
        name="lsp_parse_failure",
        source=SYNTAX_ERROR_SOURCE,
        expected_diagnostic_codes=(),
        expected_failure_kinds=("parse_error",),
        expect_checker_refresh_on_save=False,
        expected_axis_token_count=0,
        expected_semantic_token_int_count=0,
    ),
    LspConformanceCase(
        name="lsp_contracted_axis_metadata",
        source=CONTRACT_SOURCE,
        expected_diagnostic_codes=(),
        expected_failure_kinds=(),
        expect_checker_refresh_on_save=False,
        expected_axis_token_count=6,
        expected_semantic_token_int_count=30,
        expected_inlay_labels=("contract", "contract"),
        hover_line=3,
        hover_column=16,
        expected_hover_contains=("**Axis** `k`", "occurrences: 2", "contracted"),
    ),
)
