from collections.abc import Callable, Mapping
from dataclasses import dataclass
from fractions import Fraction

from .result import (
    DynamicCaseResult,
    DynamicRun,
    FixedCaseResult,
    FixedRun,
    PairedEvidence,
    TestResult,
    TimingSummary,
    UnavailableRun,
)
from .types import LibraryName
from .workload import (
    DimensionMode,
    DynamicWorkloadComparison,
    DynamicWorkloadMetadata,
)

_LIBRARY_NAMES: tuple[LibraryName, ...] = ("einf", "einops", "einx")


def _format_summary(summary: TimingSummary | None) -> str:
    if summary is None:
        return "n/a | n/a | n/a | n/a | n/a | n/a"
    return (
        f"{summary.count} | {summary.p25_ms:.4f} | {summary.median_ms:.4f} | "
        f"{summary.p75_ms:.4f} | {summary.iqr_ms:.4f} | {summary.p95_ms:.4f}"
    )


def _format_shape(shape: tuple[int, ...]) -> str:
    return " x ".join(map(str, shape)) if shape else "scalar"


def _format_shapes(shapes: tuple[tuple[int, ...], ...]) -> str:
    return ", ".join(f"`{_format_shape(shape)}`" for shape in shapes)


def _format_ratio(ratio: Fraction) -> str:
    exact = (
        str(ratio.numerator)
        if ratio.denominator == 1
        else f"{ratio.numerator}/{ratio.denominator}"
    )
    return f"`{exact}` ({float(ratio):.3f}x)"


def _render_workload(
    *,
    metadata: DynamicWorkloadMetadata,
    comparison: DynamicWorkloadComparison,
) -> list[str]:
    ratios = dict(comparison.dimension_ratios)
    lines = [
        "Workload:",
        "",
        (
            f"- Profile comparison: `{comparison.scale}` / "
            f"`{comparison.reference_scale}`"
        ),
    ]
    for dimension in metadata.dimensions:
        mode = (
            "sampled per batch"
            if dimension.mode is DimensionMode.SAMPLED
            else "fixed at profile base"
        )
        lines.append(
            f"- `{dimension.name}`: {mode}; base `{dimension.base}`; "
            f"inclusive range `[{dimension.minimum}, {dimension.maximum}]`; "
            f"base ratio {_format_ratio(ratios[dimension.name])}"
        )
    lines.extend(
        [
            f"- Base input shapes: {_format_shapes(metadata.base_input_shapes)}",
            f"- Base output shapes: {_format_shapes(metadata.base_output_shapes)}",
            (
                f"- Total base input elements: `{metadata.base_input_elements}`; "
                "ratio "
                f"{_format_ratio(comparison.base_input_elements_ratio)}"
            ),
            (
                f"- Total base output elements: `{metadata.base_output_elements}`; "
                "ratio "
                f"{_format_ratio(comparison.base_output_elements_ratio)}"
            ),
            "",
        ]
    )
    return lines


def _format_round_medians(
    *,
    round_summaries: tuple[TimingSummary, ...] | None,
    library_names: tuple[LibraryName, ...],
    resolver: Callable[[LibraryName, int], TimingSummary | None],
) -> list[str]:
    if round_summaries is None:
        return []
    lines = ["", "Round median summaries (ms):", ""]
    for round_index in range(len(round_summaries)):
        entries: list[str] = []
        for lib_name in library_names:
            summary = resolver(lib_name, round_index)
            if summary is None:
                entries.append(f"{lib_name}=n/a")
            else:
                entries.append(f"{lib_name}={summary.median_ms:.4f}")
        lines.append(f"- round {round_index + 1}: `{', '.join(entries)}`")
    return lines


def _render_paired_comparisons(
    *,
    evidence: PairedEvidence,
    heading: str,
    unit_label: str,
) -> list[str]:
    if not evidence.comparisons:
        return []

    lines = [
        "",
        heading,
        "",
        (
            "| Comparison | Call pairs | "
            f"{unit_label} | Ratio | Bootstrap level | Bootstrap interval | Difference |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for comparison in evidence.comparisons:
        percent_difference = (comparison.latency_ratio - 1.0) * 100.0
        lines.append(
            f"| {comparison.competitor} / {comparison.baseline} | "
            f"{comparison.call_pair_count} | "
            f"{comparison.paired_unit_count} | "
            f"{comparison.latency_ratio:.4f} | "
            f"{comparison.confidence_level:.0%} | "
            f"[{comparison.confidence_interval_low:.4f}, "
            f"{comparison.confidence_interval_high:.4f}] | "
            f"{percent_difference:+.2f}% |"
        )
    return lines


@dataclass(frozen=True, slots=True)
class MarkdownPrinter:
    """Render benchmark test results to markdown."""

    def _render_fixed_case_table(self, case_result: FixedCaseResult) -> str:
        lines = [
            f"### {case_result.case.name}",
            "",
            case_result.case.description,
            "",
            "Execution forms:",
            "",
            f"- `einf`: `{case_result.case.calls.einf}`",
            f"- `einops`: `{case_result.case.calls.einops}`",
            f"- `einx`: `{case_result.case.calls.einx}`",
            "",
        ]
        if case_result.round_orders:
            lines.extend(
                ["Round base order (paired execution rotates within each round):", ""]
            )
            for round_index, order in enumerate(case_result.round_orders, start=1):
                lines.append(f"- round {round_index}: `{' -> '.join(order)}`")
            lines.append("")
        lines.extend(
            [
                "| Library | Cold n | Cold p25 (ms) | Cold median (ms) | Cold p75 (ms) | Cold IQR (ms) | Cold p95 (ms) | Warm n | Warm p25 (ms) | Warm median (ms) | Warm p75 (ms) | Warm IQR (ms) | Warm p95 (ms) |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for lib_name in _LIBRARY_NAMES:
            run = case_result.runs[lib_name]
            if isinstance(run, UnavailableRun):
                lines.append(
                    f"| {lib_name} | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a ({run.reason}) |"
                )
                continue
            lines.append(
                f"| {lib_name} | {_format_summary(run.cold)} | {_format_summary(run.warm)} |"
            )

        lines.extend(
            _render_paired_comparisons(
                evidence=case_result.cold_evidence,
                heading="Cold paired latency ratios (competitor / einf):",
                unit_label="Paired trial units",
            )
        )
        lines.extend(
            _render_paired_comparisons(
                evidence=case_result.warm_evidence,
                heading="Warm paired latency ratios (competitor / einf):",
                unit_label="Paired timing units",
            )
        )

        def resolve_warm_round(
            lib_name: LibraryName, round_index: int
        ) -> TimingSummary | None:
            run = case_result.runs[lib_name]
            if not isinstance(run, FixedRun):
                return None
            return run.warm_rounds[round_index]

        warm_round_lines = _format_round_medians(
            round_summaries=next(
                (
                    run.warm_rounds
                    for run in case_result.runs.values()
                    if isinstance(run, FixedRun)
                ),
                None,
            ),
            library_names=_LIBRARY_NAMES,
            resolver=resolve_warm_round,
        )
        lines.extend(warm_round_lines)
        lines.append("")
        return "\n".join(lines)

    def _render_dynamic_case_table(
        self,
        case_result: DynamicCaseResult,
        workload_comparison: DynamicWorkloadComparison,
    ) -> str:
        lines = [
            f"### {case_result.case.name}",
            "",
            case_result.case.description,
            "",
        ]
        lines.extend(
            _render_workload(
                metadata=case_result.workload,
                comparison=workload_comparison,
            )
        )
        lines.extend(
            [
                "Execution forms:",
                "",
                f"- `einf`: `{case_result.case.calls.einf}`",
                f"- `einops`: `{case_result.case.calls.einops}`",
                f"- `einx`: `{case_result.case.calls.einx}`",
                "",
                "| Library | Call observations | Median (ms) | Mean (ms) | P25 (ms) | P75 (ms) | P95 (ms) | Min (ms) | Max (ms) |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for lib_name in _LIBRARY_NAMES:
            run = case_result.runs[lib_name]
            if isinstance(run, UnavailableRun):
                lines.append(
                    f"| {lib_name} | n/a ({run.reason}) | - | - | - | - | - | - | - |"
                )
                continue
            summary = run.summary
            lines.append(
                f"| {lib_name} | {summary.count} | {summary.median_ms:.4f} | "
                f"{summary.mean_ms:.4f} | {summary.p25_ms:.4f} | {summary.p75_ms:.4f} | "
                f"{summary.p95_ms:.4f} | {summary.min_ms:.4f} | {summary.max_ms:.4f} |"
            )

        lines.extend(
            _render_paired_comparisons(
                evidence=case_result.evidence,
                heading="Paired latency ratios (competitor / einf):",
                unit_label="Paired workload units",
            )
        )

        lines.extend(
            ["", "Round base order (paired execution rotates within each round):", ""]
        )
        max_orders = 12
        for round_index, order in enumerate(
            case_result.round_orders[:max_orders], start=1
        ):
            lines.append(f"- round {round_index}: `{' -> '.join(order)}`")
        if len(case_result.round_orders) > max_orders:
            hidden = len(case_result.round_orders) - max_orders
            lines.append(f"- ... `{hidden}` additional rounds omitted for brevity")

        def resolve_dynamic_round(
            lib_name: LibraryName, round_index: int
        ) -> TimingSummary | None:
            run = case_result.runs[lib_name]
            if not isinstance(run, DynamicRun):
                return None
            return run.round_summaries[round_index]

        dynamic_round_lines = _format_round_medians(
            round_summaries=next(
                (
                    run.round_summaries
                    for run in case_result.runs.values()
                    if isinstance(run, DynamicRun)
                ),
                None,
            ),
            library_names=_LIBRARY_NAMES,
            resolver=resolve_dynamic_round,
        )
        lines.extend(dynamic_round_lines)
        lines.append("")
        return "\n".join(lines)

    def render_fixed(self, result: TestResult[FixedCaseResult]) -> str:
        """Render one fixed benchmark report."""
        lines = [result.title, "", "## Configuration", ""]
        lines.extend(f"- {entry}" for entry in result.configuration)
        lines.extend(["", "## Methodology", ""])
        lines.extend(f"- {entry}" for entry in result.methodology)
        lines.extend(["", "## Results", ""])
        for case_result in result.case_results:
            lines.append(self._render_fixed_case_table(case_result))
        lines.extend(["## Notes", ""])
        lines.extend(f"- {entry}" for entry in result.notes)
        lines.append("")
        return "\n".join(lines)

    def render_dynamic(
        self,
        result: TestResult[DynamicCaseResult],
        *,
        workload_comparisons: Mapping[str, DynamicWorkloadComparison],
    ) -> str:
        """Render one dynamic benchmark report."""
        lines = [result.title, "", "## Configuration", ""]
        lines.extend(f"- {entry}" for entry in result.configuration)
        lines.extend(["", "## Methodology", ""])
        lines.extend(f"- {entry}" for entry in result.methodology)
        lines.extend(["", "## Results", ""])
        for case_result in result.case_results:
            lines.append(
                self._render_dynamic_case_table(
                    case_result,
                    workload_comparisons[case_result.case.name],
                )
            )
        lines.extend(["## Notes", ""])
        lines.extend(f"- {entry}" for entry in result.notes)
        lines.append("")
        return "\n".join(lines)
