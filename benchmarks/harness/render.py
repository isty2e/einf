from dataclasses import dataclass

from .result import CaseResult, TestResult, TimingSummary


def _format_summary(summary: TimingSummary | None) -> str:
    if summary is None:
        return "n/a | n/a | n/a | n/a | n/a | n/a"
    return (
        f"{summary.count} | {summary.p25_ms:.4f} | {summary.median_ms:.4f} | "
        f"{summary.p75_ms:.4f} | {summary.iqr_ms:.4f} | {summary.p95_ms:.4f}"
    )


def _format_round_medians(
    *,
    round_summaries: tuple[TimingSummary, ...] | None,
    library_names: tuple[str, ...],
    resolver,
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


@dataclass(frozen=True, slots=True)
class MarkdownPrinter:
    """Render benchmark test results to markdown."""

    def _render_fixed_case_table(self, case_result: CaseResult) -> str:
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
        for lib_name in ("einf", "einops", "einx"):
            run = case_result.runs[lib_name]
            if not run.available:
                lines.append(
                    f"| {lib_name} | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a ({run.reason}) |"
                )
                continue
            lines.append(
                f"| {lib_name} | {_format_summary(run.cold)} | {_format_summary(run.warm)} |"
            )

        def resolve_warm_round(lib_name: str, round_index: int) -> TimingSummary | None:
            run = case_result.runs[lib_name]
            if not run.available or run.warm_rounds is None:
                return None
            return run.warm_rounds[round_index]

        warm_round_lines = _format_round_medians(
            round_summaries=next(
                (
                    run.warm_rounds
                    for run in case_result.runs.values()
                    if run.available and run.warm_rounds is not None
                ),
                None,
            ),
            library_names=("einf", "einops", "einx"),
            resolver=resolve_warm_round,
        )
        lines.extend(warm_round_lines)
        lines.append("")
        return "\n".join(lines)

    def _render_dynamic_case_table(self, case_result: CaseResult) -> str:
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
            "| Library | Samples | Median (ms) | Mean (ms) | P25 (ms) | P75 (ms) | P95 (ms) | Min (ms) | Max (ms) |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for lib_name in ("einf", "einops", "einx"):
            run = case_result.runs[lib_name]
            if not run.available:
                lines.append(
                    f"| {lib_name} | n/a ({run.reason}) | - | - | - | - | - | - | - |"
                )
                continue
            summary = run.dynamic
            if summary is None:
                lines.append(f"| {lib_name} | n/a | - | - | - | - | - | - | - |")
                continue
            lines.append(
                f"| {lib_name} | {summary.count} | {summary.median_ms:.4f} | "
                f"{summary.mean_ms:.4f} | {summary.p25_ms:.4f} | {summary.p75_ms:.4f} | "
                f"{summary.p95_ms:.4f} | {summary.min_ms:.4f} | {summary.max_ms:.4f} |"
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
            lib_name: str, round_index: int
        ) -> TimingSummary | None:
            run = case_result.runs[lib_name]
            if not run.available or run.dynamic_rounds is None:
                return None
            return run.dynamic_rounds[round_index]

        dynamic_round_lines = _format_round_medians(
            round_summaries=next(
                (
                    run.dynamic_rounds
                    for run in case_result.runs.values()
                    if run.available and run.dynamic_rounds is not None
                ),
                None,
            ),
            library_names=("einf", "einops", "einx"),
            resolver=resolve_dynamic_round,
        )
        lines.extend(dynamic_round_lines)
        lines.append("")
        return "\n".join(lines)

    def render_fixed(self, result: TestResult) -> str:
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

    def render_dynamic(self, result: TestResult) -> str:
        """Render one dynamic benchmark report."""
        lines = [result.title, "", "## Configuration", ""]
        lines.extend(f"- {entry}" for entry in result.configuration)
        lines.extend(["", "## Methodology", ""])
        lines.extend(f"- {entry}" for entry in result.methodology)
        lines.extend(["", "## Results", ""])
        for case_result in result.case_results:
            lines.append(self._render_dynamic_case_table(case_result))
        lines.extend(["## Notes", ""])
        lines.extend(f"- {entry}" for entry in result.notes)
        lines.append("")
        return "\n".join(lines)
