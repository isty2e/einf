#!/usr/bin/env python3
"""Measure `einf-lsp` semantic-analysis and cached feature latency."""

import argparse
import asyncio
import json
import platform
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import TypeAlias

from lsprotocol import types as lsp

from benchmarks.shared.metadata import version_or_missing
from einf.analysis.checkers import (
    CheckerExecutionPolicy,
    CheckerExecutor,
    CheckerResult,
    build_checker_adapters,
)
from einf.analysis.lsp import LspService, encode_semantic_tokens
from einf.analysis.lsp.checker_coordinator import (
    DocumentCheckerRequest,
    LspCheckerCoordinator,
)
from einf.analysis.lsp.position_codec import LspPositionCodec
from einf.analysis.model import TextPosition
from einf.analysis.report import AnalysisFileReport

JsonValue: TypeAlias = (
    str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]
)


@dataclass(frozen=True, slots=True)
class TimingSummary:
    """Distribution summary for one latency sample set."""

    count: int
    median_ms: float
    p25_ms: float
    p75_ms: float
    p95_ms: float
    max_ms: float


@dataclass(frozen=True, slots=True)
class LspLatencyCase:
    """One source shape measured by the LSP latency smoke."""

    name: str
    source: str


@dataclass(frozen=True, slots=True)
class LspCaseLatency:
    """Latency summaries for one source case."""

    name: str
    source_bytes: int
    axis_tokens: int
    diagnostics: int
    failures: int
    open_document: TimingSummary
    change_document: TimingSummary
    semantic_tokens: TimingSummary
    inlay_hints: TimingSummary | None
    hover: TimingSummary | None
    save_with_checkers: TimingSummary | None


@dataclass(frozen=True, slots=True)
class LspLatencyReport:
    """Serialized `einf-lsp` latency smoke report."""

    parser: str
    repeats: int
    checkers: tuple[str, ...]
    python: str
    lsprotocol: str
    pygls: str
    cases: tuple[LspCaseLatency, ...]


def _percentile(sorted_values: list[float], fraction: float) -> float:
    if not sorted_values:
        raise ValueError("cannot summarize an empty sample set")
    position = (len(sorted_values) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def _summarize(samples: list[float]) -> TimingSummary:
    values = sorted(samples)
    return TimingSummary(
        count=len(values),
        median_ms=_percentile(values, 0.5),
        p25_ms=_percentile(values, 0.25),
        p75_ms=_percentile(values, 0.75),
        p95_ms=_percentile(values, 0.95),
        max_ms=values[-1],
    )


def _measure(repeats: int, invoke: Callable[[], None]) -> TimingSummary:
    samples: list[float] = []
    for _ in range(repeats):
        started = perf_counter()
        invoke()
        samples.append((perf_counter() - started) * 1000.0)
    return _summarize(samples)


def _contract_source(calls: int) -> str:
    block = (
        "from einf import ax, axes, contract\n"
        'i, k, j = axes("i", "k", "j")\n'
        "contract((ax[i, k], ax[k, j]), ax[i, j])\n"
    )
    return "\n".join(block for _ in range(calls))


def _default_cases() -> tuple[LspLatencyCase, ...]:
    return (
        LspLatencyCase(
            name="small_valid",
            source=(
                "from einf import ax, axes, rearrange\n"
                'b = axes("b")[0]\n'
                "rearrange(ax[b], ax[b])\n"
            ),
        ),
        LspLatencyCase(name="medium_contracts", source=_contract_source(40)),
        LspLatencyCase(name="token_heavy_contracts", source=_contract_source(200)),
        LspLatencyCase(
            name="irrelevant_python",
            source="\n".join(f"value_{i} = {i}" for i in range(300)),
        ),
    )


def _first_hover_position(report: AnalysisFileReport) -> TextPosition:
    if not report.axis_tokens:
        return TextPosition(line=1, column=0)
    return report.axis_tokens[0].span.start


def _measure_case(
    case: LspLatencyCase,
    *,
    parser: str,
    repeats: int,
    checkers: tuple[str, ...],
    work_dir: Path,
) -> LspCaseLatency:
    path = (work_dir / f"{case.name}.py").resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(case.source, encoding="utf-8")
    uri = path.as_uri()

    open_service = LspService(parser)

    def open_document() -> None:
        _ = open_service.open_document(uri=uri, source=case.source, version=None)

    open_summary = _measure(repeats, open_document)

    change_service = LspService(parser)
    state = change_service.open_document(uri=uri, source=case.source, version=0)
    position_codec = LspPositionCodec(
        lines=state.source_lines,
        encoding=lsp.PositionEncodingKind.Utf16,
    )
    change_version = 0

    def change_document() -> None:
        nonlocal change_version, state
        change_version += 1
        state = change_service.change_document(
            uri=uri,
            source=case.source,
            version=change_version,
        )

    change_summary = _measure(repeats, change_document)

    def encode_tokens() -> None:
        _ = encode_semantic_tokens(
            state.report.axis_tokens,
            position_codec=position_codec,
        )

    semantic_summary = _measure(repeats, encode_tokens)
    inlay_summary, hover_summary = _measure_optional_feature_latency(
        report=state.report,
        position_codec=position_codec,
        repeats=repeats,
    )

    checker_summary = None
    if checkers:
        checker_summary = _measure_save_with_checkers(
            parser=parser,
            checkers=checkers,
            path=path,
            source=case.source,
            repeats=repeats,
        )

    return LspCaseLatency(
        name=case.name,
        source_bytes=len(case.source.encode("utf-8")),
        axis_tokens=len(state.report.axis_tokens),
        diagnostics=len(state.report.diagnostics),
        failures=len(state.report.failures),
        open_document=open_summary,
        change_document=change_summary,
        semantic_tokens=semantic_summary,
        inlay_hints=inlay_summary,
        hover=hover_summary,
        save_with_checkers=checker_summary,
    )


def _measure_save_with_checkers(
    *,
    parser: str,
    checkers: tuple[str, ...],
    path: Path,
    source: str,
    repeats: int,
) -> TimingSummary:
    async def scenario() -> TimingSummary:
        service = LspService(parser)
        coordinator = LspCheckerCoordinator(
            adapters=build_checker_adapters(checkers),
            executor=CheckerExecutor(CheckerExecutionPolicy()),
        )
        uri = path.as_uri()
        samples: list[float] = []

        def commit(
            request: DocumentCheckerRequest,
            result: CheckerResult,
        ) -> bool:
            current = service.get_document_state(uri=request.uri)
            if current is None or current.version != request.version:
                return False
            service.commit_document_state(current.with_checker_result(result))
            return True

        try:
            for version in range(repeats):
                started = perf_counter()
                state = service.change_document(
                    uri=uri,
                    source=source,
                    version=version,
                )
                await coordinator.check(
                    DocumentCheckerRequest(
                        uri=uri,
                        path=path,
                        version=state.version,
                    ),
                    commit=commit,
                )
                samples.append((perf_counter() - started) * 1000.0)
        finally:
            await coordinator.close()
        return _summarize(samples)

    return asyncio.run(scenario())


def _measure_optional_feature_latency(
    *,
    report: AnalysisFileReport,
    position_codec: LspPositionCodec,
    repeats: int,
) -> tuple[TimingSummary | None, TimingSummary | None]:
    try:
        from einf.analysis.lsp.hover import build_hover
        from einf.analysis.lsp.inlay_hints import build_inlay_hints
    except ModuleNotFoundError:
        return None, None

    hover_position = _first_hover_position(report)

    def build_inlay() -> None:
        _ = build_inlay_hints(
            axis_tokens=report.axis_tokens,
            visible_range=None,
            position_codec=position_codec,
        )

    def build_hover_at_position() -> None:
        _ = build_hover(
            axis_tokens=report.axis_tokens,
            position=hover_position,
        )

    return (
        _measure(repeats, build_inlay),
        _measure(repeats, build_hover_at_position),
    )


def _build_report(
    *,
    parser: str,
    repeats: int,
    checkers: tuple[str, ...],
    work_dir: Path,
) -> LspLatencyReport:
    cases = tuple(
        _measure_case(
            case,
            parser=parser,
            repeats=repeats,
            checkers=checkers,
            work_dir=work_dir,
        )
        for case in _default_cases()
    )
    return LspLatencyReport(
        parser=parser,
        repeats=repeats,
        checkers=checkers,
        python=platform.python_version(),
        lsprotocol=version_or_missing("lsprotocol"),
        pygls=version_or_missing("pygls"),
        cases=cases,
    )


def _to_json(report: LspLatencyReport) -> JsonValue:
    return {
        "parser": report.parser,
        "repeats": report.repeats,
        "checkers": list(report.checkers),
        "python": report.python,
        "lsprotocol": report.lsprotocol,
        "pygls": report.pygls,
        "cases": [_case_to_json(case) for case in report.cases],
    }


def _summary_to_json(summary: TimingSummary | None) -> JsonValue:
    if summary is None:
        return None
    return {
        "count": summary.count,
        "median_ms": summary.median_ms,
        "p25_ms": summary.p25_ms,
        "p75_ms": summary.p75_ms,
        "p95_ms": summary.p95_ms,
        "max_ms": summary.max_ms,
    }


def _case_to_json(case: LspCaseLatency) -> JsonValue:
    return {
        "name": case.name,
        "source_bytes": case.source_bytes,
        "axis_tokens": case.axis_tokens,
        "diagnostics": case.diagnostics,
        "failures": case.failures,
        "open_document": _summary_to_json(case.open_document),
        "change_document": _summary_to_json(case.change_document),
        "semantic_tokens": _summary_to_json(case.semantic_tokens),
        "inlay_hints": _summary_to_json(case.inlay_hints),
        "hover": _summary_to_json(case.hover),
        "save_with_checkers": _summary_to_json(case.save_with_checkers),
    }


def _format_summary(summary: TimingSummary | None) -> str:
    if summary is None:
        return "n/a"
    return f"{summary.median_ms:.3f} [{summary.p25_ms:.3f} - {summary.p75_ms:.3f}]"


def _to_markdown(report: LspLatencyReport) -> str:
    lines = [
        "# LSP Latency Smoke",
        "",
        "## Configuration",
        "",
        f"- Python: `{report.python}`",
        f"- parser: `{report.parser}`",
        f"- repeats: `{report.repeats}`",
        f"- checkers: `{', '.join(report.checkers) if report.checkers else 'none'}`",
        f"- lsprotocol: `{report.lsprotocol}`",
        f"- pygls: `{report.pygls}`",
        "- table units: `ms`, median `[p25 - p75]`",
        "",
        "## Results",
        "",
        "| Case | Bytes | Tokens | Diagnostics | Open | Change | Semantic tokens | Inlay hints | Hover | Save + checkers |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for case in report.cases:
        lines.append(
            "| "
            f"{case.name} | "
            f"{case.source_bytes} | "
            f"{case.axis_tokens} | "
            f"{case.diagnostics + case.failures} | "
            f"{_format_summary(case.open_document)} | "
            f"{_format_summary(case.change_document)} | "
            f"{_format_summary(case.semantic_tokens)} | "
            f"{_format_summary(case.inlay_hints)} | "
            f"{_format_summary(case.hover)} | "
            f"{_format_summary(case.save_with_checkers)} |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `Open` and `Change` include semantic parsing and analysis.",
            "- `Semantic tokens`, `Inlay hints`, and `Hover` use cached document state.",
            "- `Save + checkers` is present only when `--checker` is provided.",
        ]
    )
    return "\n".join(lines) + "\n"


def _write_text_output(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parser", choices=("ast", "libcst"), default="ast")
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--checker", action="append", default=[])
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path("/tmp/einf-lsp-latency"),
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--json-output", type=Path)
    return parser.parse_args()


def main() -> int:
    """Run the LSP latency smoke benchmark."""
    args = _parse_args()
    report = _build_report(
        parser=args.parser,
        repeats=args.repeats,
        checkers=tuple(args.checker),
        work_dir=args.work_dir,
    )
    markdown = _to_markdown(report)
    if args.output is not None:
        _write_text_output(args.output, markdown)
    else:
        print(markdown, end="")
    if args.json_output is not None:
        _write_text_output(
            args.json_output,
            json.dumps(_to_json(report), indent=2, sort_keys=True),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
