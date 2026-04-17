#!/usr/bin/env python3
"""Profile one warm benchmark case with cProfile call-tree output."""

import argparse
import cProfile
import json
import platform
from _lsprof import profiler_entry
from dataclasses import asdict, dataclass
from pathlib import Path
from types import CodeType
from typing import Literal

from benchmarks.profile.overhead_breakdown import (
    BackendName,
    OverheadCase,
    _dynamic_cases,
    _fixed_cases,
    _fixed_sizes,
    _touch_output,
)
from benchmarks.shared import version_or_missing

SortName = Literal["cumtime", "tottime", "ncalls"]


@dataclass(frozen=True, slots=True)
class WarmCallTreeRow:
    """One rendered cProfile row."""

    filename: str
    line: int
    function: str
    primitive_calls: int
    total_calls: int
    total_time_s: float
    cumulative_time_s: float


@dataclass(frozen=True, slots=True)
class WarmCallTreeReport:
    """Serialized warm-path cProfile report for one benchmark case."""

    backend: BackendName
    mode: str
    scale: str
    seed: int
    case_name: str
    call_repr: str
    warmup: int
    loops: int
    sort_by: SortName
    top: int
    python: str
    numpy: str
    torch: str
    einops: str
    einx: str
    rows: tuple[WarmCallTreeRow, ...]


def _cases_for(
    *,
    mode: str,
    scale: str,
    seed: int,
    backend: BackendName,
) -> tuple[OverheadCase, ...]:
    sizes = _fixed_sizes(scale)
    if mode == "fixed":
        return _fixed_cases(
            sizes=sizes,
            seed=seed,
            scale=scale,
            backend=backend,
        )
    if mode == "dynamic":
        return _dynamic_cases(
            sizes=sizes,
            seed=seed,
            scale=scale,
            backend=backend,
        )
    raise ValueError(f"unsupported mode: {mode}")


def _find_case(
    *,
    mode: str,
    scale: str,
    seed: int,
    backend: BackendName,
    case_name: str,
) -> OverheadCase:
    for case in _cases_for(mode=mode, scale=scale, seed=seed, backend=backend):
        if case.name == case_name:
            return case
    raise ValueError(f"unknown case {case_name!r} for mode={mode!r}, scale={scale!r}")


def _profile_invoke(
    *,
    invoke,
    warmup: int,
    loops: int,
) -> cProfile.Profile:
    for _ in range(warmup):
        _touch_output(invoke())
    profile = cProfile.Profile()
    profile.enable()
    for _ in range(loops):
        _touch_output(invoke())
    profile.disable()
    return profile


def _sort_key(
    entry: profiler_entry,
    /,
    *,
    sort_by: SortName,
) -> tuple[float, float, int, str, int, str]:
    filename, line, function = _entry_location(entry)
    if sort_by == "cumtime":
        primary = entry.totaltime
    elif sort_by == "tottime":
        primary = entry.inlinetime
    else:
        primary = float(entry.callcount)
    return (
        primary,
        entry.totaltime,
        entry.callcount,
        filename,
        line,
        function,
    )


def _entry_location(entry: profiler_entry, /) -> tuple[str, int, str]:
    code = entry.code
    if isinstance(code, CodeType):
        return code.co_filename, code.co_firstlineno, code.co_name
    return "<built-in>", 0, code


def _extract_rows(
    entries: list[profiler_entry],
    /,
    *,
    sort_by: SortName,
    top: int,
) -> tuple[WarmCallTreeRow, ...]:
    sorted_stats = sorted(
        entries,
        key=lambda entry: _sort_key(entry, sort_by=sort_by),
        reverse=True,
    )
    rows: list[WarmCallTreeRow] = []
    for entry in sorted_stats[:top]:
        filename, line, function = _entry_location(entry)
        rows.append(
            WarmCallTreeRow(
                filename=filename,
                line=line,
                function=function,
                primitive_calls=entry.callcount - entry.reccallcount,
                total_calls=entry.callcount,
                total_time_s=entry.inlinetime,
                cumulative_time_s=entry.totaltime,
            )
        )
    return tuple(rows)


def _build_report(
    *,
    backend: BackendName,
    mode: str,
    scale: str,
    seed: int,
    case: OverheadCase,
    warmup: int,
    loops: int,
    sort_by: SortName,
    top: int,
) -> WarmCallTreeReport:
    invoke = case.build_invoke()
    profile = _profile_invoke(
        invoke=invoke,
        warmup=warmup,
        loops=loops,
    )
    rows = _extract_rows(profile.getstats(), sort_by=sort_by, top=top)
    return WarmCallTreeReport(
        backend=backend,
        mode=mode,
        scale=scale,
        seed=seed,
        case_name=case.name,
        call_repr=case.call_repr,
        warmup=warmup,
        loops=loops,
        sort_by=sort_by,
        top=top,
        python=platform.python_version(),
        numpy=version_or_missing("numpy"),
        torch=version_or_missing("torch"),
        einops=version_or_missing("einops"),
        einx=version_or_missing("einx"),
        rows=rows,
    )


def _to_json(report: WarmCallTreeReport, /) -> dict[str, object]:
    return asdict(report)


def _to_markdown(report: WarmCallTreeReport, /) -> str:
    lines = [
        "# Warm Call Tree (einf)",
        "",
        "Warm-path cProfile report for one benchmark case after warmup.",
        "",
        "## Repro",
        "",
        "```bash",
        "PYTHONPATH=src python benchmarks/profile/warm_calltree.py \\",
        f"  --backend {report.backend} \\",
        f"  --mode {report.mode} \\",
        f"  --scale {report.scale} \\",
        f"  --case {report.case_name} \\",
        f"  --warmup {report.warmup} \\",
        f"  --loops {report.loops} \\",
        f"  --sort {report.sort_by} \\",
        f"  --top {report.top} \\",
        "  --output docs/benchmarks/2026-04-09-warm-calltree.md",
        "```",
        "",
        "## Environment",
        "",
        f"- Backend: `{report.backend}`",
        f"- Python: `{report.python}`",
        f"- NumPy: `{report.numpy}`",
        f"- torch: `{report.torch}`",
        f"- einops: `{report.einops}`",
        f"- einx: `{report.einx}`",
        "",
        "## Case",
        "",
        f"- Mode: `{report.mode}`",
        f"- Scale: `{report.scale}`",
        f"- Case: `{report.case_name}`",
        f"- Call: `{report.call_repr}`",
        f"- Warmup iterations: `{report.warmup}`",
        f"- Profiled iterations: `{report.loops}`",
        f"- Sort: `{report.sort_by}`",
        "",
        "## Top Frames",
        "",
        "| File | Line | Function | Primitive calls | Total calls | Total time (s) | Cumulative time (s) |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for row in report.rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row.filename}`",
                    str(row.line),
                    f"`{row.function}`",
                    str(row.primitive_calls),
                    str(row.total_calls),
                    f"{row.total_time_s:.6f}",
                    f"{row.cumulative_time_s:.6f}",
                ]
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Profile one warm benchmark case with cProfile.",
    )
    parser.add_argument("--backend", choices=("numpy", "torch"), default="numpy")
    parser.add_argument("--mode", choices=("fixed", "dynamic"), required=True)
    parser.add_argument("--scale", choices=("medium", "large"), required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--seed", type=int, default=20260215)
    parser.add_argument("--warmup", type=int, default=32)
    parser.add_argument("--loops", type=int, default=256)
    parser.add_argument("--sort", choices=("cumtime", "tottime", "ncalls"), default="cumtime")
    parser.add_argument("--top", type=int, default=40)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--raw-output", type=Path, default=None)
    args = parser.parse_args()

    case = _find_case(
        mode=args.mode,
        scale=args.scale,
        seed=args.seed,
        backend=args.backend,
        case_name=args.case,
    )
    report = _build_report(
        backend=args.backend,
        mode=args.mode,
        scale=args.scale,
        seed=args.seed,
        case=case,
        warmup=args.warmup,
        loops=args.loops,
        sort_by=args.sort,
        top=args.top,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(_to_markdown(report), encoding="utf-8")
    print(_to_markdown(report))
    print(f"\nWrote report: {args.output}")

    if args.raw_output is not None:
        args.raw_output.parent.mkdir(parents=True, exist_ok=True)
        args.raw_output.write_text(
            json.dumps(_to_json(report), indent=2),
            encoding="utf-8",
        )
        print(f"Wrote raw artifact: {args.raw_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
