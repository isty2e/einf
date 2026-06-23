import json
from pathlib import Path

import numpy as np
import pytest

from benchmarks.guardrail.policy import load_overhead_report
from benchmarks.profile.overhead_breakdown import (
    STAGE_TARGETS,
    CaseResult,
    OverheadCase,
    ScenarioResult,
    _profile_case,
    _resolve_target,
    _to_json,
)


def test_overhead_stage_targets_resolve() -> None:
    for targets in STAGE_TARGETS.values():
        for target in targets:
            _resolve_target(target)


def test_overhead_json_emits_residual_field(tmp_path: Path) -> None:
    result = (
        ScenarioResult(
            scenario="fixed-medium",
            mode="fixed",
            scale="medium",
            cases=(
                CaseResult(
                    name="rearrange_flatten",
                    call_repr="rearrange(...)",
                    loops=100,
                    unpatched_call_ms=1.0,
                    instrumented_call_ms=1.1,
                    stage_ms_per_call={
                        "input_shape": 0.05,
                        "solve": 0.1,
                        "normalize_context": 0.0,
                        "backend_checks": 0.0,
                        "plan_select": 0.0,
                        "step_specialize": 0.0,
                        "runner_resolve": 0.1,
                        "fusion": 0.0,
                        "step_run": 0.05,
                        "primitive": 0.1,
                        "kernel": 0.6,
                    },
                    residual_ms_per_call=0.2,
                ),
            ),
        ),
    )

    payload = _to_json(result, backend="numpy")
    path = tmp_path / "report.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    loaded = load_overhead_report(path)
    case = loaded["scenarios"][0]["cases"][0]
    assert case.get("residual_ms_per_call") == 0.2


def test_guardrail_loader_accepts_residual_field(tmp_path: Path) -> None:
    payload = {
        "meta": {
            "backend": "numpy",
            "python": "3.11",
            "numpy": "1.26",
            "torch": "not-installed",
            "einops": "not-installed",
            "einx": "not-installed",
            "stages": ["__call__", "solve", "runner_resolve", "fusion", "kernel"],
        },
        "scenarios": [
            {
                "scenario": "fixed-medium",
                "mode": "fixed",
                "scale": "medium",
                "cases": [
                    {
                        "name": "rearrange_flatten",
                        "call_repr": "rearrange(...)",
                        "loops": 100,
                        "unpatched_call_ms": 1.0,
                        "instrumented_call_ms": 1.1,
                        "residual_ms_per_call": 0.2,
                        "stage_ms_per_call": {
                            "solve": 0.1,
                            "runner_resolve": 0.1,
                            "fusion": 0.0,
                            "kernel": 0.7,
                        },
                    }
                ],
            }
        ],
    }
    path = tmp_path / "report.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    loaded = load_overhead_report(path)
    case = loaded["scenarios"][0]["cases"][0]
    assert case.get("residual_ms_per_call") == 0.2


def test_profile_case_builds_fresh_invoke_for_each_timing_pass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    build_tokens: list[int] = []
    call_tokens: list[int] = []

    def build_invoke():
        token = len(build_tokens) + 1
        build_tokens.append(token)

        def invoke() -> np.ndarray:
            call_tokens.append(token)
            return np.zeros((1,), dtype=np.float32)

        return invoke

    def fake_measure_call_median_ms(
        *,
        invoke,
        loops: int,
        warmup: int,
        repeats: int,
    ) -> float:
        _ = loops, warmup, repeats
        invoke()
        return 1.0

    def fake_measure_stage_ms_per_call(
        *,
        invoke,
        loops: int,
        warmup: int,
    ) -> tuple[dict[str, float], float, float]:
        _ = loops, warmup
        invoke()
        return {}, 1.1, 0.1

    monkeypatch.setattr(
        "benchmarks.profile.overhead_breakdown._measure_call_median_ms",
        fake_measure_call_median_ms,
    )
    monkeypatch.setattr(
        "benchmarks.profile.overhead_breakdown._measure_stage_ms_per_call",
        fake_measure_stage_ms_per_call,
    )

    case = OverheadCase(
        name="fake",
        call_repr="fake()",
        build_invoke=build_invoke,
        loops=10,
    )

    result = _profile_case(case)

    assert result.unpatched_call_ms == 1.0
    assert result.instrumented_call_ms == 1.1
    assert result.residual_ms_per_call == 0.1
    assert build_tokens == [1, 2]
    assert call_tokens == [1, 2]
