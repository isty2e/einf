import json
from pathlib import Path

import numpy as np
import pytest

from benchmarks.guardrail.policy import (
    OVERHEAD_REPORT_SCHEMA_VERSION,
    OverheadExecutionResourcesDict,
    load_overhead_report,
)
from benchmarks.profile import overhead_breakdown
from benchmarks.profile.overhead_breakdown import (
    STAGE_TARGETS,
    STAGES,
    CaseResult,
    OverheadCase,
    ScenarioResult,
    _profile_case,
    _resolve_target,
    _to_json,
)


def _resolved_targets() -> dict[str, tuple[str, ...]]:
    return {stage: (f"einf.{stage}",) for stage in STAGES if stage != "__call__"}


def _execution_resources() -> OverheadExecutionResourcesDict:
    return {
        "process_cpu_affinity": None,
        "native_threadpools": [
            {
                "user_api": "blas",
                "internal_api": "openblas",
                "prefix": "libopenblas",
                "num_threads": 10,
                "version": "0.3.30",
                "threading_layer": "openmp",
                "architecture": "VORTEX",
            }
        ],
    }


def test_overhead_stage_targets_resolve() -> None:
    for targets in STAGE_TARGETS.values():
        for target in targets:
            _resolve_target(target)


def test_expected_einf_source_root_is_checked_before_profiling(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "overhead_breakdown",
            "--expect-einf-source-root",
            str(tmp_path),
        ],
    )

    def reject_source(checkout_root: Path) -> None:
        assert checkout_root == tmp_path
        raise RuntimeError("wrong source")

    monkeypatch.setattr(
        overhead_breakdown,
        "require_einf_source_root",
        reject_source,
    )
    monkeypatch.setattr(
        overhead_breakdown,
        "_resolved_stage_target_names",
        lambda: pytest.fail("profiling preflight must not run"),
    )

    with pytest.raises(RuntimeError, match="wrong source"):
        overhead_breakdown.main()


def test_receipt_sources_must_remain_stable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def require_subject_content(expected: str) -> None:
        if expected != "subject-after":
            raise RuntimeError("imported einf source changed during measurement")

    monkeypatch.setattr(
        overhead_breakdown,
        "_harness_source_sha256",
        lambda: "harness-after",
    )
    monkeypatch.setattr(
        overhead_breakdown,
        "require_stable_einf_source_content",
        require_subject_content,
    )

    overhead_breakdown._require_stable_receipt_sources(
        harness_source_sha256="harness-after",
        subject_content_sha256="subject-after",
    )
    with pytest.raises(RuntimeError, match="profiler source changed"):
        overhead_breakdown._require_stable_receipt_sources(
            harness_source_sha256="harness-before",
            subject_content_sha256="subject-after",
        )
    with pytest.raises(RuntimeError, match="einf source changed"):
        overhead_breakdown._require_stable_receipt_sources(
            harness_source_sha256="harness-after",
            subject_content_sha256="subject-before",
        )


def test_execution_resources_capture_affinity_and_effective_threads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        overhead_breakdown,
        "_process_cpu_affinity",
        lambda: [1, 3],
    )
    monkeypatch.setattr(
        overhead_breakdown,
        "threadpool_info",
        lambda: [
            {
                "user_api": "blas",
                "internal_api": "openblas",
                "prefix": "libopenblas",
                "num_threads": 2,
                "version": "0.3.30",
                "threading_layer": "openmp",
                "architecture": "VORTEX",
            },
            {
                "user_api": "openmp",
                "internal_api": "openmp",
                "prefix": "libomp",
                "num_threads": 8,
                "version": None,
                "threading_layer": None,
                "architecture": None,
            },
        ],
    )

    class Torch:
        @staticmethod
        def get_num_threads() -> int:
            return 4

        @staticmethod
        def get_num_interop_threads() -> int:
            return 2

    monkeypatch.setattr(overhead_breakdown, "torch", Torch)

    resources = overhead_breakdown._execution_resources_metadata("torch")
    numpy_resources = overhead_breakdown._execution_resources_metadata("numpy")

    assert resources["process_cpu_affinity"] == [1, 3]
    assert [pool["user_api"] for pool in resources["native_threadpools"]] == [
        "blas",
        "openmp",
    ]
    assert resources.get("torch_threads") == {"intra_op": 4, "inter_op": 2}
    assert [pool["user_api"] for pool in numpy_resources["native_threadpools"]] == [
        "blas"
    ]
    assert "torch_threads" not in numpy_resources


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

    payload = _to_json(
        result,
        backend="numpy",
        seed=20260215,
        resolved_stage_targets=_resolved_targets(),
        harness_source_sha256="0" * 64,
        subject_source={
            "kind": "git_checkout",
            "distribution_version": "0.2.0.dev1",
            "git_revision": "a" * 40,
            "git_dirty": False,
            "content_sha256": "1" * 64,
        },
        execution_resources=_execution_resources(),
    )
    path = tmp_path / "report.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    loaded = load_overhead_report(path)
    assert loaded["schema_version"] == OVERHEAD_REPORT_SCHEMA_VERSION
    assert loaded["meta"]["execution_target"] == {
        "backend": "numpy",
        "requested_device": "cpu",
        "resolved_device": "cpu",
    }
    assert loaded["meta"]["seed"] == 20260215
    assert len(loaded["meta"]["harness_source_sha256"]) == 64
    assert len(loaded["meta"]["subject_source"]["content_sha256"]) == 64
    assert "torch" not in loaded["meta"]["environment"]
    case = loaded["scenarios"][0]["cases"][0]
    assert case.get("residual_ms_per_call") == 0.2


def test_guardrail_loader_accepts_residual_field(tmp_path: Path) -> None:
    payload = {
        "schema_version": OVERHEAD_REPORT_SCHEMA_VERSION,
        "meta": {
            "capture_id": "00000000-0000-4000-8000-000000000000",
            "harness_source_sha256": "0" * 64,
            "subject_source": {
                "kind": "git_checkout",
                "distribution_version": "0.2.0.dev1",
                "git_revision": "a" * 40,
                "git_dirty": False,
                "content_sha256": "1" * 64,
            },
            "execution_target": {
                "backend": "numpy",
                "requested_device": "cpu",
                "resolved_device": "cpu",
            },
            "host": {
                "system": "Darwin",
                "machine": "arm64",
                "cpu_model": "Apple M1 Pro",
                "logical_cpu_count": 10,
            },
            "execution_resources": _execution_resources(),
            "environment": {
                "python": "3.11",
                "numpy": "1.26",
                "array_api_compat": "1.12",
                "opt_einsum": "3.4",
            },
            "seed": 20260215,
            "stages": ["__call__", "solve", "runner_resolve", "fusion", "kernel"],
            "resolved_stage_targets": {
                "solve": ["einf.solve"],
                "runner_resolve": ["einf.runner_resolve"],
                "fusion": ["einf.fusion"],
                "kernel": ["einf.kernel"],
            },
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
