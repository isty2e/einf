import json
import sys

import pytest

from benchmarks.compare.einf_einops_einx import (
    _receipt_payload as _fixed_receipt_payload,
)
from benchmarks.compare.einf_einops_einx import (
    main as fixed_main,
)
from benchmarks.compare.einf_einops_einx_dynamic import (
    _receipt_payload,
)
from benchmarks.harness import (
    AvailableRun,
    BackendSpec,
    BenchmarkCase,
    BenchSizes,
    CaseCalls,
    DynamicCaseResult,
    DynamicShapeWorkload,
    DynamicTaskConfig,
    DynamicWorkloadMetadata,
    FixedCaseResult,
    FixedTaskConfig,
    LibraryTimingObservation,
    PairedComparison,
    PairedEvidence,
    TimingSummary,
)
from benchmarks.harness.comparison import compare_library_timings
from benchmarks.harness.result import DynamicInputUnit
from benchmarks.harness.types import LibraryName


def _observations(
    values: tuple[tuple[int, int, int, float, float], ...],
) -> tuple[LibraryTimingObservation, ...]:
    observations: list[LibraryTimingObservation] = []
    for round_index, batch_index, repeat_index, einf_ms, einops_ms in values:
        latency_by_library: tuple[tuple[LibraryName, float], ...] = (
            ("einf", einf_ms),
            ("einops", einops_ms),
        )
        for order_position, (library, latency_ms) in enumerate(latency_by_library):
            observations.append(
                LibraryTimingObservation(
                    round_index=round_index,
                    unit_index=batch_index,
                    repeat_index=repeat_index,
                    library=library,
                    order_position=order_position,
                    latency_ms=latency_ms,
                )
            )
    return tuple(observations)


def _summary(*, mean_ms: float) -> TimingSummary:
    return TimingSummary(
        count=2,
        p25_ms=mean_ms,
        median_ms=mean_ms,
        p75_ms=mean_ms,
        iqr_ms=0.0,
        p95_ms=mean_ms,
        mean_ms=mean_ms,
        min_ms=mean_ms,
        max_ms=mean_ms,
    )


def _case() -> BenchmarkCase:
    return BenchmarkCase(
        name="dynamic_case",
        description="demo",
        calls=CaseCalls(einf="a()", einops="b()", einx="c()"),
        reference=lambda inputs: inputs[0],
        make_einf_runner=lambda: lambda inputs: inputs[0],
        make_einops_runner=lambda: lambda inputs: inputs[0],
        make_einx_runner=lambda: lambda inputs: inputs[0],
    )


def _workload() -> tuple[BenchSizes, DynamicWorkloadMetadata]:
    sizes = BenchSizes(b=1, n=2, d=3, h=4, w=5, r=6, j=7)
    workload = DynamicShapeWorkload(
        sampled_dimensions=("b",),
        input_shapes=lambda dimensions: ((dimensions["b"], dimensions["d"]),),
        output_shapes=lambda dimensions: (
            (dimensions["b"], dimensions["d"], dimensions["r"]),
        ),
    )
    return sizes, workload.metadata(sizes=sizes)


def test_paired_comparison_aggregates_repeats_at_batch_level() -> None:
    observations = _observations(
        (
            (0, 0, 0, 1.0, 2.0),
            (0, 0, 1, 3.0, 6.0),
            (0, 1, 0, 2.0, 6.0),
            (0, 1, 1, 2.0, 6.0),
            (1, 0, 0, 4.0, 4.0),
            (1, 0, 1, 4.0, 4.0),
            (1, 1, 0, 8.0, 16.0),
            (1, 1, 1, 8.0, 16.0),
        )
    )

    first = compare_library_timings(
        observations=observations,
        libraries=("einf", "einops"),
        baseline="einf",
        bootstrap_seed=17,
    )
    second = compare_library_timings(
        observations=tuple(reversed(observations)),
        libraries=("einf", "einops"),
        baseline="einf",
        bootstrap_seed=17,
    )

    assert first == second
    assert len(first) == 1
    comparison = first[0]
    assert comparison.call_pair_count == 8
    assert comparison.paired_unit_count == 4
    assert comparison.latency_ratio == pytest.approx(1.875)
    assert comparison.confidence_interval_low > 0.0
    assert comparison.confidence_interval_high >= comparison.confidence_interval_low


def test_paired_comparison_rejects_insufficient_units_per_round() -> None:
    observations = _observations(
        (
            (0, 0, 0, 1.0, 2.0),
            (1, 0, 0, 9.0, 9.0),
        )
    )

    with pytest.raises(
        ValueError,
        match="at least two measured units per round",
    ):
        compare_library_timings(
            observations=observations,
            libraries=("einf", "einops"),
            baseline="einf",
            bootstrap_seed=3,
        )


def test_paired_comparison_rejects_incomplete_call_pair() -> None:
    observations = _observations(((0, 0, 0, 1.0, 2.0),))

    with pytest.raises(ValueError, match="incomplete paired timing observation"):
        compare_library_timings(
            observations=observations[:-1],
            libraries=("einf", "einops"),
            baseline="einf",
            bootstrap_seed=0,
        )


def test_paired_comparison_rejects_duplicate_call_identity() -> None:
    observations = _observations(((0, 0, 0, 1.0, 2.0),))

    with pytest.raises(ValueError, match="duplicate paired timing observation"):
        compare_library_timings(
            observations=observations + observations[:1],
            libraries=("einf", "einops"),
            baseline="einf",
            bootstrap_seed=0,
        )


def test_paired_comparison_rejects_duplicate_members() -> None:
    observations = _observations(((0, 0, 0, 1.0, 2.0),))

    with pytest.raises(ValueError, match="members must be unique"):
        compare_library_timings(
            observations=observations,
            libraries=("einf", "einf"),
            baseline="einf",
            bootstrap_seed=0,
        )


def test_dynamic_receipt_preserves_observation_and_analysis_identity() -> None:
    sizes, workload = _workload()
    workload_comparison = workload.compare_to(
        workload,
        scale="medium",
        reference_scale="medium",
    )
    observation = LibraryTimingObservation(
        round_index=1,
        unit_index=2,
        repeat_index=3,
        library="einf",
        order_position=0,
        latency_ms=1.25,
    )
    comparison: PairedComparison[LibraryName] = PairedComparison(
        baseline="einf",
        competitor="einops",
        call_pair_count=8,
        paired_unit_count=4,
        latency_ratio=1.5,
        confidence_level=0.95,
        confidence_interval_low=1.2,
        confidence_interval_high=1.8,
        bootstrap_resamples=10_000,
        bootstrap_seed=9,
    )
    result = DynamicCaseResult(
        case=_case(),
        workload=workload,
        realized_units=(
            DynamicInputUnit(
                round_index=1,
                unit_index=2,
                stream_index=3,
                seed=11,
                input_shapes=((1, 3),),
            ),
        ),
        runs={
            "einf": AvailableRun(
                summary=_summary(mean_ms=1.0),
                round_summaries=(_summary(mean_ms=1.0),),
            ),
            "einops": AvailableRun(
                summary=_summary(mean_ms=1.5),
                round_summaries=(_summary(mean_ms=1.5),),
            ),
            "einx": AvailableRun(
                summary=_summary(mean_ms=2.0),
                round_summaries=(_summary(mean_ms=2.0),),
            ),
        },
        round_orders=[("einf", "einops", "einx")],
        evidence=PairedEvidence(
            observations=(observation,),
            comparisons=(comparison,),
        ),
    )
    config = DynamicTaskConfig(
        scale="medium",
        seed=5,
        batches=4,
        warmup_batches=1,
        repeats=2,
        rounds=1,
        round_order_seed=7,
    )

    payload = _receipt_payload(
        config=config,
        sizes=sizes,
        case_results=[result],
        workload_comparisons={"dynamic_case": workload_comparison},
        backend=BackendSpec(name="numpy"),
    )

    assert payload["schema_version"] == 6
    assert payload["execution_target"] == {
        "backend": "numpy",
        "requested_device": "cpu",
        "resolved_device": "cpu",
    }
    measurement_contract = payload["measurement_contract"]
    assert isinstance(measurement_contract, dict)
    assert (
        measurement_contract["synchronization"]
        == "before_timer_start_and_before_timer_stop"
    )
    assert (
        measurement_contract["input_pairing"]
        == "same_prepared_tensor_objects_per_coordinate"
    )
    cases = payload["cases"]
    assert isinstance(cases, list)
    json.dumps(payload)
    case_payload = cases[0]
    assert case_payload["case"]["name"] == "dynamic_case"
    assert case_payload["realized_input_units"] == [
        {
            "round_index": 1,
            "unit_index": 2,
            "stream_index": 3,
            "seed": 11,
            "input_shapes": [[1, 3]],
        }
    ]
    assert case_payload["validation"] == {
        "scope": "all_distinct_measured_coordinates",
        "coordinate_source": "realized_input_units",
        "coordinate_count": 1,
        "members": ["einf"],
        "executions_per_member_per_coordinate": 1,
        "phase": "after_measurement_before_receipt",
        "input_replay": "deterministic_seed_regeneration",
        "output_check": "numerical_reference_match",
    }
    assert case_payload["workload"] == {
        "dimensions": [
            {
                "name": "b",
                "mode": "sampled",
                "base": 1,
                "minimum": 1,
                "maximum": 1,
            },
            {
                "name": "d",
                "mode": "fixed",
                "base": 3,
                "minimum": 3,
                "maximum": 3,
            },
            {
                "name": "r",
                "mode": "fixed",
                "base": 6,
                "minimum": 6,
                "maximum": 6,
            },
        ],
        "base_input_shapes": [[1, 3]],
        "base_output_shapes": [[1, 3, 6]],
        "base_input_elements": 3,
        "base_output_elements": 18,
        "scale_comparison": {
            "scale": "medium",
            "reference_scale": "medium",
            "dimension_ratios": [
                {"name": "b", "numerator": 1, "denominator": 1},
                {"name": "d", "numerator": 1, "denominator": 1},
                {"name": "r", "numerator": 1, "denominator": 1},
            ],
            "base_input_elements_ratio": {"numerator": 1, "denominator": 1},
            "base_output_elements_ratio": {"numerator": 1, "denominator": 1},
        },
    }
    measurement = case_payload["measurements"][0]
    assert measurement["phase"] == "steady"
    assert measurement["observations"] == [
        {
            "round_index": 1,
            "unit_index": 2,
            "repeat_index": 3,
            "library": "einf",
            "order_position": 0,
            "latency_ms": 1.25,
        }
    ]
    assert measurement["comparisons"][0]["paired_unit_count"] == 4


def test_fixed_receipt_uses_the_same_paired_evidence_shape() -> None:
    sizes, _ = _workload()
    observation = LibraryTimingObservation(
        round_index=0,
        unit_index=1,
        repeat_index=2,
        library="einf",
        order_position=0,
        latency_ms=1.25,
    )
    comparison: PairedComparison[LibraryName] = PairedComparison(
        baseline="einf",
        competitor="einops",
        call_pair_count=8,
        paired_unit_count=4,
        latency_ratio=1.5,
        confidence_level=0.95,
        confidence_interval_low=1.2,
        confidence_interval_high=1.8,
        bootstrap_resamples=10_000,
        bootstrap_seed=9,
    )
    evidence = PairedEvidence(
        observations=(observation,),
        comparisons=(comparison,),
    )
    result = FixedCaseResult(
        case=_case(),
        runs={
            library: AvailableRun(
                summary=_summary(mean_ms=1.0),
                round_summaries=(_summary(mean_ms=1.0),),
            )
            for library in ("einf", "einops", "einx")
        },
        round_orders=[("einf", "einops", "einx")],
        evidence=evidence,
    )
    config = FixedTaskConfig(
        scale="medium",
        seed=5,
        rounds=1,
        warmup=0,
        repeats=2,
        iterations=3,
    )

    payload = _fixed_receipt_payload(
        config=config,
        sizes=sizes,
        case_results=[result],
        backend=BackendSpec(name="numpy"),
    )

    assert payload["schema_version"] == 5
    assert payload["execution_target"] == {
        "backend": "numpy",
        "requested_device": "cpu",
        "resolved_device": "cpu",
    }
    cases = payload["cases"]
    assert isinstance(cases, list)
    json.dumps(payload)
    measurements = cases[0]["measurements"]
    assert [measurement["phase"] for measurement in measurements] == ["steady"]
    assert measurements[0]["observations"][0]["unit_index"] == 1
    assert measurements[0]["comparisons"][0]["paired_unit_count"] == 4


def test_fixed_main_rejects_degenerate_paired_units(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "argv", ["einf_einops_einx.py", "--repeats", "1"])

    with pytest.raises(ValueError, match="repeats must be >= 2"):
        fixed_main()
