import json
from pathlib import Path

import pytest

from benchmarks.compare.einf_einops_einx_dynamic import (
    _raw_payload,
    _resolve_raw_output_path,
)
from benchmarks.harness import (
    BenchmarkCase,
    BenchSizes,
    CaseCalls,
    DynamicCaseResult,
    DynamicObservation,
    DynamicRun,
    DynamicShapeWorkload,
    DynamicTaskConfig,
    DynamicWorkloadMetadata,
    PairedComparison,
    TimingSummary,
)
from benchmarks.harness.comparison import compare_paired_observations
from benchmarks.harness.types import LibraryName


def _observations(
    values: tuple[tuple[int, int, int, float, float], ...],
) -> tuple[DynamicObservation, ...]:
    observations: list[DynamicObservation] = []
    for round_index, batch_index, repeat_index, einf_ms, einops_ms in values:
        latency_by_library: tuple[tuple[LibraryName, float], ...] = (
            ("einf", einf_ms),
            ("einops", einops_ms),
        )
        for order_position, (library, latency_ms) in enumerate(latency_by_library):
            observations.append(
                DynamicObservation(
                    round_index=round_index,
                    measured_batch_index=batch_index,
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

    first = compare_paired_observations(
        observations=observations,
        libraries=("einf", "einops"),
        baseline="einf",
        bootstrap_seed=17,
    )
    second = compare_paired_observations(
        observations=tuple(reversed(observations)),
        libraries=("einf", "einops"),
        baseline="einf",
        bootstrap_seed=17,
    )

    assert first == second
    assert len(first) == 1
    comparison = first[0]
    assert comparison.call_pair_count == 8
    assert comparison.paired_batch_count == 4
    assert comparison.latency_ratio == pytest.approx(1.875)
    assert comparison.confidence_interval_low > 0.0
    assert comparison.confidence_interval_high >= comparison.confidence_interval_low


def test_paired_comparison_rejects_insufficient_batches_per_round() -> None:
    observations = _observations(
        (
            (0, 0, 0, 1.0, 2.0),
            (1, 0, 0, 9.0, 9.0),
        )
    )

    with pytest.raises(
        ValueError,
        match="at least two measured batches per round",
    ):
        compare_paired_observations(
            observations=observations,
            libraries=("einf", "einops"),
            baseline="einf",
            bootstrap_seed=3,
        )


def test_paired_comparison_rejects_incomplete_call_pair() -> None:
    observations = _observations(((0, 0, 0, 1.0, 2.0),))

    with pytest.raises(ValueError, match="incomplete paired timing observation"):
        compare_paired_observations(
            observations=observations[:-1],
            libraries=("einf", "einops"),
            baseline="einf",
            bootstrap_seed=0,
        )


def test_paired_comparison_rejects_duplicate_call_identity() -> None:
    observations = _observations(((0, 0, 0, 1.0, 2.0),))

    with pytest.raises(ValueError, match="duplicate paired timing observation"):
        compare_paired_observations(
            observations=observations + observations[:1],
            libraries=("einf", "einops"),
            baseline="einf",
            bootstrap_seed=0,
        )


def test_paired_comparison_rejects_duplicate_members() -> None:
    observations = _observations(((0, 0, 0, 1.0, 2.0),))

    with pytest.raises(ValueError, match="members must be unique"):
        compare_paired_observations(
            observations=observations,
            libraries=("einf", "einf"),
            baseline="einf",
            bootstrap_seed=0,
        )


def test_dynamic_raw_payload_preserves_observation_and_analysis_identity() -> None:
    sizes, workload = _workload()
    workload_comparison = workload.compare_to(
        workload,
        scale="medium",
        reference_scale="medium",
    )
    observation = DynamicObservation(
        round_index=1,
        measured_batch_index=2,
        repeat_index=3,
        library="einf",
        order_position=0,
        latency_ms=1.25,
    )
    comparison: PairedComparison[LibraryName] = PairedComparison(
        baseline="einf",
        competitor="einops",
        call_pair_count=8,
        paired_batch_count=4,
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
        runs={
            "einf": DynamicRun(
                summary=_summary(mean_ms=1.0),
                round_summaries=(_summary(mean_ms=1.0),),
            ),
            "einops": DynamicRun(
                summary=_summary(mean_ms=1.5),
                round_summaries=(_summary(mean_ms=1.5),),
            ),
            "einx": DynamicRun(
                summary=_summary(mean_ms=2.0),
                round_summaries=(_summary(mean_ms=2.0),),
            ),
        },
        round_orders=[("einf", "einops", "einx")],
        observations=(observation,),
        comparisons=(comparison,),
    )
    config = DynamicTaskConfig(
        backend="numpy",
        scale="medium",
        seed=5,
        batches=4,
        warmup_batches=1,
        repeats=2,
        rounds=1,
        round_order_seed=7,
        parity_checks=0,
    )

    payload = _raw_payload(
        config=config,
        sizes=sizes,
        case_results=[result],
        workload_comparisons={"dynamic_case": workload_comparison},
    )

    assert payload["schema_version"] == 2
    cases = payload["cases"]
    assert isinstance(cases, list)
    json.dumps(payload)
    case_payload = cases[0]
    assert case_payload["case"]["name"] == "dynamic_case"
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
    assert case_payload["observations"] == [
        {
            "round_index": 1,
            "measured_batch_index": 2,
            "repeat_index": 3,
            "library": "einf",
            "order_position": 0,
            "latency_ms": 1.25,
        }
    ]
    assert case_payload["comparisons"][0]["paired_batch_count"] == 4


@pytest.mark.parametrize(
    ("output", "raw_output", "expected"),
    (
        (Path("report.md"), None, Path("report.json")),
        (Path("report.md"), Path("raw/data.json"), Path("raw/data.json")),
        (None, Path("raw/data.json"), Path("raw/data.json")),
        (None, None, None),
    ),
)
def test_resolve_raw_output_path(
    output: Path | None,
    raw_output: Path | None,
    expected: Path | None,
) -> None:
    assert _resolve_raw_output_path(output=output, raw_output=raw_output) == expected


def test_resolve_raw_output_path_rejects_collision() -> None:
    with pytest.raises(ValueError, match="must use different paths"):
        _resolve_raw_output_path(
            output=Path("report.md"),
            raw_output=Path("report.md"),
        )


def test_resolve_raw_output_path_rejects_implicit_json_collision() -> None:
    with pytest.raises(ValueError, match="non-JSON suffix"):
        _resolve_raw_output_path(
            output=Path("report.json"),
            raw_output=None,
        )
