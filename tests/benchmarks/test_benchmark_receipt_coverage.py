from dataclasses import replace
from itertools import product
from typing import Literal

import pytest

pytest.importorskip("numpy")

import benchmarks.compare.einf_einops_einx as fixed_module
import benchmarks.compare.einf_einops_einx_dynamic as dynamic_module
import benchmarks.compare.expression_parity as expression_module
from benchmarks.harness import (
    AvailableRun,
    BackendSpec,
    BenchmarkCase,
    BenchSizes,
    CaseCalls,
    DynamicCaseResult,
    DynamicShapeWorkload,
    DynamicTaskConfig,
    FixedCaseResult,
    FixedTaskConfig,
    LibraryTimingObservation,
    PairedComparison,
    PairedEvidence,
    TimingSummary,
    UnavailableRun,
)
from benchmarks.harness.receipt import validation_coverage_payload
from benchmarks.harness.result import DynamicInputUnit
from benchmarks.harness.types import LibraryName

_Family = Literal["fixed", "dynamic", "expression"]


def _serialize(family: _Family, defect: str = "") -> dict[str, object]:
    identities = list(product(range(2), range(3), range(4)))
    if defect == "missing_round":
        identities = [identity for identity in identities if identity[0] == 0]
    elif defect == "missing_unit":
        identities = [identity for identity in identities if identity[1] != 2]
    elif defect == "foreign_round":
        identities = [
            (round_index + 1, unit, repeat) for round_index, unit, repeat in identities
        ]
    elif defect == "foreign_unit":
        identities = [
            (round_index, unit + 1, repeat) for round_index, unit, repeat in identities
        ]
    elif defect == "missing_execution":
        identities.pop()
    elif defect == "duplicate_execution":
        identities.append(identities[0])
    elif defect == "extra_execution":
        identities.append((0, 0, 4))
    elif defect == "shifted_execution":
        identities = [
            (round_index, unit, repeat + 1) for round_index, unit, repeat in identities
        ]
    elif defect == "empty_available":
        identities = []

    libraries: tuple[LibraryName, ...] = ("einf", "einops")
    if defect == "missing_available_member":
        libraries = ("einf",)
    elif defect == "unavailable_member":
        libraries = ("einf", "einops", "einx")
    observations = tuple(
        LibraryTimingObservation(
            round_index=round_index,
            unit_index=unit_index,
            repeat_index=repeat_index,
            library=library,
            order_position=position,
            latency_ms=1.0 + position,
        )
        for round_index, unit_index, repeat_index in reversed(identities)
        for position, library in enumerate(libraries)
    )
    if defect == "incomplete_second_member":
        observations = observations[:-1]
    realized_units = tuple(
        DynamicInputUnit(
            round_index=round_index,
            unit_index=unit_index,
            stream_index=unit_index + 1,
            seed=round_index * 10 + unit_index,
            input_shapes=((2,),),
        )
        for round_index, unit_index in sorted({identity[:2] for identity in identities})
    )
    summary = TimingSummary(
        count=24,
        p25_ms=1.0,
        median_ms=1.0,
        p75_ms=1.0,
        iqr_ms=0.0,
        p95_ms=1.0,
        mean_ms=1.0,
        min_ms=1.0,
        max_ms=1.0,
    )
    case = BenchmarkCase(
        name="coverage",
        description="Synthetic receipt coverage",
        calls=CaseCalls(einf="identity()", einops="identity()", einx="identity()"),
        reference=lambda batch: batch[0],
        make_einf_runner=lambda: lambda batch: batch[0],
        make_einops_runner=lambda: lambda batch: batch[0],
        make_einx_runner=lambda: lambda batch: batch[0],
    )
    runs: dict[LibraryName, AvailableRun | UnavailableRun] = {
        "einf": AvailableRun(summary, (summary, summary)),
        "einops": AvailableRun(summary, (summary, summary)),
        "einx": UnavailableRun("not installed"),
    }
    orders: list[tuple[LibraryName, ...]] = [("einops", "einf"), ("einf", "einops")]
    if defect == "all_unavailable":
        runs = {name: UnavailableRun("not installed") for name in runs}
        observations = ()
        realized_units = ()
        orders = []

    evidence = PairedEvidence(observations=observations, comparisons=())
    sizes = BenchSizes(b=2, n=2, d=2, h=2, w=2, r=2, j=2)
    backend = BackendSpec(name="numpy")
    source: dict[str, str | bool | None] = {"kind": "test"}
    if family == "fixed":
        config = FixedTaskConfig(
            scale="small", seed=7, rounds=2, warmup=1, repeats=3, iterations=4
        )
        if defect == "configuration_rounds":
            config = replace(config, rounds=3)
        return fixed_module._receipt_payload(
            config=config,
            sizes=sizes,
            case_results=[FixedCaseResult(case, runs, orders, evidence)],
            backend=backend,
            einf_source=source,
        )

    dynamic_config = DynamicTaskConfig(
        scale="medium",
        seed=7,
        rounds=2,
        batches=4,
        warmup_batches=1,
        repeats=4,
        round_order_seed=11,
    )
    if defect == "configuration_rounds":
        dynamic_config = replace(dynamic_config, rounds=3)
    if family == "dynamic":
        workload = DynamicShapeWorkload(
            sampled_dimensions=(),
            input_shapes=lambda dimensions: ((dimensions["b"],),),
            output_shapes=lambda dimensions: ((dimensions["b"],),),
        ).metadata(sizes=sizes)
        return dynamic_module._receipt_payload(
            config=dynamic_config,
            sizes=sizes,
            case_results=[
                DynamicCaseResult(
                    case, workload, realized_units, runs, orders, evidence
                )
            ],
            workload_comparisons={
                case.name: workload.compare_to(
                    workload, scale="medium", reference_scale="medium"
                )
            },
            backend=backend,
            einf_source=source,
        )

    strategy_names = {"einf": "target", "einops": "equivalent", "einx": "unavailable"}
    expression_case = expression_module.ExpressionParityCase(
        name=case.name,
        description=case.description,
        target_name="target",
        batch_factory=lambda generator: (generator.randn_numpy((2,)),),
        runner_specs=tuple(
            expression_module.ExpressionRunnerSpec(
                name=name,
                semantics="output_equivalent",
                description=name,
                call_repr="identity()",
                available=name != "unavailable",
                reason="test",
                reference=lambda batch: batch[0],
                make_runner=lambda: lambda batch: batch[0],
            )
            for name in strategy_names.values()
        ),
    )
    result = expression_module.ExpressionCaseResult(
        case=expression_case,
        realized_units=realized_units,
        runs={
            name: expression_module.ExpressionRun(summary, (summary, summary))
            for name in ("target", "equivalent")
        },
        round_orders=[("equivalent", "target"), ("target", "equivalent")],
        observations=tuple(
            expression_module.ExpressionObservation(
                round_index=observation.round_index,
                unit_index=observation.unit_index,
                repeat_index=observation.repeat_index,
                strategy=strategy_names[observation.library],
                order_position=observation.order_position,
                latency_ms=observation.latency_ms,
            )
            for observation in observations
        ),
        comparisons=(
            PairedComparison(
                baseline="target",
                competitor="equivalent",
                call_pair_count=24,
                paired_unit_count=6,
                latency_ratio=2.0,
                confidence_level=0.95,
                confidence_interval_low=1.0,
                confidence_interval_high=3.0,
                bootstrap_resamples=10,
                bootstrap_seed=7,
            ),
        ),
    )
    return expression_module._to_json(
        expression_module.ExpressionParityReport(
            title="coverage",
            configuration=[],
            methodology=[],
            case_results=(result,),
            notes=[],
        ),
        backend=backend,
        config=dynamic_config,
        einf_source=source,
    )


@pytest.mark.parametrize("family", ("fixed", "dynamic", "expression"))
@pytest.mark.parametrize(
    "defect",
    (
        "missing_round",
        "missing_unit",
        "foreign_round",
        "foreign_unit",
        "missing_execution",
        "duplicate_execution",
        "extra_execution",
        "shifted_execution",
        "missing_available_member",
        "incomplete_second_member",
        "unavailable_member",
        "empty_available",
        "configuration_rounds",
    ),
)
def test_receipt_rejects_incomplete_execution_coverage(
    family: _Family, defect: str
) -> None:
    with pytest.raises(ValueError):
        _serialize(family, defect)


@pytest.mark.parametrize("family", ("fixed", "dynamic", "expression"))
def test_receipt_accepts_complete_reordered_execution_coverage(family: _Family) -> None:
    payload = _serialize(family)
    cases = payload["case_results" if family == "expression" else "cases"]
    assert isinstance(cases, list)
    validation = cases[0]["validation"]
    assert validation["coordinate_count"] == 6
    assert validation["executions_per_member_per_coordinate"] == 4
    assert validation["members"] == (
        ["target", "equivalent"] if family == "expression" else ["einf", "einops"]
    )


@pytest.mark.parametrize("family", ("fixed", "dynamic"))
def test_receipt_preserves_all_unavailable_case(family: _Family) -> None:
    payload = _serialize(family, "all_unavailable")
    cases = payload["cases"]
    assert isinstance(cases, list)
    assert cases[0]["validation"]["coordinate_count"] == 0
    assert cases[0]["validation"]["members"] == []


@pytest.mark.parametrize("mask", range(256))
def test_coverage_admits_only_the_complete_small_execution_grid(mask: int) -> None:
    grid = tuple(product(range(2), range(2), range(2)))
    identities = tuple(
        identity for index, identity in enumerate(grid) if mask & (1 << index)
    )

    def serialize() -> dict[str, object]:
        return validation_coverage_payload(
            coordinate_source="measurement_observations",
            expected_rounds=2,
            expected_units_per_round=2,
            expected_executions_per_member_per_coordinate=2,
            expected_members=("einf",),
            execution_identities_by_member={"einf": identities},
        )

    if mask == 255:
        assert serialize()["coordinate_count"] == 4
    else:
        with pytest.raises(ValueError, match="do not cover every configured execution"):
            serialize()


@pytest.mark.parametrize(
    "identity",
    (
        (-1, 0, 0),
        (0, -1, 0),
        (0, 0, -1),
        (1, 0, 0),
        (0, 1, 0),
        (0, 0, 1),
        (False, 0, 0),
        (0, False, 0),
        (0, 0, False),
        (0.0, 0, 0),
        (0, 0.0, 0),
        (0, 0, 0.0),
    ),
)
def test_coverage_rejects_invalid_grid_indices(identity: tuple[int, int, int]) -> None:
    with pytest.raises(ValueError, match="outside benchmark configuration"):
        validation_coverage_payload(
            coordinate_source="measurement_observations",
            expected_rounds=1,
            expected_units_per_round=1,
            expected_executions_per_member_per_coordinate=1,
            expected_members=("einf",),
            execution_identities_by_member={"einf": (identity,)},
        )


@pytest.mark.parametrize("members", (("einf", "einf"), ("einf", "foreign"), ()))
def test_coverage_rejects_declared_member_mismatch(members: tuple[str, ...]) -> None:
    with pytest.raises(ValueError, match="members"):
        validation_coverage_payload(
            coordinate_source="measurement_observations",
            expected_rounds=1,
            expected_units_per_round=1,
            expected_executions_per_member_per_coordinate=1,
            expected_members=members,
            execution_identities_by_member={"einf": ((0, 0, 0),)},
        )


@pytest.mark.parametrize("dimension", ("rounds", "units", "executions"))
@pytest.mark.parametrize("value", (0, -1, True, 1.0))
def test_coverage_rejects_invalid_schedule_dimensions(
    dimension: str, value: int
) -> None:
    with pytest.raises(ValueError, match="must be a positive integer"):
        validation_coverage_payload(
            coordinate_source="measurement_observations",
            expected_rounds=value if dimension == "rounds" else 1,
            expected_units_per_round=value if dimension == "units" else 1,
            expected_executions_per_member_per_coordinate=value
            if dimension == "executions"
            else 1,
            expected_members=("einf",),
            execution_identities_by_member={"einf": ((0, 0, 0),)},
        )
