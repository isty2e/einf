from pathlib import Path

import numpy as np

from benchmarks.compare.expression_parity import (
    ExpressionCaseResult,
    ExpressionParityReport,
    ExpressionRun,
    _find_case,
    _render_markdown,
    _to_json,
    _validate_output,
)
from benchmarks.harness import BackendSpec, TimingSummary, dynamic_sizes_for_scale


def test_find_case_exposes_gap_strategies() -> None:
    case = _find_case(
        sizes=dynamic_sizes_for_scale("large"),
        case_name="einop_contract_split_dynamic",
    )

    names = [spec.name for spec in case.runner_specs]
    assert names == [
        "einf",
        "einops",
        "einx",
        "torch_matmul_only",
        "torch_matmul_split",
        "torch_matmul_slice",
    ]
    assert (
        next(
            spec for spec in case.runner_specs if spec.name == "torch_matmul_only"
        ).role
        == "baseline"
    )


def test_validate_output_uses_runner_specific_reference() -> None:
    case = _find_case(
        sizes=dynamic_sizes_for_scale("medium"),
        case_name="einop_contract_split_dynamic",
    )
    runner_spec = next(
        spec for spec in case.runner_specs if spec.name == "torch_matmul_only"
    )

    lhs = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    rhs = np.arange(20, dtype=np.float32).reshape(4, 5)
    batch = (lhs, rhs)
    output = runner_spec.reference(batch)

    _validate_output(
        backend=BackendSpec(name="numpy"),
        runner_spec=runner_spec,
        batch=batch,
        output=output,
    )


def test_expression_parity_renderers_include_roles_and_calls(tmp_path: Path) -> None:
    case = _find_case(
        sizes=dynamic_sizes_for_scale("large"),
        case_name="einop_contract_split_dynamic",
    )
    summary = TimingSummary(
        count=12,
        p25_ms=5.0,
        median_ms=5.5,
        p75_ms=6.0,
        iqr_ms=1.0,
        p95_ms=6.8,
        mean_ms=5.6,
        min_ms=4.9,
        max_ms=7.1,
    )
    runs = {
        spec.name: ExpressionRun(
            available=True,
            reason="available",
            role=spec.role,
            dynamic=summary,
        )
        for spec in case.runner_specs
    }
    report = ExpressionParityReport(
        title="# Gap Expression Parity Benchmark",
        configuration=["backend: `torch`"],
        methodology=["serial dynamic batch timing"],
        case_results=(
            ExpressionCaseResult(
                case=case,
                runs=runs,
                round_orders=[tuple(runs.keys())],
            ),
        ),
        notes=["torch_matmul_only is baseline"],
    )

    payload = _to_json(report)
    assert payload["title"] == "# Gap Expression Parity Benchmark"
    case_results = payload["case_results"]
    assert isinstance(case_results, list)
    case_payload = case_results[0]["case"]
    assert isinstance(case_payload, dict)
    runner_specs = case_payload["runner_specs"]
    assert isinstance(runner_specs, list)
    runner_payload = runner_specs[0]
    assert isinstance(runner_payload, dict)
    assert runner_payload["name"] == "einf"
    assert "reference" not in runner_payload

    markdown = _render_markdown(report)
    path = tmp_path / "expression-parity.md"
    path.write_text(markdown, encoding="utf-8")

    assert "torch_matmul_only" in markdown
    assert "| Strategy | Role | Samples | Median (ms) |" in markdown
    assert "`baseline`" in markdown
