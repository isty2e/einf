from pathlib import Path

from benchmarks.audit.expression_layout import (
    LayoutAuditReport,
    LayoutSample,
    LayoutSnapshot,
    LayoutStrategyReport,
    _find_gap_case,
    _render_markdown,
    _summarize_samples,
    _to_json,
)
from benchmarks.harness import dynamic_sizes_for_scale


def test_find_gap_case_accepts_known_case() -> None:
    case = _find_gap_case(
        case_name="einop_contract_split_dynamic",
        sizes=dynamic_sizes_for_scale("large"),
    )
    assert case.name == "einop_contract_split_dynamic"


def test_summarize_samples_orders_fastest_and_slowest() -> None:
    sample_fast = LayoutSample(
        batch_index=1,
        latency_ms=1.0,
        snapshots=(
            LayoutSnapshot(
                label="out0",
                shape=(2, 3, 4),
                stride=(12, 4, 1),
                is_contiguous=True,
                storage_offset=0,
            ),
        ),
    )
    sample_slow = LayoutSample(
        batch_index=2,
        latency_ms=5.0,
        snapshots=(
            LayoutSnapshot(
                label="out0",
                shape=(2, 3, 4),
                stride=(12, 4, 1),
                is_contiguous=True,
                storage_offset=0,
            ),
        ),
    )
    summary = _summarize_samples(samples=[sample_slow, sample_fast], top_k=1)
    assert summary.fastest_samples[0].batch_index == 1
    assert summary.slowest_samples[0].batch_index == 2


def test_layout_audit_renderers_emit_snapshot_data(tmp_path: Path) -> None:
    report = LayoutAuditReport(
        title="# Gap Expression Layout Audit",
        configuration=["backend: `torch`"],
        methodology=["layout snapshot audit"],
        strategies=(
            LayoutStrategyReport(
                name="einf",
                call_repr="einop(...)",
                summary=_summarize_samples(
                    samples=[
                        LayoutSample(
                            batch_index=0,
                            latency_ms=2.0,
                            snapshots=(
                                LayoutSnapshot(
                                    label="contracted",
                                    shape=(2, 3, 4),
                                    stride=(12, 4, 1),
                                    is_contiguous=True,
                                    storage_offset=0,
                                ),
                            ),
                        ),
                    ],
                    top_k=1,
                ),
            ),
        ),
        notes=["tail explanation only"],
    )

    payload = _to_json(report)
    assert payload["title"] == "# Gap Expression Layout Audit"

    markdown = _render_markdown(report)
    path = tmp_path / "layout-audit.md"
    path.write_text(markdown, encoding="utf-8")

    assert "Most common stride signatures:" in markdown
    assert "contiguous=True" in markdown
