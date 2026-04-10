from pathlib import Path

import numpy as np
import pytest

from benchmarks.profile.warm_calltree import (
    WarmCallTreeReport,
    WarmCallTreeRow,
    _find_case,
    _profile_invoke,
    _to_json,
    _to_markdown,
)


def test_find_case_resolves_known_case() -> None:
    case = _find_case(
        mode="fixed",
        scale="medium",
        seed=20260215,
        backend="numpy",
        case_name="contract",
    )

    assert case.name == "contract"
    assert "contract(" in case.call_repr


def test_profile_invoke_warms_before_enabling_profiler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    class FakeProfile:
        def enable(self) -> None:
            events.append("enable")

        def disable(self) -> None:
            events.append("disable")

    def invoke() -> np.ndarray:
        events.append("call")
        return np.zeros((1,), dtype=np.float32)

    monkeypatch.setattr(
        "benchmarks.profile.warm_calltree.cProfile.Profile",
        FakeProfile,
    )

    _ = _profile_invoke(
        invoke=invoke,
        warmup=2,
        loops=3,
    )

    assert events == ["call", "call", "enable", "call", "call", "call", "disable"]


def test_warm_calltree_renderers_include_case_metadata(tmp_path: Path) -> None:
    report = WarmCallTreeReport(
        backend="numpy",
        mode="fixed",
        scale="medium",
        seed=20260215,
        case_name="contract",
        call_repr="contract(...)",
        warmup=16,
        loops=64,
        sort_by="cumtime",
        top=10,
        python="3.11.0",
        numpy="1.26.4",
        torch="not-installed",
        einops="0.8.0",
        einx="0.3.0",
        rows=(
            WarmCallTreeRow(
                filename="src/einf/plans/abstract.py",
                line=123,
                function="resolve_single_output_runner",
                primitive_calls=64,
                total_calls=64,
                total_time_s=0.01,
                cumulative_time_s=0.02,
            ),
        ),
    )

    payload = _to_json(report)
    assert payload["case_name"] == "contract"

    markdown = _to_markdown(report)
    path = tmp_path / "warm-calltree.md"
    path.write_text(markdown, encoding="utf-8")

    assert "Warm Call Tree (einf)" in markdown
    assert "`contract(...)`" in markdown
    assert "resolve_single_output_runner" in markdown
