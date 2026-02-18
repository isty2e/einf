import numpy as np

from einf import ax, axes, einop


def test_einop_uses_single_einsum_carrier_path_by_default() -> None:
    b, h, w, d, j, k = axes("b", "h", "w", "d", "j", "k")
    op = einop(
        (ax[b, (h + w), d], ax[d, j], ax[j, k]),
        (ax[b, h, k], ax[b, w, k]),
    ).with_sizes(h=2, w=1)

    _ = op(
        np.arange(2 * 3 * 5).reshape(2, 3, 5),
        np.arange(5 * 4).reshape(5, 4),
        np.arange(4 * 6).reshape(4, 6),
    )

    plan = op.plan_dict()
    assert plan["kind"] == "einsum_carrier_then_unary"
    assert len(plan["steps"]) == 2
    assert plan["steps"][0] == {"op": "einsum", "equation": "abc,cd,de->abe"}
    assert plan["steps"][1] == {"op": "axis_slice"}
