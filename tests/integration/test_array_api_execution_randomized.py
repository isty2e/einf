import numpy as np

from einf import ax, axes, packs, rearrange, reduce, repeat


def test_rearrange_randomized_split_concat_roundtrip() -> None:
    rng = np.random.RandomState(101)
    b, n, m, d = axes("b", "n", "m", "d")

    for _ in range(60):
        b_size = int(rng.randint(0, 4))
        n_size = int(rng.randint(0, 4))
        m_size = int(rng.randint(0, 4))
        d_size = int(rng.randint(0, 4))

        split_op = rearrange(ax[b, (n + m), d], (ax[b, n, d], ax[b, m, d])).with_sizes(
            n=n_size,
            m=m_size,
        )
        concat_op = rearrange((ax[b, n, d], ax[b, m, d]), ax[b, (n + m), d]).with_sizes(
            n=n_size,
            m=m_size,
        )

        tensor = rng.randint(
            -5, 6, size=(b_size, n_size + m_size, d_size), dtype=np.int64
        )
        left, right = split_op(tensor)
        rebuilt = concat_op(left, right)

        np.testing.assert_array_equal(rebuilt, tensor)


def test_rearrange_randomized_nested_plus_three_way_roundtrip() -> None:
    rng = np.random.RandomState(102)
    b, n1, n2, n3, d = axes("b", "n1", "n2", "n3", "d")

    for _ in range(50):
        b_size = int(rng.randint(0, 4))
        n1_size = int(rng.randint(0, 4))
        n2_size = int(rng.randint(0, 4))
        n3_size = int(rng.randint(0, 4))
        d_size = int(rng.randint(0, 4))

        split_op = rearrange(
            ax[b, (n1 + (n2 + n3)), d],
            (ax[b, n1, d], ax[b, n2, d], ax[b, n3, d]),
        ).with_sizes(
            n1=n1_size,
            n2=n2_size,
            n3=n3_size,
        )
        concat_op = rearrange(
            (ax[b, n1, d], ax[b, n2, d], ax[b, n3, d]),
            ax[b, (n1 + (n2 + n3)), d],
        ).with_sizes(
            n1=n1_size,
            n2=n2_size,
            n3=n3_size,
        )

        tensor = rng.randint(
            -7,
            8,
            size=(b_size, n1_size + n2_size + n3_size, d_size),
            dtype=np.int64,
        )
        first, second, third = split_op(tensor)
        rebuilt = concat_op(first, second, third)

        np.testing.assert_array_equal(rebuilt, tensor)


def test_rearrange_randomized_product_expand_contract_roundtrip() -> None:
    rng = np.random.RandomState(103)
    b, h, w, d = axes("b", "h", "w", "d")

    for _ in range(50):
        b_size = int(rng.randint(0, 4))
        h_size = int(rng.randint(0, 4))
        w_size = int(rng.randint(0, 4))
        d_size = int(rng.randint(0, 4))

        expand_op = rearrange(ax[b, (h * w), d], ax[b, h, w, d]).with_sizes(
            h=h_size,
            w=w_size,
        )
        contract_op = rearrange(ax[b, h, w, d], ax[b, (h * w), d]).with_sizes(
            h=h_size,
            w=w_size,
        )

        tensor = rng.randint(
            -9,
            10,
            size=(b_size, h_size * w_size, d_size),
            dtype=np.int64,
        )
        rebuilt = contract_op(expand_op(tensor))

        np.testing.assert_array_equal(rebuilt, tensor)


def test_rearrange_randomized_pack_roundtrip_variable_rank() -> None:
    rng = np.random.RandomState(104)
    (b,) = axes("b")
    (tail,) = packs("tail")
    forward = rearrange(ax[b, tail], ax[tail, b])
    inverse = rearrange(ax[tail, b], ax[b, tail])

    for _ in range(70):
        b_size = int(rng.randint(0, 4))
        tail_rank = int(rng.randint(0, 4))
        tail_shape = tuple(int(value) for value in rng.randint(0, 4, size=tail_rank))
        tensor_shape = (b_size, *tail_shape)
        tensor = rng.randint(-3, 4, size=tensor_shape, dtype=np.int64)

        moved = forward(tensor)
        rebuilt = inverse(moved)

        assert moved.shape == (*tail_shape, b_size)
        np.testing.assert_array_equal(rebuilt, tensor)


def test_inflate_reduce_randomized_scaling_identity() -> None:
    rng = np.random.RandomState(105)
    b, d, r = axes("b", "d", "r")
    reduce_op = reduce(ax[b, d, r], ax[b, d])

    for _ in range(70):
        b_size = int(rng.randint(0, 4))
        d_size = int(rng.randint(0, 4))
        r_size = int(rng.randint(0, 5))

        inflate_op = repeat(ax[b, d], ax[b, d, r]).with_sizes(r=r_size)
        tensor = rng.randint(-4, 5, size=(b_size, d_size), dtype=np.int64)

        collapsed = getattr(reduce_op, "__call__")(inflate_op(tensor))
        expected = tensor * r_size

        np.testing.assert_array_equal(collapsed, expected)


def test_reduce_randomized_default_sum_matches_ordered_phase_sums() -> None:
    rng = np.random.RandomState(106)
    b, h, d = axes("b", "h", "d")

    default_op = reduce(ax[b, h, d], ax[b])
    phase_hd_op = reduce(ax[b, h, d], ax[b]).reduce_by((ax[h], "sum"), (ax[d], "sum"))
    phase_dh_op = reduce(ax[b, h, d], ax[b]).reduce_by((ax[d], "sum"), (ax[h], "sum"))

    for _ in range(70):
        b_size = int(rng.randint(0, 4))
        h_size = int(rng.randint(0, 4))
        d_size = int(rng.randint(0, 4))
        tensor = rng.randint(-6, 7, size=(b_size, h_size, d_size), dtype=np.int64)

        default_out = getattr(default_op, "__call__")(tensor)
        phase_hd_out = getattr(phase_hd_op, "__call__")(tensor)
        phase_dh_out = getattr(phase_dh_op, "__call__")(tensor)

        np.testing.assert_array_equal(default_out, phase_hd_out)
        np.testing.assert_array_equal(default_out, phase_dh_out)
