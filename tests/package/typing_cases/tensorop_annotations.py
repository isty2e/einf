import sys
from collections.abc import Callable
from typing import TypeVar

if sys.version_info >= (3, 11):
    from typing import Self, assert_type
else:
    from typing_extensions import Self, assert_type

from einf import (
    ReducibleTensorOp,
    TensorLike,
    TensorOp,
    ax,
    axes,
    contract,
    einop,
    rearrange,
    reduce,
    view,
)

T = TypeVar("T", bound=TensorLike)


class Dummy:
    @property
    def shape(self) -> tuple[int, ...]:
        return (2, 3)

    def __getitem__(self, key: object, /) -> Self:
        return self


b, n, d = axes("b", "n", "d")
plain: TensorOp[[Dummy], Dummy] = rearrange(ax[b, n], ax[n, b])
binary: TensorOp[[Dummy, Dummy], Dummy] = contract((ax[b, n], ax[n, d]), ax[b, d])
split: TensorOp[[Dummy], tuple[Dummy, Dummy]] = view(ax[b + n], (ax[b], ax[n]))
reducer: ReducibleTensorOp[[Dummy], Dummy] = reduce(ax[b, n], ax[b])
_ = assert_type(reducer.reduce_by(reducer="mean")(Dummy()), Dummy)
_ = assert_type(reducer.reduce_by(reducer=(ax[n], "sum"))(Dummy()), Dummy)


def apply(op: TensorOp[[T], T], tensor: T) -> T:
    _ = assert_type(op(tensor), T)
    return op.with_sizes(b=2)(tensor)


def apply_reducer(op: ReducibleTensorOp[[T], T], tensor: T) -> T:
    return op.reduce_by("sum").with_sizes(b=2)(tensor)


def inspect(op: TensorOp[[Dummy], Dummy], tensor: Dummy) -> None:
    _ = assert_type(op(tensor), Dummy)
    _ = assert_type(op.name, str)
    _ = assert_type(op.sizes_items, tuple[tuple[str, int], ...])


def callable_view(op: TensorOp[[T], T]) -> Callable[[T], T]:
    return op


def split_view(op: TensorOp[[T], tuple[T, T]]) -> Callable[[T], tuple[T, T]]:
    return op.with_sizes(b=2)


_ = assert_type(apply(rearrange(ax[b, n], ax[n, b]), Dummy()), Dummy)
_ = assert_type(apply_reducer(reduce(ax[b, n], ax[b]), Dummy()), Dummy)

op_1_1: TensorOp[[Dummy], Dummy] = rearrange(ax[b], ax[b])
red_1_1: ReducibleTensorOp[[Dummy], Dummy] = einop(ax[b], ax[b])


def configured_1_1(op: TensorOp[[T], T], red: ReducibleTensorOp[[T], T], x0: T) -> T:
    _ = assert_type(op.with_sizes(b=1)(x0), T)
    return red.reduce_by("sum").with_sizes(b=1)(x0)


op_1_2: TensorOp[[Dummy], tuple[Dummy, Dummy]] = rearrange(ax[b], (ax[b], ax[b]))
red_1_2: ReducibleTensorOp[[Dummy], tuple[Dummy, Dummy]] = einop(ax[b], (ax[b], ax[b]))


def configured_1_2(
    op: TensorOp[[T], tuple[T, T]], red: ReducibleTensorOp[[T], tuple[T, T]], x0: T
) -> tuple[T, T]:
    _ = assert_type(op.with_sizes(b=1)(x0), tuple[T, T])
    return red.reduce_by("sum").with_sizes(b=1)(x0)


op_1_3: TensorOp[[Dummy], tuple[Dummy, Dummy, Dummy]] = rearrange(
    ax[b], (ax[b], ax[b], ax[b])
)
red_1_3: ReducibleTensorOp[[Dummy], tuple[Dummy, Dummy, Dummy]] = einop(
    ax[b], (ax[b], ax[b], ax[b])
)


def configured_1_3(
    op: TensorOp[[T], tuple[T, T, T]],
    red: ReducibleTensorOp[[T], tuple[T, T, T]],
    x0: T,
) -> tuple[T, T, T]:
    _ = assert_type(op.with_sizes(b=1)(x0), tuple[T, T, T])
    return red.reduce_by("sum").with_sizes(b=1)(x0)


op_1_4: TensorOp[[Dummy], tuple[Dummy, Dummy, Dummy, Dummy]] = rearrange(
    ax[b], (ax[b], ax[b], ax[b], ax[b])
)
red_1_4: ReducibleTensorOp[[Dummy], tuple[Dummy, Dummy, Dummy, Dummy]] = einop(
    ax[b], (ax[b], ax[b], ax[b], ax[b])
)


def configured_1_4(
    op: TensorOp[[T], tuple[T, T, T, T]],
    red: ReducibleTensorOp[[T], tuple[T, T, T, T]],
    x0: T,
) -> tuple[T, T, T, T]:
    _ = assert_type(op.with_sizes(b=1)(x0), tuple[T, T, T, T])
    return red.reduce_by("sum").with_sizes(b=1)(x0)


op_1_5: TensorOp[[Dummy], tuple[Dummy, Dummy, Dummy, Dummy, Dummy]] = rearrange(
    ax[b], (ax[b], ax[b], ax[b], ax[b], ax[b])
)
red_1_5: ReducibleTensorOp[[Dummy], tuple[Dummy, Dummy, Dummy, Dummy, Dummy]] = einop(
    ax[b], (ax[b], ax[b], ax[b], ax[b], ax[b])
)


def configured_1_5(
    op: TensorOp[[T], tuple[T, T, T, T, T]],
    red: ReducibleTensorOp[[T], tuple[T, T, T, T, T]],
    x0: T,
) -> tuple[T, T, T, T, T]:
    _ = assert_type(op.with_sizes(b=1)(x0), tuple[T, T, T, T, T])
    return red.reduce_by("sum").with_sizes(b=1)(x0)


op_1_6: TensorOp[[Dummy], tuple[Dummy, Dummy, Dummy, Dummy, Dummy, Dummy]] = rearrange(
    ax[b], (ax[b], ax[b], ax[b], ax[b], ax[b], ax[b])
)
red_1_6: ReducibleTensorOp[[Dummy], tuple[Dummy, Dummy, Dummy, Dummy, Dummy, Dummy]] = (
    einop(ax[b], (ax[b], ax[b], ax[b], ax[b], ax[b], ax[b]))
)


def configured_1_6(
    op: TensorOp[[T], tuple[T, T, T, T, T, T]],
    red: ReducibleTensorOp[[T], tuple[T, T, T, T, T, T]],
    x0: T,
) -> tuple[T, T, T, T, T, T]:
    _ = assert_type(op.with_sizes(b=1)(x0), tuple[T, T, T, T, T, T])
    return red.reduce_by("sum").with_sizes(b=1)(x0)


op_2_1: TensorOp[[Dummy, Dummy], Dummy] = rearrange((ax[b], ax[b]), ax[b])
red_2_1: ReducibleTensorOp[[Dummy, Dummy], Dummy] = einop((ax[b], ax[b]), ax[b])


def configured_2_1(
    op: TensorOp[[T, T], T], red: ReducibleTensorOp[[T, T], T], x0: T, x1: T
) -> T:
    _ = assert_type(op.with_sizes(b=1)(x0, x1), T)
    return red.reduce_by("sum").with_sizes(b=1)(x0, x1)


op_2_2: TensorOp[[Dummy, Dummy], tuple[Dummy, Dummy]] = rearrange(
    (ax[b], ax[b]), (ax[b], ax[b])
)
red_2_2: ReducibleTensorOp[[Dummy, Dummy], tuple[Dummy, Dummy]] = einop(
    (ax[b], ax[b]), (ax[b], ax[b])
)


def configured_2_2(
    op: TensorOp[[T, T], tuple[T, T]],
    red: ReducibleTensorOp[[T, T], tuple[T, T]],
    x0: T,
    x1: T,
) -> tuple[T, T]:
    _ = assert_type(op.with_sizes(b=1)(x0, x1), tuple[T, T])
    return red.reduce_by("sum").with_sizes(b=1)(x0, x1)


op_2_3: TensorOp[[Dummy, Dummy], tuple[Dummy, Dummy, Dummy]] = rearrange(
    (ax[b], ax[b]), (ax[b], ax[b], ax[b])
)
red_2_3: ReducibleTensorOp[[Dummy, Dummy], tuple[Dummy, Dummy, Dummy]] = einop(
    (ax[b], ax[b]), (ax[b], ax[b], ax[b])
)


def configured_2_3(
    op: TensorOp[[T, T], tuple[T, T, T]],
    red: ReducibleTensorOp[[T, T], tuple[T, T, T]],
    x0: T,
    x1: T,
) -> tuple[T, T, T]:
    _ = assert_type(op.with_sizes(b=1)(x0, x1), tuple[T, T, T])
    return red.reduce_by("sum").with_sizes(b=1)(x0, x1)


op_2_4: TensorOp[[Dummy, Dummy], tuple[Dummy, Dummy, Dummy, Dummy]] = rearrange(
    (ax[b], ax[b]), (ax[b], ax[b], ax[b], ax[b])
)
red_2_4: ReducibleTensorOp[[Dummy, Dummy], tuple[Dummy, Dummy, Dummy, Dummy]] = einop(
    (ax[b], ax[b]), (ax[b], ax[b], ax[b], ax[b])
)


def configured_2_4(
    op: TensorOp[[T, T], tuple[T, T, T, T]],
    red: ReducibleTensorOp[[T, T], tuple[T, T, T, T]],
    x0: T,
    x1: T,
) -> tuple[T, T, T, T]:
    _ = assert_type(op.with_sizes(b=1)(x0, x1), tuple[T, T, T, T])
    return red.reduce_by("sum").with_sizes(b=1)(x0, x1)


op_2_5: TensorOp[[Dummy, Dummy], tuple[Dummy, Dummy, Dummy, Dummy, Dummy]] = rearrange(
    (ax[b], ax[b]), (ax[b], ax[b], ax[b], ax[b], ax[b])
)
red_2_5: ReducibleTensorOp[[Dummy, Dummy], tuple[Dummy, Dummy, Dummy, Dummy, Dummy]] = (
    einop((ax[b], ax[b]), (ax[b], ax[b], ax[b], ax[b], ax[b]))
)


def configured_2_5(
    op: TensorOp[[T, T], tuple[T, T, T, T, T]],
    red: ReducibleTensorOp[[T, T], tuple[T, T, T, T, T]],
    x0: T,
    x1: T,
) -> tuple[T, T, T, T, T]:
    _ = assert_type(op.with_sizes(b=1)(x0, x1), tuple[T, T, T, T, T])
    return red.reduce_by("sum").with_sizes(b=1)(x0, x1)


op_2_6: TensorOp[[Dummy, Dummy], tuple[Dummy, Dummy, Dummy, Dummy, Dummy, Dummy]] = (
    rearrange((ax[b], ax[b]), (ax[b], ax[b], ax[b], ax[b], ax[b], ax[b]))
)
red_2_6: ReducibleTensorOp[
    [Dummy, Dummy], tuple[Dummy, Dummy, Dummy, Dummy, Dummy, Dummy]
] = einop((ax[b], ax[b]), (ax[b], ax[b], ax[b], ax[b], ax[b], ax[b]))


def configured_2_6(
    op: TensorOp[[T, T], tuple[T, T, T, T, T, T]],
    red: ReducibleTensorOp[[T, T], tuple[T, T, T, T, T, T]],
    x0: T,
    x1: T,
) -> tuple[T, T, T, T, T, T]:
    _ = assert_type(op.with_sizes(b=1)(x0, x1), tuple[T, T, T, T, T, T])
    return red.reduce_by("sum").with_sizes(b=1)(x0, x1)


op_3_1: TensorOp[[Dummy, Dummy, Dummy], Dummy] = rearrange((ax[b], ax[b], ax[b]), ax[b])
red_3_1: ReducibleTensorOp[[Dummy, Dummy, Dummy], Dummy] = einop(
    (ax[b], ax[b], ax[b]), ax[b]
)


def configured_3_1(
    op: TensorOp[[T, T, T], T],
    red: ReducibleTensorOp[[T, T, T], T],
    x0: T,
    x1: T,
    x2: T,
) -> T:
    _ = assert_type(op.with_sizes(b=1)(x0, x1, x2), T)
    return red.reduce_by("sum").with_sizes(b=1)(x0, x1, x2)


op_3_2: TensorOp[[Dummy, Dummy, Dummy], tuple[Dummy, Dummy]] = rearrange(
    (ax[b], ax[b], ax[b]), (ax[b], ax[b])
)
red_3_2: ReducibleTensorOp[[Dummy, Dummy, Dummy], tuple[Dummy, Dummy]] = einop(
    (ax[b], ax[b], ax[b]), (ax[b], ax[b])
)


def configured_3_2(
    op: TensorOp[[T, T, T], tuple[T, T]],
    red: ReducibleTensorOp[[T, T, T], tuple[T, T]],
    x0: T,
    x1: T,
    x2: T,
) -> tuple[T, T]:
    _ = assert_type(op.with_sizes(b=1)(x0, x1, x2), tuple[T, T])
    return red.reduce_by("sum").with_sizes(b=1)(x0, x1, x2)


op_3_3: TensorOp[[Dummy, Dummy, Dummy], tuple[Dummy, Dummy, Dummy]] = rearrange(
    (ax[b], ax[b], ax[b]), (ax[b], ax[b], ax[b])
)
red_3_3: ReducibleTensorOp[[Dummy, Dummy, Dummy], tuple[Dummy, Dummy, Dummy]] = einop(
    (ax[b], ax[b], ax[b]), (ax[b], ax[b], ax[b])
)


def configured_3_3(
    op: TensorOp[[T, T, T], tuple[T, T, T]],
    red: ReducibleTensorOp[[T, T, T], tuple[T, T, T]],
    x0: T,
    x1: T,
    x2: T,
) -> tuple[T, T, T]:
    _ = assert_type(op.with_sizes(b=1)(x0, x1, x2), tuple[T, T, T])
    return red.reduce_by("sum").with_sizes(b=1)(x0, x1, x2)


op_3_4: TensorOp[[Dummy, Dummy, Dummy], tuple[Dummy, Dummy, Dummy, Dummy]] = rearrange(
    (ax[b], ax[b], ax[b]), (ax[b], ax[b], ax[b], ax[b])
)
red_3_4: ReducibleTensorOp[[Dummy, Dummy, Dummy], tuple[Dummy, Dummy, Dummy, Dummy]] = (
    einop((ax[b], ax[b], ax[b]), (ax[b], ax[b], ax[b], ax[b]))
)


def configured_3_4(
    op: TensorOp[[T, T, T], tuple[T, T, T, T]],
    red: ReducibleTensorOp[[T, T, T], tuple[T, T, T, T]],
    x0: T,
    x1: T,
    x2: T,
) -> tuple[T, T, T, T]:
    _ = assert_type(op.with_sizes(b=1)(x0, x1, x2), tuple[T, T, T, T])
    return red.reduce_by("sum").with_sizes(b=1)(x0, x1, x2)


op_3_5: TensorOp[[Dummy, Dummy, Dummy], tuple[Dummy, Dummy, Dummy, Dummy, Dummy]] = (
    rearrange((ax[b], ax[b], ax[b]), (ax[b], ax[b], ax[b], ax[b], ax[b]))
)
red_3_5: ReducibleTensorOp[
    [Dummy, Dummy, Dummy], tuple[Dummy, Dummy, Dummy, Dummy, Dummy]
] = einop((ax[b], ax[b], ax[b]), (ax[b], ax[b], ax[b], ax[b], ax[b]))


def configured_3_5(
    op: TensorOp[[T, T, T], tuple[T, T, T, T, T]],
    red: ReducibleTensorOp[[T, T, T], tuple[T, T, T, T, T]],
    x0: T,
    x1: T,
    x2: T,
) -> tuple[T, T, T, T, T]:
    _ = assert_type(op.with_sizes(b=1)(x0, x1, x2), tuple[T, T, T, T, T])
    return red.reduce_by("sum").with_sizes(b=1)(x0, x1, x2)


op_3_6: TensorOp[
    [Dummy, Dummy, Dummy], tuple[Dummy, Dummy, Dummy, Dummy, Dummy, Dummy]
] = rearrange((ax[b], ax[b], ax[b]), (ax[b], ax[b], ax[b], ax[b], ax[b], ax[b]))
red_3_6: ReducibleTensorOp[
    [Dummy, Dummy, Dummy], tuple[Dummy, Dummy, Dummy, Dummy, Dummy, Dummy]
] = einop((ax[b], ax[b], ax[b]), (ax[b], ax[b], ax[b], ax[b], ax[b], ax[b]))


def configured_3_6(
    op: TensorOp[[T, T, T], tuple[T, T, T, T, T, T]],
    red: ReducibleTensorOp[[T, T, T], tuple[T, T, T, T, T, T]],
    x0: T,
    x1: T,
    x2: T,
) -> tuple[T, T, T, T, T, T]:
    _ = assert_type(op.with_sizes(b=1)(x0, x1, x2), tuple[T, T, T, T, T, T])
    return red.reduce_by("sum").with_sizes(b=1)(x0, x1, x2)


op_4_1: TensorOp[[Dummy, Dummy, Dummy, Dummy], Dummy] = rearrange(
    (ax[b], ax[b], ax[b], ax[b]), ax[b]
)
red_4_1: ReducibleTensorOp[[Dummy, Dummy, Dummy, Dummy], Dummy] = einop(
    (ax[b], ax[b], ax[b], ax[b]), ax[b]
)


def configured_4_1(
    op: TensorOp[[T, T, T, T], T],
    red: ReducibleTensorOp[[T, T, T, T], T],
    x0: T,
    x1: T,
    x2: T,
    x3: T,
) -> T:
    _ = assert_type(op.with_sizes(b=1)(x0, x1, x2, x3), T)
    return red.reduce_by("sum").with_sizes(b=1)(x0, x1, x2, x3)


op_4_2: TensorOp[[Dummy, Dummy, Dummy, Dummy], tuple[Dummy, Dummy]] = rearrange(
    (ax[b], ax[b], ax[b], ax[b]), (ax[b], ax[b])
)
red_4_2: ReducibleTensorOp[[Dummy, Dummy, Dummy, Dummy], tuple[Dummy, Dummy]] = einop(
    (ax[b], ax[b], ax[b], ax[b]), (ax[b], ax[b])
)


def configured_4_2(
    op: TensorOp[[T, T, T, T], tuple[T, T]],
    red: ReducibleTensorOp[[T, T, T, T], tuple[T, T]],
    x0: T,
    x1: T,
    x2: T,
    x3: T,
) -> tuple[T, T]:
    _ = assert_type(op.with_sizes(b=1)(x0, x1, x2, x3), tuple[T, T])
    return red.reduce_by("sum").with_sizes(b=1)(x0, x1, x2, x3)


op_4_3: TensorOp[[Dummy, Dummy, Dummy, Dummy], tuple[Dummy, Dummy, Dummy]] = rearrange(
    (ax[b], ax[b], ax[b], ax[b]), (ax[b], ax[b], ax[b])
)
red_4_3: ReducibleTensorOp[[Dummy, Dummy, Dummy, Dummy], tuple[Dummy, Dummy, Dummy]] = (
    einop((ax[b], ax[b], ax[b], ax[b]), (ax[b], ax[b], ax[b]))
)


def configured_4_3(
    op: TensorOp[[T, T, T, T], tuple[T, T, T]],
    red: ReducibleTensorOp[[T, T, T, T], tuple[T, T, T]],
    x0: T,
    x1: T,
    x2: T,
    x3: T,
) -> tuple[T, T, T]:
    _ = assert_type(op.with_sizes(b=1)(x0, x1, x2, x3), tuple[T, T, T])
    return red.reduce_by("sum").with_sizes(b=1)(x0, x1, x2, x3)


op_4_4: TensorOp[[Dummy, Dummy, Dummy, Dummy], tuple[Dummy, Dummy, Dummy, Dummy]] = (
    rearrange((ax[b], ax[b], ax[b], ax[b]), (ax[b], ax[b], ax[b], ax[b]))
)
red_4_4: ReducibleTensorOp[
    [Dummy, Dummy, Dummy, Dummy], tuple[Dummy, Dummy, Dummy, Dummy]
] = einop((ax[b], ax[b], ax[b], ax[b]), (ax[b], ax[b], ax[b], ax[b]))


def configured_4_4(
    op: TensorOp[[T, T, T, T], tuple[T, T, T, T]],
    red: ReducibleTensorOp[[T, T, T, T], tuple[T, T, T, T]],
    x0: T,
    x1: T,
    x2: T,
    x3: T,
) -> tuple[T, T, T, T]:
    _ = assert_type(op.with_sizes(b=1)(x0, x1, x2, x3), tuple[T, T, T, T])
    return red.reduce_by("sum").with_sizes(b=1)(x0, x1, x2, x3)


op_4_5: TensorOp[
    [Dummy, Dummy, Dummy, Dummy], tuple[Dummy, Dummy, Dummy, Dummy, Dummy]
] = rearrange((ax[b], ax[b], ax[b], ax[b]), (ax[b], ax[b], ax[b], ax[b], ax[b]))
red_4_5: ReducibleTensorOp[
    [Dummy, Dummy, Dummy, Dummy], tuple[Dummy, Dummy, Dummy, Dummy, Dummy]
] = einop((ax[b], ax[b], ax[b], ax[b]), (ax[b], ax[b], ax[b], ax[b], ax[b]))


def configured_4_5(
    op: TensorOp[[T, T, T, T], tuple[T, T, T, T, T]],
    red: ReducibleTensorOp[[T, T, T, T], tuple[T, T, T, T, T]],
    x0: T,
    x1: T,
    x2: T,
    x3: T,
) -> tuple[T, T, T, T, T]:
    _ = assert_type(op.with_sizes(b=1)(x0, x1, x2, x3), tuple[T, T, T, T, T])
    return red.reduce_by("sum").with_sizes(b=1)(x0, x1, x2, x3)


op_4_6: TensorOp[
    [Dummy, Dummy, Dummy, Dummy], tuple[Dummy, Dummy, Dummy, Dummy, Dummy, Dummy]
] = rearrange((ax[b], ax[b], ax[b], ax[b]), (ax[b], ax[b], ax[b], ax[b], ax[b], ax[b]))
red_4_6: ReducibleTensorOp[
    [Dummy, Dummy, Dummy, Dummy], tuple[Dummy, Dummy, Dummy, Dummy, Dummy, Dummy]
] = einop((ax[b], ax[b], ax[b], ax[b]), (ax[b], ax[b], ax[b], ax[b], ax[b], ax[b]))


def configured_4_6(
    op: TensorOp[[T, T, T, T], tuple[T, T, T, T, T, T]],
    red: ReducibleTensorOp[[T, T, T, T], tuple[T, T, T, T, T, T]],
    x0: T,
    x1: T,
    x2: T,
    x3: T,
) -> tuple[T, T, T, T, T, T]:
    _ = assert_type(op.with_sizes(b=1)(x0, x1, x2, x3), tuple[T, T, T, T, T, T])
    return red.reduce_by("sum").with_sizes(b=1)(x0, x1, x2, x3)


op_5_1: TensorOp[[Dummy, Dummy, Dummy, Dummy, Dummy], Dummy] = rearrange(
    (ax[b], ax[b], ax[b], ax[b], ax[b]), ax[b]
)
red_5_1: ReducibleTensorOp[[Dummy, Dummy, Dummy, Dummy, Dummy], Dummy] = einop(
    (ax[b], ax[b], ax[b], ax[b], ax[b]), ax[b]
)


def configured_5_1(
    op: TensorOp[[T, T, T, T, T], T],
    red: ReducibleTensorOp[[T, T, T, T, T], T],
    x0: T,
    x1: T,
    x2: T,
    x3: T,
    x4: T,
) -> T:
    _ = assert_type(op.with_sizes(b=1)(x0, x1, x2, x3, x4), T)
    return red.reduce_by("sum").with_sizes(b=1)(x0, x1, x2, x3, x4)


op_5_2: TensorOp[[Dummy, Dummy, Dummy, Dummy, Dummy], tuple[Dummy, Dummy]] = rearrange(
    (ax[b], ax[b], ax[b], ax[b], ax[b]), (ax[b], ax[b])
)
red_5_2: ReducibleTensorOp[[Dummy, Dummy, Dummy, Dummy, Dummy], tuple[Dummy, Dummy]] = (
    einop((ax[b], ax[b], ax[b], ax[b], ax[b]), (ax[b], ax[b]))
)


def configured_5_2(
    op: TensorOp[[T, T, T, T, T], tuple[T, T]],
    red: ReducibleTensorOp[[T, T, T, T, T], tuple[T, T]],
    x0: T,
    x1: T,
    x2: T,
    x3: T,
    x4: T,
) -> tuple[T, T]:
    _ = assert_type(op.with_sizes(b=1)(x0, x1, x2, x3, x4), tuple[T, T])
    return red.reduce_by("sum").with_sizes(b=1)(x0, x1, x2, x3, x4)


op_5_3: TensorOp[[Dummy, Dummy, Dummy, Dummy, Dummy], tuple[Dummy, Dummy, Dummy]] = (
    rearrange((ax[b], ax[b], ax[b], ax[b], ax[b]), (ax[b], ax[b], ax[b]))
)
red_5_3: ReducibleTensorOp[
    [Dummy, Dummy, Dummy, Dummy, Dummy], tuple[Dummy, Dummy, Dummy]
] = einop((ax[b], ax[b], ax[b], ax[b], ax[b]), (ax[b], ax[b], ax[b]))


def configured_5_3(
    op: TensorOp[[T, T, T, T, T], tuple[T, T, T]],
    red: ReducibleTensorOp[[T, T, T, T, T], tuple[T, T, T]],
    x0: T,
    x1: T,
    x2: T,
    x3: T,
    x4: T,
) -> tuple[T, T, T]:
    _ = assert_type(op.with_sizes(b=1)(x0, x1, x2, x3, x4), tuple[T, T, T])
    return red.reduce_by("sum").with_sizes(b=1)(x0, x1, x2, x3, x4)


op_5_4: TensorOp[
    [Dummy, Dummy, Dummy, Dummy, Dummy], tuple[Dummy, Dummy, Dummy, Dummy]
] = rearrange((ax[b], ax[b], ax[b], ax[b], ax[b]), (ax[b], ax[b], ax[b], ax[b]))
red_5_4: ReducibleTensorOp[
    [Dummy, Dummy, Dummy, Dummy, Dummy], tuple[Dummy, Dummy, Dummy, Dummy]
] = einop((ax[b], ax[b], ax[b], ax[b], ax[b]), (ax[b], ax[b], ax[b], ax[b]))


def configured_5_4(
    op: TensorOp[[T, T, T, T, T], tuple[T, T, T, T]],
    red: ReducibleTensorOp[[T, T, T, T, T], tuple[T, T, T, T]],
    x0: T,
    x1: T,
    x2: T,
    x3: T,
    x4: T,
) -> tuple[T, T, T, T]:
    _ = assert_type(op.with_sizes(b=1)(x0, x1, x2, x3, x4), tuple[T, T, T, T])
    return red.reduce_by("sum").with_sizes(b=1)(x0, x1, x2, x3, x4)


op_5_5: TensorOp[
    [Dummy, Dummy, Dummy, Dummy, Dummy], tuple[Dummy, Dummy, Dummy, Dummy, Dummy]
] = rearrange((ax[b], ax[b], ax[b], ax[b], ax[b]), (ax[b], ax[b], ax[b], ax[b], ax[b]))
red_5_5: ReducibleTensorOp[
    [Dummy, Dummy, Dummy, Dummy, Dummy], tuple[Dummy, Dummy, Dummy, Dummy, Dummy]
] = einop((ax[b], ax[b], ax[b], ax[b], ax[b]), (ax[b], ax[b], ax[b], ax[b], ax[b]))


def configured_5_5(
    op: TensorOp[[T, T, T, T, T], tuple[T, T, T, T, T]],
    red: ReducibleTensorOp[[T, T, T, T, T], tuple[T, T, T, T, T]],
    x0: T,
    x1: T,
    x2: T,
    x3: T,
    x4: T,
) -> tuple[T, T, T, T, T]:
    _ = assert_type(op.with_sizes(b=1)(x0, x1, x2, x3, x4), tuple[T, T, T, T, T])
    return red.reduce_by("sum").with_sizes(b=1)(x0, x1, x2, x3, x4)


op_5_6: TensorOp[
    [Dummy, Dummy, Dummy, Dummy, Dummy], tuple[Dummy, Dummy, Dummy, Dummy, Dummy, Dummy]
] = rearrange(
    (ax[b], ax[b], ax[b], ax[b], ax[b]), (ax[b], ax[b], ax[b], ax[b], ax[b], ax[b])
)
red_5_6: ReducibleTensorOp[
    [Dummy, Dummy, Dummy, Dummy, Dummy], tuple[Dummy, Dummy, Dummy, Dummy, Dummy, Dummy]
] = einop(
    (ax[b], ax[b], ax[b], ax[b], ax[b]), (ax[b], ax[b], ax[b], ax[b], ax[b], ax[b])
)


def configured_5_6(
    op: TensorOp[[T, T, T, T, T], tuple[T, T, T, T, T, T]],
    red: ReducibleTensorOp[[T, T, T, T, T], tuple[T, T, T, T, T, T]],
    x0: T,
    x1: T,
    x2: T,
    x3: T,
    x4: T,
) -> tuple[T, T, T, T, T, T]:
    _ = assert_type(op.with_sizes(b=1)(x0, x1, x2, x3, x4), tuple[T, T, T, T, T, T])
    return red.reduce_by("sum").with_sizes(b=1)(x0, x1, x2, x3, x4)


op_6_1: TensorOp[[Dummy, Dummy, Dummy, Dummy, Dummy, Dummy], Dummy] = rearrange(
    (ax[b], ax[b], ax[b], ax[b], ax[b], ax[b]), ax[b]
)
red_6_1: ReducibleTensorOp[[Dummy, Dummy, Dummy, Dummy, Dummy, Dummy], Dummy] = einop(
    (ax[b], ax[b], ax[b], ax[b], ax[b], ax[b]), ax[b]
)


def configured_6_1(
    op: TensorOp[[T, T, T, T, T, T], T],
    red: ReducibleTensorOp[[T, T, T, T, T, T], T],
    x0: T,
    x1: T,
    x2: T,
    x3: T,
    x4: T,
    x5: T,
) -> T:
    _ = assert_type(op.with_sizes(b=1)(x0, x1, x2, x3, x4, x5), T)
    return red.reduce_by("sum").with_sizes(b=1)(x0, x1, x2, x3, x4, x5)


op_6_2: TensorOp[[Dummy, Dummy, Dummy, Dummy, Dummy, Dummy], tuple[Dummy, Dummy]] = (
    rearrange((ax[b], ax[b], ax[b], ax[b], ax[b], ax[b]), (ax[b], ax[b]))
)
red_6_2: ReducibleTensorOp[
    [Dummy, Dummy, Dummy, Dummy, Dummy, Dummy], tuple[Dummy, Dummy]
] = einop((ax[b], ax[b], ax[b], ax[b], ax[b], ax[b]), (ax[b], ax[b]))


def configured_6_2(
    op: TensorOp[[T, T, T, T, T, T], tuple[T, T]],
    red: ReducibleTensorOp[[T, T, T, T, T, T], tuple[T, T]],
    x0: T,
    x1: T,
    x2: T,
    x3: T,
    x4: T,
    x5: T,
) -> tuple[T, T]:
    _ = assert_type(op.with_sizes(b=1)(x0, x1, x2, x3, x4, x5), tuple[T, T])
    return red.reduce_by("sum").with_sizes(b=1)(x0, x1, x2, x3, x4, x5)


op_6_3: TensorOp[
    [Dummy, Dummy, Dummy, Dummy, Dummy, Dummy], tuple[Dummy, Dummy, Dummy]
] = rearrange((ax[b], ax[b], ax[b], ax[b], ax[b], ax[b]), (ax[b], ax[b], ax[b]))
red_6_3: ReducibleTensorOp[
    [Dummy, Dummy, Dummy, Dummy, Dummy, Dummy], tuple[Dummy, Dummy, Dummy]
] = einop((ax[b], ax[b], ax[b], ax[b], ax[b], ax[b]), (ax[b], ax[b], ax[b]))


def configured_6_3(
    op: TensorOp[[T, T, T, T, T, T], tuple[T, T, T]],
    red: ReducibleTensorOp[[T, T, T, T, T, T], tuple[T, T, T]],
    x0: T,
    x1: T,
    x2: T,
    x3: T,
    x4: T,
    x5: T,
) -> tuple[T, T, T]:
    _ = assert_type(op.with_sizes(b=1)(x0, x1, x2, x3, x4, x5), tuple[T, T, T])
    return red.reduce_by("sum").with_sizes(b=1)(x0, x1, x2, x3, x4, x5)


op_6_4: TensorOp[
    [Dummy, Dummy, Dummy, Dummy, Dummy, Dummy], tuple[Dummy, Dummy, Dummy, Dummy]
] = rearrange((ax[b], ax[b], ax[b], ax[b], ax[b], ax[b]), (ax[b], ax[b], ax[b], ax[b]))
red_6_4: ReducibleTensorOp[
    [Dummy, Dummy, Dummy, Dummy, Dummy, Dummy], tuple[Dummy, Dummy, Dummy, Dummy]
] = einop((ax[b], ax[b], ax[b], ax[b], ax[b], ax[b]), (ax[b], ax[b], ax[b], ax[b]))


def configured_6_4(
    op: TensorOp[[T, T, T, T, T, T], tuple[T, T, T, T]],
    red: ReducibleTensorOp[[T, T, T, T, T, T], tuple[T, T, T, T]],
    x0: T,
    x1: T,
    x2: T,
    x3: T,
    x4: T,
    x5: T,
) -> tuple[T, T, T, T]:
    _ = assert_type(op.with_sizes(b=1)(x0, x1, x2, x3, x4, x5), tuple[T, T, T, T])
    return red.reduce_by("sum").with_sizes(b=1)(x0, x1, x2, x3, x4, x5)


op_6_5: TensorOp[
    [Dummy, Dummy, Dummy, Dummy, Dummy, Dummy], tuple[Dummy, Dummy, Dummy, Dummy, Dummy]
] = rearrange(
    (ax[b], ax[b], ax[b], ax[b], ax[b], ax[b]), (ax[b], ax[b], ax[b], ax[b], ax[b])
)
red_6_5: ReducibleTensorOp[
    [Dummy, Dummy, Dummy, Dummy, Dummy, Dummy], tuple[Dummy, Dummy, Dummy, Dummy, Dummy]
] = einop(
    (ax[b], ax[b], ax[b], ax[b], ax[b], ax[b]), (ax[b], ax[b], ax[b], ax[b], ax[b])
)


def configured_6_5(
    op: TensorOp[[T, T, T, T, T, T], tuple[T, T, T, T, T]],
    red: ReducibleTensorOp[[T, T, T, T, T, T], tuple[T, T, T, T, T]],
    x0: T,
    x1: T,
    x2: T,
    x3: T,
    x4: T,
    x5: T,
) -> tuple[T, T, T, T, T]:
    _ = assert_type(op.with_sizes(b=1)(x0, x1, x2, x3, x4, x5), tuple[T, T, T, T, T])
    return red.reduce_by("sum").with_sizes(b=1)(x0, x1, x2, x3, x4, x5)


op_6_6: TensorOp[
    [Dummy, Dummy, Dummy, Dummy, Dummy, Dummy],
    tuple[Dummy, Dummy, Dummy, Dummy, Dummy, Dummy],
] = rearrange(
    (ax[b], ax[b], ax[b], ax[b], ax[b], ax[b]),
    (ax[b], ax[b], ax[b], ax[b], ax[b], ax[b]),
)
red_6_6: ReducibleTensorOp[
    [Dummy, Dummy, Dummy, Dummy, Dummy, Dummy],
    tuple[Dummy, Dummy, Dummy, Dummy, Dummy, Dummy],
] = einop(
    (ax[b], ax[b], ax[b], ax[b], ax[b], ax[b]),
    (ax[b], ax[b], ax[b], ax[b], ax[b], ax[b]),
)


def configured_6_6(
    op: TensorOp[[T, T, T, T, T, T], tuple[T, T, T, T, T, T]],
    red: ReducibleTensorOp[[T, T, T, T, T, T], tuple[T, T, T, T, T, T]],
    x0: T,
    x1: T,
    x2: T,
    x3: T,
    x4: T,
    x5: T,
) -> tuple[T, T, T, T, T, T]:
    _ = assert_type(op.with_sizes(b=1)(x0, x1, x2, x3, x4, x5), tuple[T, T, T, T, T, T])
    return red.reduce_by("sum").with_sizes(b=1)(x0, x1, x2, x3, x4, x5)
