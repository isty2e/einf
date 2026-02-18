from typing import Protocol, TypeAlias, TypeVar, overload

from typing_extensions import Self, Unpack

from ..axis import AxisSide, AxisTerms
from ..plans.abstract import AbstractPlan
from ..plans.render import PlanDict
from ..reduction.schema import ReducerCallable, ReducerPlan
from ..signature import Signature
from ..tensor_types import TensorLike

# Type-surface policy:
# 1. exact arity overloads are enumerated up to N_max = 6,
# 2. beyond N_max, fallback overloads preserve 1-1 / 1-N / N-1 / N-N quadrants.
_TensorFamily = TypeVar("_TensorFamily", bound=TensorLike)

_Side1: TypeAlias = AxisTerms
_Side2: TypeAlias = tuple[AxisTerms, AxisTerms]
_Side3: TypeAlias = tuple[AxisTerms, AxisTerms, AxisTerms]
_Side4: TypeAlias = tuple[AxisTerms, AxisTerms, AxisTerms, AxisTerms]
_Side5: TypeAlias = tuple[AxisTerms, AxisTerms, AxisTerms, AxisTerms, AxisTerms]
_Side6: TypeAlias = tuple[
    AxisTerms, AxisTerms, AxisTerms, AxisTerms, AxisTerms, AxisTerms
]
_Side7Plus: TypeAlias = tuple[
    AxisTerms,
    AxisTerms,
    AxisTerms,
    AxisTerms,
    AxisTerms,
    AxisTerms,
    AxisTerms,
    Unpack[tuple[AxisTerms, ...]],
]

class _TensorOpSurface(Protocol):
    signature: Signature
    abstract_plan: AbstractPlan
    lhs: AxisSide
    rhs: AxisSide
    sizes: dict[str, int]
    reducer_plan: ReducerPlan | None

    def with_sizes(self, **sizes: int) -> Self: ...
    def plan(self) -> str: ...
    def plan_dict(self) -> PlanDict: ...

class TensorOp(_TensorOpSurface, Protocol): ...

class _ReducerTensorOpSurface(_TensorOpSurface, Protocol):
    @overload
    def reduce_by(self, reducer: str) -> Self: ...
    @overload
    def reduce_by(self, reducer: ReducerCallable) -> Self: ...
    @overload
    def reduce_by(
        self, phase: tuple[AxisTerms, str], *phases: tuple[AxisTerms, str]
    ) -> Self: ...
    @overload
    def reduce_by(
        self,
        phase: tuple[AxisTerms, ReducerCallable],
        *phases: tuple[AxisTerms, ReducerCallable],
    ) -> Self: ...
    @overload
    def reduce_by(
        self,
        phase: tuple[AxisTerms, str | ReducerCallable],
        *phases: tuple[AxisTerms, str | ReducerCallable],
    ) -> Self: ...

class _TensorOp_1_1(_TensorOpSurface, Protocol):
    def __call__(self, x1: _TensorFamily, /) -> _TensorFamily: ...

class _ReduceTensorOp_1_1(_TensorOp_1_1, _ReducerTensorOpSurface, Protocol): ...

class _TensorOp_1_2(_TensorOpSurface, Protocol):
    def __call__(self, x1: _TensorFamily, /) -> tuple[_TensorFamily, _TensorFamily]: ...

class _TensorOp_1_3(_TensorOpSurface, Protocol):
    def __call__(
        self, x1: _TensorFamily, /
    ) -> tuple[_TensorFamily, _TensorFamily, _TensorFamily]: ...

class _TensorOp_1_4(_TensorOpSurface, Protocol):
    def __call__(
        self, x1: _TensorFamily, /
    ) -> tuple[_TensorFamily, _TensorFamily, _TensorFamily, _TensorFamily]: ...

class _TensorOp_1_5(_TensorOpSurface, Protocol):
    def __call__(
        self, x1: _TensorFamily, /
    ) -> tuple[
        _TensorFamily, _TensorFamily, _TensorFamily, _TensorFamily, _TensorFamily
    ]: ...

class _TensorOp_1_6(_TensorOpSurface, Protocol):
    def __call__(
        self, x1: _TensorFamily, /
    ) -> tuple[
        _TensorFamily,
        _TensorFamily,
        _TensorFamily,
        _TensorFamily,
        _TensorFamily,
        _TensorFamily,
    ]: ...

class _TensorOp_2_1(_TensorOpSurface, Protocol):
    def __call__(self, x1: _TensorFamily, x2: _TensorFamily, /) -> _TensorFamily: ...

class _TensorOp_2_2(_TensorOpSurface, Protocol):
    def __call__(
        self, x1: _TensorFamily, x2: _TensorFamily, /
    ) -> tuple[_TensorFamily, _TensorFamily]: ...

class _TensorOp_2_3(_TensorOpSurface, Protocol):
    def __call__(
        self, x1: _TensorFamily, x2: _TensorFamily, /
    ) -> tuple[_TensorFamily, _TensorFamily, _TensorFamily]: ...

class _TensorOp_2_4(_TensorOpSurface, Protocol):
    def __call__(
        self, x1: _TensorFamily, x2: _TensorFamily, /
    ) -> tuple[_TensorFamily, _TensorFamily, _TensorFamily, _TensorFamily]: ...

class _TensorOp_2_5(_TensorOpSurface, Protocol):
    def __call__(
        self, x1: _TensorFamily, x2: _TensorFamily, /
    ) -> tuple[
        _TensorFamily, _TensorFamily, _TensorFamily, _TensorFamily, _TensorFamily
    ]: ...

class _TensorOp_2_6(_TensorOpSurface, Protocol):
    def __call__(
        self, x1: _TensorFamily, x2: _TensorFamily, /
    ) -> tuple[
        _TensorFamily,
        _TensorFamily,
        _TensorFamily,
        _TensorFamily,
        _TensorFamily,
        _TensorFamily,
    ]: ...

class _TensorOp_3_1(_TensorOpSurface, Protocol):
    def __call__(
        self, x1: _TensorFamily, x2: _TensorFamily, x3: _TensorFamily, /
    ) -> _TensorFamily: ...

class _TensorOp_3_2(_TensorOpSurface, Protocol):
    def __call__(
        self, x1: _TensorFamily, x2: _TensorFamily, x3: _TensorFamily, /
    ) -> tuple[_TensorFamily, _TensorFamily]: ...

class _TensorOp_3_3(_TensorOpSurface, Protocol):
    def __call__(
        self, x1: _TensorFamily, x2: _TensorFamily, x3: _TensorFamily, /
    ) -> tuple[_TensorFamily, _TensorFamily, _TensorFamily]: ...

class _TensorOp_3_4(_TensorOpSurface, Protocol):
    def __call__(
        self, x1: _TensorFamily, x2: _TensorFamily, x3: _TensorFamily, /
    ) -> tuple[_TensorFamily, _TensorFamily, _TensorFamily, _TensorFamily]: ...

class _TensorOp_3_5(_TensorOpSurface, Protocol):
    def __call__(
        self, x1: _TensorFamily, x2: _TensorFamily, x3: _TensorFamily, /
    ) -> tuple[
        _TensorFamily, _TensorFamily, _TensorFamily, _TensorFamily, _TensorFamily
    ]: ...

class _TensorOp_3_6(_TensorOpSurface, Protocol):
    def __call__(
        self, x1: _TensorFamily, x2: _TensorFamily, x3: _TensorFamily, /
    ) -> tuple[
        _TensorFamily,
        _TensorFamily,
        _TensorFamily,
        _TensorFamily,
        _TensorFamily,
        _TensorFamily,
    ]: ...

class _TensorOp_4_1(_TensorOpSurface, Protocol):
    def __call__(
        self,
        x1: _TensorFamily,
        x2: _TensorFamily,
        x3: _TensorFamily,
        x4: _TensorFamily,
        /,
    ) -> _TensorFamily: ...

class _TensorOp_4_2(_TensorOpSurface, Protocol):
    def __call__(
        self,
        x1: _TensorFamily,
        x2: _TensorFamily,
        x3: _TensorFamily,
        x4: _TensorFamily,
        /,
    ) -> tuple[_TensorFamily, _TensorFamily]: ...

class _TensorOp_4_3(_TensorOpSurface, Protocol):
    def __call__(
        self,
        x1: _TensorFamily,
        x2: _TensorFamily,
        x3: _TensorFamily,
        x4: _TensorFamily,
        /,
    ) -> tuple[_TensorFamily, _TensorFamily, _TensorFamily]: ...

class _TensorOp_4_4(_TensorOpSurface, Protocol):
    def __call__(
        self,
        x1: _TensorFamily,
        x2: _TensorFamily,
        x3: _TensorFamily,
        x4: _TensorFamily,
        /,
    ) -> tuple[_TensorFamily, _TensorFamily, _TensorFamily, _TensorFamily]: ...

class _TensorOp_4_5(_TensorOpSurface, Protocol):
    def __call__(
        self,
        x1: _TensorFamily,
        x2: _TensorFamily,
        x3: _TensorFamily,
        x4: _TensorFamily,
        /,
    ) -> tuple[
        _TensorFamily, _TensorFamily, _TensorFamily, _TensorFamily, _TensorFamily
    ]: ...

class _TensorOp_4_6(_TensorOpSurface, Protocol):
    def __call__(
        self,
        x1: _TensorFamily,
        x2: _TensorFamily,
        x3: _TensorFamily,
        x4: _TensorFamily,
        /,
    ) -> tuple[
        _TensorFamily,
        _TensorFamily,
        _TensorFamily,
        _TensorFamily,
        _TensorFamily,
        _TensorFamily,
    ]: ...

class _TensorOp_5_1(_TensorOpSurface, Protocol):
    def __call__(
        self,
        x1: _TensorFamily,
        x2: _TensorFamily,
        x3: _TensorFamily,
        x4: _TensorFamily,
        x5: _TensorFamily,
        /,
    ) -> _TensorFamily: ...

class _TensorOp_5_2(_TensorOpSurface, Protocol):
    def __call__(
        self,
        x1: _TensorFamily,
        x2: _TensorFamily,
        x3: _TensorFamily,
        x4: _TensorFamily,
        x5: _TensorFamily,
        /,
    ) -> tuple[_TensorFamily, _TensorFamily]: ...

class _TensorOp_5_3(_TensorOpSurface, Protocol):
    def __call__(
        self,
        x1: _TensorFamily,
        x2: _TensorFamily,
        x3: _TensorFamily,
        x4: _TensorFamily,
        x5: _TensorFamily,
        /,
    ) -> tuple[_TensorFamily, _TensorFamily, _TensorFamily]: ...

class _TensorOp_5_4(_TensorOpSurface, Protocol):
    def __call__(
        self,
        x1: _TensorFamily,
        x2: _TensorFamily,
        x3: _TensorFamily,
        x4: _TensorFamily,
        x5: _TensorFamily,
        /,
    ) -> tuple[_TensorFamily, _TensorFamily, _TensorFamily, _TensorFamily]: ...

class _TensorOp_5_5(_TensorOpSurface, Protocol):
    def __call__(
        self,
        x1: _TensorFamily,
        x2: _TensorFamily,
        x3: _TensorFamily,
        x4: _TensorFamily,
        x5: _TensorFamily,
        /,
    ) -> tuple[
        _TensorFamily, _TensorFamily, _TensorFamily, _TensorFamily, _TensorFamily
    ]: ...

class _TensorOp_5_6(_TensorOpSurface, Protocol):
    def __call__(
        self,
        x1: _TensorFamily,
        x2: _TensorFamily,
        x3: _TensorFamily,
        x4: _TensorFamily,
        x5: _TensorFamily,
        /,
    ) -> tuple[
        _TensorFamily,
        _TensorFamily,
        _TensorFamily,
        _TensorFamily,
        _TensorFamily,
        _TensorFamily,
    ]: ...

class _TensorOp_6_1(_TensorOpSurface, Protocol):
    def __call__(
        self,
        x1: _TensorFamily,
        x2: _TensorFamily,
        x3: _TensorFamily,
        x4: _TensorFamily,
        x5: _TensorFamily,
        x6: _TensorFamily,
        /,
    ) -> _TensorFamily: ...

class _TensorOp_6_2(_TensorOpSurface, Protocol):
    def __call__(
        self,
        x1: _TensorFamily,
        x2: _TensorFamily,
        x3: _TensorFamily,
        x4: _TensorFamily,
        x5: _TensorFamily,
        x6: _TensorFamily,
        /,
    ) -> tuple[_TensorFamily, _TensorFamily]: ...

class _TensorOp_6_3(_TensorOpSurface, Protocol):
    def __call__(
        self,
        x1: _TensorFamily,
        x2: _TensorFamily,
        x3: _TensorFamily,
        x4: _TensorFamily,
        x5: _TensorFamily,
        x6: _TensorFamily,
        /,
    ) -> tuple[_TensorFamily, _TensorFamily, _TensorFamily]: ...

class _TensorOp_6_4(_TensorOpSurface, Protocol):
    def __call__(
        self,
        x1: _TensorFamily,
        x2: _TensorFamily,
        x3: _TensorFamily,
        x4: _TensorFamily,
        x5: _TensorFamily,
        x6: _TensorFamily,
        /,
    ) -> tuple[_TensorFamily, _TensorFamily, _TensorFamily, _TensorFamily]: ...

class _TensorOp_6_5(_TensorOpSurface, Protocol):
    def __call__(
        self,
        x1: _TensorFamily,
        x2: _TensorFamily,
        x3: _TensorFamily,
        x4: _TensorFamily,
        x5: _TensorFamily,
        x6: _TensorFamily,
        /,
    ) -> tuple[
        _TensorFamily, _TensorFamily, _TensorFamily, _TensorFamily, _TensorFamily
    ]: ...

class _TensorOp_6_6(_TensorOpSurface, Protocol):
    def __call__(
        self,
        x1: _TensorFamily,
        x2: _TensorFamily,
        x3: _TensorFamily,
        x4: _TensorFamily,
        x5: _TensorFamily,
        x6: _TensorFamily,
        /,
    ) -> tuple[
        _TensorFamily,
        _TensorFamily,
        _TensorFamily,
        _TensorFamily,
        _TensorFamily,
        _TensorFamily,
    ]: ...

class _TensorOp_1_N(_TensorOpSurface, Protocol):
    def __call__(self, x1: _TensorFamily, /) -> tuple[_TensorFamily, ...]: ...

class _TensorOp_N_1(_TensorOpSurface, Protocol):
    def __call__(
        self, x1: _TensorFamily, x2: _TensorFamily, /, *xs: _TensorFamily
    ) -> _TensorFamily: ...

class _TensorOp_N_N(_TensorOpSurface, Protocol):
    def __call__(
        self, x1: _TensorFamily, x2: _TensorFamily, /, *xs: _TensorFamily
    ) -> tuple[_TensorFamily, ...]: ...

class _EinopTensorOp_1_1(_TensorOp_1_1, _ReducerTensorOpSurface, Protocol): ...
class _EinopTensorOp_1_2(_TensorOp_1_2, _ReducerTensorOpSurface, Protocol): ...
class _EinopTensorOp_1_3(_TensorOp_1_3, _ReducerTensorOpSurface, Protocol): ...
class _EinopTensorOp_1_4(_TensorOp_1_4, _ReducerTensorOpSurface, Protocol): ...
class _EinopTensorOp_1_5(_TensorOp_1_5, _ReducerTensorOpSurface, Protocol): ...
class _EinopTensorOp_1_6(_TensorOp_1_6, _ReducerTensorOpSurface, Protocol): ...
class _EinopTensorOp_2_1(_TensorOp_2_1, _ReducerTensorOpSurface, Protocol): ...
class _EinopTensorOp_2_2(_TensorOp_2_2, _ReducerTensorOpSurface, Protocol): ...
class _EinopTensorOp_2_3(_TensorOp_2_3, _ReducerTensorOpSurface, Protocol): ...
class _EinopTensorOp_2_4(_TensorOp_2_4, _ReducerTensorOpSurface, Protocol): ...
class _EinopTensorOp_2_5(_TensorOp_2_5, _ReducerTensorOpSurface, Protocol): ...
class _EinopTensorOp_2_6(_TensorOp_2_6, _ReducerTensorOpSurface, Protocol): ...
class _EinopTensorOp_3_1(_TensorOp_3_1, _ReducerTensorOpSurface, Protocol): ...
class _EinopTensorOp_3_2(_TensorOp_3_2, _ReducerTensorOpSurface, Protocol): ...
class _EinopTensorOp_3_3(_TensorOp_3_3, _ReducerTensorOpSurface, Protocol): ...
class _EinopTensorOp_3_4(_TensorOp_3_4, _ReducerTensorOpSurface, Protocol): ...
class _EinopTensorOp_3_5(_TensorOp_3_5, _ReducerTensorOpSurface, Protocol): ...
class _EinopTensorOp_3_6(_TensorOp_3_6, _ReducerTensorOpSurface, Protocol): ...
class _EinopTensorOp_4_1(_TensorOp_4_1, _ReducerTensorOpSurface, Protocol): ...
class _EinopTensorOp_4_2(_TensorOp_4_2, _ReducerTensorOpSurface, Protocol): ...
class _EinopTensorOp_4_3(_TensorOp_4_3, _ReducerTensorOpSurface, Protocol): ...
class _EinopTensorOp_4_4(_TensorOp_4_4, _ReducerTensorOpSurface, Protocol): ...
class _EinopTensorOp_4_5(_TensorOp_4_5, _ReducerTensorOpSurface, Protocol): ...
class _EinopTensorOp_4_6(_TensorOp_4_6, _ReducerTensorOpSurface, Protocol): ...
class _EinopTensorOp_5_1(_TensorOp_5_1, _ReducerTensorOpSurface, Protocol): ...
class _EinopTensorOp_5_2(_TensorOp_5_2, _ReducerTensorOpSurface, Protocol): ...
class _EinopTensorOp_5_3(_TensorOp_5_3, _ReducerTensorOpSurface, Protocol): ...
class _EinopTensorOp_5_4(_TensorOp_5_4, _ReducerTensorOpSurface, Protocol): ...
class _EinopTensorOp_5_5(_TensorOp_5_5, _ReducerTensorOpSurface, Protocol): ...
class _EinopTensorOp_5_6(_TensorOp_5_6, _ReducerTensorOpSurface, Protocol): ...
class _EinopTensorOp_6_1(_TensorOp_6_1, _ReducerTensorOpSurface, Protocol): ...
class _EinopTensorOp_6_2(_TensorOp_6_2, _ReducerTensorOpSurface, Protocol): ...
class _EinopTensorOp_6_3(_TensorOp_6_3, _ReducerTensorOpSurface, Protocol): ...
class _EinopTensorOp_6_4(_TensorOp_6_4, _ReducerTensorOpSurface, Protocol): ...
class _EinopTensorOp_6_5(_TensorOp_6_5, _ReducerTensorOpSurface, Protocol): ...
class _EinopTensorOp_6_6(_TensorOp_6_6, _ReducerTensorOpSurface, Protocol): ...
class _EinopTensorOp_1_N(_TensorOp_1_N, _ReducerTensorOpSurface, Protocol): ...
class _EinopTensorOp_N_1(_TensorOp_N_1, _ReducerTensorOpSurface, Protocol): ...
class _EinopTensorOp_N_N(_TensorOp_N_N, _ReducerTensorOpSurface, Protocol): ...

@overload
def view(lhs: _Side1, rhs: _Side1) -> _TensorOp_1_1: ...
@overload
def view(lhs: _Side1, rhs: _Side2) -> _TensorOp_1_2: ...
@overload
def view(lhs: _Side1, rhs: _Side3) -> _TensorOp_1_3: ...
@overload
def view(lhs: _Side1, rhs: _Side4) -> _TensorOp_1_4: ...
@overload
def view(lhs: _Side1, rhs: _Side5) -> _TensorOp_1_5: ...
@overload
def view(lhs: _Side1, rhs: _Side6) -> _TensorOp_1_6: ...
@overload
def view(lhs: _Side1, rhs: _Side7Plus) -> _TensorOp_1_N: ...
@overload
def rearrange(lhs: _Side1, rhs: _Side1) -> _TensorOp_1_1: ...
@overload
def rearrange(lhs: _Side1, rhs: _Side2) -> _TensorOp_1_2: ...
@overload
def rearrange(lhs: _Side1, rhs: _Side3) -> _TensorOp_1_3: ...
@overload
def rearrange(lhs: _Side1, rhs: _Side4) -> _TensorOp_1_4: ...
@overload
def rearrange(lhs: _Side1, rhs: _Side5) -> _TensorOp_1_5: ...
@overload
def rearrange(lhs: _Side1, rhs: _Side6) -> _TensorOp_1_6: ...
@overload
def rearrange(lhs: _Side2, rhs: _Side1) -> _TensorOp_2_1: ...
@overload
def rearrange(lhs: _Side2, rhs: _Side2) -> _TensorOp_2_2: ...
@overload
def rearrange(lhs: _Side2, rhs: _Side3) -> _TensorOp_2_3: ...
@overload
def rearrange(lhs: _Side2, rhs: _Side4) -> _TensorOp_2_4: ...
@overload
def rearrange(lhs: _Side2, rhs: _Side5) -> _TensorOp_2_5: ...
@overload
def rearrange(lhs: _Side2, rhs: _Side6) -> _TensorOp_2_6: ...
@overload
def rearrange(lhs: _Side3, rhs: _Side1) -> _TensorOp_3_1: ...
@overload
def rearrange(lhs: _Side3, rhs: _Side2) -> _TensorOp_3_2: ...
@overload
def rearrange(lhs: _Side3, rhs: _Side3) -> _TensorOp_3_3: ...
@overload
def rearrange(lhs: _Side3, rhs: _Side4) -> _TensorOp_3_4: ...
@overload
def rearrange(lhs: _Side3, rhs: _Side5) -> _TensorOp_3_5: ...
@overload
def rearrange(lhs: _Side3, rhs: _Side6) -> _TensorOp_3_6: ...
@overload
def rearrange(lhs: _Side4, rhs: _Side1) -> _TensorOp_4_1: ...
@overload
def rearrange(lhs: _Side4, rhs: _Side2) -> _TensorOp_4_2: ...
@overload
def rearrange(lhs: _Side4, rhs: _Side3) -> _TensorOp_4_3: ...
@overload
def rearrange(lhs: _Side4, rhs: _Side4) -> _TensorOp_4_4: ...
@overload
def rearrange(lhs: _Side4, rhs: _Side5) -> _TensorOp_4_5: ...
@overload
def rearrange(lhs: _Side4, rhs: _Side6) -> _TensorOp_4_6: ...
@overload
def rearrange(lhs: _Side5, rhs: _Side1) -> _TensorOp_5_1: ...
@overload
def rearrange(lhs: _Side5, rhs: _Side2) -> _TensorOp_5_2: ...
@overload
def rearrange(lhs: _Side5, rhs: _Side3) -> _TensorOp_5_3: ...
@overload
def rearrange(lhs: _Side5, rhs: _Side4) -> _TensorOp_5_4: ...
@overload
def rearrange(lhs: _Side5, rhs: _Side5) -> _TensorOp_5_5: ...
@overload
def rearrange(lhs: _Side5, rhs: _Side6) -> _TensorOp_5_6: ...
@overload
def rearrange(lhs: _Side6, rhs: _Side1) -> _TensorOp_6_1: ...
@overload
def rearrange(lhs: _Side6, rhs: _Side2) -> _TensorOp_6_2: ...
@overload
def rearrange(lhs: _Side6, rhs: _Side3) -> _TensorOp_6_3: ...
@overload
def rearrange(lhs: _Side6, rhs: _Side4) -> _TensorOp_6_4: ...
@overload
def rearrange(lhs: _Side6, rhs: _Side5) -> _TensorOp_6_5: ...
@overload
def rearrange(lhs: _Side6, rhs: _Side6) -> _TensorOp_6_6: ...
@overload
def rearrange(lhs: _Side1, rhs: _Side7Plus) -> _TensorOp_1_N: ...
@overload
def rearrange(lhs: _Side7Plus, rhs: _Side1) -> _TensorOp_N_1: ...
@overload
def rearrange(lhs: _Side7Plus, rhs: _Side7Plus) -> _TensorOp_N_N: ...
def repeat(lhs: _Side1, rhs: _Side1) -> _TensorOp_1_1: ...
def reduce(lhs: _Side1, rhs: _Side1) -> _ReduceTensorOp_1_1: ...
@overload
def contract(lhs: _Side1, rhs: _Side1) -> _TensorOp_1_1: ...
@overload
def contract(lhs: _Side2, rhs: _Side1) -> _TensorOp_2_1: ...
@overload
def contract(lhs: _Side3, rhs: _Side1) -> _TensorOp_3_1: ...
@overload
def contract(lhs: _Side4, rhs: _Side1) -> _TensorOp_4_1: ...
@overload
def contract(lhs: _Side5, rhs: _Side1) -> _TensorOp_5_1: ...
@overload
def contract(lhs: _Side6, rhs: _Side1) -> _TensorOp_6_1: ...
@overload
def contract(lhs: _Side7Plus, rhs: _Side1) -> _TensorOp_N_1: ...
@overload
def einop(lhs: _Side1, rhs: _Side1) -> _EinopTensorOp_1_1: ...
@overload
def einop(lhs: _Side1, rhs: _Side2) -> _EinopTensorOp_1_2: ...
@overload
def einop(lhs: _Side1, rhs: _Side3) -> _EinopTensorOp_1_3: ...
@overload
def einop(lhs: _Side1, rhs: _Side4) -> _EinopTensorOp_1_4: ...
@overload
def einop(lhs: _Side1, rhs: _Side5) -> _EinopTensorOp_1_5: ...
@overload
def einop(lhs: _Side1, rhs: _Side6) -> _EinopTensorOp_1_6: ...
@overload
def einop(lhs: _Side2, rhs: _Side1) -> _EinopTensorOp_2_1: ...
@overload
def einop(lhs: _Side2, rhs: _Side2) -> _EinopTensorOp_2_2: ...
@overload
def einop(lhs: _Side2, rhs: _Side3) -> _EinopTensorOp_2_3: ...
@overload
def einop(lhs: _Side2, rhs: _Side4) -> _EinopTensorOp_2_4: ...
@overload
def einop(lhs: _Side2, rhs: _Side5) -> _EinopTensorOp_2_5: ...
@overload
def einop(lhs: _Side2, rhs: _Side6) -> _EinopTensorOp_2_6: ...
@overload
def einop(lhs: _Side3, rhs: _Side1) -> _EinopTensorOp_3_1: ...
@overload
def einop(lhs: _Side3, rhs: _Side2) -> _EinopTensorOp_3_2: ...
@overload
def einop(lhs: _Side3, rhs: _Side3) -> _EinopTensorOp_3_3: ...
@overload
def einop(lhs: _Side3, rhs: _Side4) -> _EinopTensorOp_3_4: ...
@overload
def einop(lhs: _Side3, rhs: _Side5) -> _EinopTensorOp_3_5: ...
@overload
def einop(lhs: _Side3, rhs: _Side6) -> _EinopTensorOp_3_6: ...
@overload
def einop(lhs: _Side4, rhs: _Side1) -> _EinopTensorOp_4_1: ...
@overload
def einop(lhs: _Side4, rhs: _Side2) -> _EinopTensorOp_4_2: ...
@overload
def einop(lhs: _Side4, rhs: _Side3) -> _EinopTensorOp_4_3: ...
@overload
def einop(lhs: _Side4, rhs: _Side4) -> _EinopTensorOp_4_4: ...
@overload
def einop(lhs: _Side4, rhs: _Side5) -> _EinopTensorOp_4_5: ...
@overload
def einop(lhs: _Side4, rhs: _Side6) -> _EinopTensorOp_4_6: ...
@overload
def einop(lhs: _Side5, rhs: _Side1) -> _EinopTensorOp_5_1: ...
@overload
def einop(lhs: _Side5, rhs: _Side2) -> _EinopTensorOp_5_2: ...
@overload
def einop(lhs: _Side5, rhs: _Side3) -> _EinopTensorOp_5_3: ...
@overload
def einop(lhs: _Side5, rhs: _Side4) -> _EinopTensorOp_5_4: ...
@overload
def einop(lhs: _Side5, rhs: _Side5) -> _EinopTensorOp_5_5: ...
@overload
def einop(lhs: _Side5, rhs: _Side6) -> _EinopTensorOp_5_6: ...
@overload
def einop(lhs: _Side6, rhs: _Side1) -> _EinopTensorOp_6_1: ...
@overload
def einop(lhs: _Side6, rhs: _Side2) -> _EinopTensorOp_6_2: ...
@overload
def einop(lhs: _Side6, rhs: _Side3) -> _EinopTensorOp_6_3: ...
@overload
def einop(lhs: _Side6, rhs: _Side4) -> _EinopTensorOp_6_4: ...
@overload
def einop(lhs: _Side6, rhs: _Side5) -> _EinopTensorOp_6_5: ...
@overload
def einop(lhs: _Side6, rhs: _Side6) -> _EinopTensorOp_6_6: ...
@overload
def einop(lhs: _Side1, rhs: _Side7Plus) -> _EinopTensorOp_1_N: ...
@overload
def einop(lhs: _Side7Plus, rhs: _Side1) -> _EinopTensorOp_N_1: ...
@overload
def einop(lhs: _Side7Plus, rhs: _Side7Plus) -> _EinopTensorOp_N_N: ...
def view(lhs: object, rhs: object) -> _TensorOp_N_N: ...
def rearrange(lhs: object, rhs: object) -> _TensorOp_N_N: ...
def contract(lhs: object, rhs: object) -> _TensorOp_N_N: ...
def einop(lhs: object, rhs: object) -> _EinopTensorOp_N_N: ...
