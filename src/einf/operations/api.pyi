import sys
from typing import Protocol, overload, type_check_only

if sys.version_info >= (3, 13):
    from typing import ParamSpec, Self, TypeVar, Unpack
else:
    from typing_extensions import ParamSpec, Self, TypeVar, Unpack

from ..axis import AxisSide, AxisTerms
from ..plans.abstract import AbstractPlan
from ..plans.render import PlanDict
from ..reduction.callable import ReducerCallable
from ..reduction.schema import ReducerPlan
from ..signature import Signature
from ..tensor_types import TensorLike

# Type-surface policy:
# 1. exact arity overloads are enumerated up to N_max = 6,
# 2. beyond N_max, fallback overloads preserve 1-1 / 1-N / N-1 / N-N quadrants.
_TensorFamily = TypeVar("_TensorFamily", bound=TensorLike)
_P = ParamSpec("_P", default=...)
_R_co = TypeVar("_R_co", covariant=True, default=object)

@type_check_only
class TensorOpMetadata(Protocol):
    @property
    def name(self) -> str: ...
    @property
    def signature(self) -> Signature: ...
    @property
    def abstract_plan(self) -> AbstractPlan: ...
    @property
    def lhs(self) -> AxisSide: ...
    @property
    def rhs(self) -> AxisSide: ...
    @property
    def sizes(self) -> dict[str, int]: ...
    @property
    def sizes_items(self) -> tuple[tuple[str, int], ...]: ...
    @property
    def supports_reducer(self) -> bool: ...
    @property
    def reducer_plan(self) -> ReducerPlan | None: ...
    def with_sizes(self, **sizes: int) -> Self: ...
    def plan(self) -> str: ...
    def plan_dict(self) -> PlanDict: ...

@type_check_only
class ReducerCustomization(Protocol):
    @overload
    def reduce_by(self, reducer: str) -> Self: ...
    @overload
    def reduce_by(self, reducer: ReducerCallable) -> Self: ...
    @overload
    def reduce_by(
        self, reducer: tuple[AxisTerms, str], *phases: tuple[AxisTerms, str]
    ) -> Self: ...
    @overload
    def reduce_by(
        self,
        reducer: tuple[AxisTerms, ReducerCallable],
        *phases: tuple[AxisTerms, ReducerCallable],
    ) -> Self: ...
    @overload
    def reduce_by(
        self,
        reducer: tuple[AxisTerms, str | ReducerCallable],
        *phases: tuple[AxisTerms, str | ReducerCallable],
    ) -> Self: ...

class TensorOp(TensorOpMetadata, Protocol[_P, _R_co]):
    """Annotate an operation's positional input types and result type.

    Notes
    -----
    Use ``TensorOp[[T], R]`` for a unary operation or ``TensorOp[[T, T], R]``
    for a binary operation. Omitting type arguments retains metadata but
    leaves the result typed as ``object``.
    """
    def __call__(self, *tensors: _P.args, **kwargs: _P.kwargs) -> _R_co:
        """Apply an operation with the annotated call signature.

        Parameters
        ----------
        tensors : P.args
            Positional tensor inputs declared by the annotation.
        kwargs : P.kwargs
            ParamSpec keyword projection. List-form annotations declare
            positional inputs only; factory operations reject keyword inputs.

        Returns
        -------
        R
            Tensor or tuple of tensors declared by the result annotation.
        """

class ReducibleTensorOp(TensorOp[_P, _R_co], ReducerCustomization, Protocol[_P, _R_co]):
    """Retain reducer customization in an annotated operation.

    Notes
    -----
    This is a distinct static capability view but aliases ``TensorOp`` at
    runtime. Use ``supports_reducer`` to inspect runtime reducer support.
    """

@type_check_only
class TensorOpInference1To1(TensorOpMetadata, Protocol):
    def __call__(self, x1: _TensorFamily, /) -> _TensorFamily: ...

@type_check_only
class ReduceTensorOpInference1To1(
    TensorOpInference1To1, ReducerCustomization, Protocol
): ...

@type_check_only
class TensorOpInference1To2(TensorOpMetadata, Protocol):
    def __call__(self, x1: _TensorFamily, /) -> tuple[_TensorFamily, _TensorFamily]: ...

@type_check_only
class TensorOpInference1To3(TensorOpMetadata, Protocol):
    def __call__(
        self, x1: _TensorFamily, /
    ) -> tuple[_TensorFamily, _TensorFamily, _TensorFamily]: ...

@type_check_only
class TensorOpInference1To4(TensorOpMetadata, Protocol):
    def __call__(
        self, x1: _TensorFamily, /
    ) -> tuple[_TensorFamily, _TensorFamily, _TensorFamily, _TensorFamily]: ...

@type_check_only
class TensorOpInference1To5(TensorOpMetadata, Protocol):
    def __call__(
        self, x1: _TensorFamily, /
    ) -> tuple[
        _TensorFamily, _TensorFamily, _TensorFamily, _TensorFamily, _TensorFamily
    ]: ...

@type_check_only
class TensorOpInference1To6(TensorOpMetadata, Protocol):
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

@type_check_only
class TensorOpInference2To1(TensorOpMetadata, Protocol):
    def __call__(self, x1: _TensorFamily, x2: _TensorFamily, /) -> _TensorFamily: ...

@type_check_only
class TensorOpInference2To2(TensorOpMetadata, Protocol):
    def __call__(
        self, x1: _TensorFamily, x2: _TensorFamily, /
    ) -> tuple[_TensorFamily, _TensorFamily]: ...

@type_check_only
class TensorOpInference2To3(TensorOpMetadata, Protocol):
    def __call__(
        self, x1: _TensorFamily, x2: _TensorFamily, /
    ) -> tuple[_TensorFamily, _TensorFamily, _TensorFamily]: ...

@type_check_only
class TensorOpInference2To4(TensorOpMetadata, Protocol):
    def __call__(
        self, x1: _TensorFamily, x2: _TensorFamily, /
    ) -> tuple[_TensorFamily, _TensorFamily, _TensorFamily, _TensorFamily]: ...

@type_check_only
class TensorOpInference2To5(TensorOpMetadata, Protocol):
    def __call__(
        self, x1: _TensorFamily, x2: _TensorFamily, /
    ) -> tuple[
        _TensorFamily, _TensorFamily, _TensorFamily, _TensorFamily, _TensorFamily
    ]: ...

@type_check_only
class TensorOpInference2To6(TensorOpMetadata, Protocol):
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

@type_check_only
class TensorOpInference3To1(TensorOpMetadata, Protocol):
    def __call__(
        self, x1: _TensorFamily, x2: _TensorFamily, x3: _TensorFamily, /
    ) -> _TensorFamily: ...

@type_check_only
class TensorOpInference3To2(TensorOpMetadata, Protocol):
    def __call__(
        self, x1: _TensorFamily, x2: _TensorFamily, x3: _TensorFamily, /
    ) -> tuple[_TensorFamily, _TensorFamily]: ...

@type_check_only
class TensorOpInference3To3(TensorOpMetadata, Protocol):
    def __call__(
        self, x1: _TensorFamily, x2: _TensorFamily, x3: _TensorFamily, /
    ) -> tuple[_TensorFamily, _TensorFamily, _TensorFamily]: ...

@type_check_only
class TensorOpInference3To4(TensorOpMetadata, Protocol):
    def __call__(
        self, x1: _TensorFamily, x2: _TensorFamily, x3: _TensorFamily, /
    ) -> tuple[_TensorFamily, _TensorFamily, _TensorFamily, _TensorFamily]: ...

@type_check_only
class TensorOpInference3To5(TensorOpMetadata, Protocol):
    def __call__(
        self, x1: _TensorFamily, x2: _TensorFamily, x3: _TensorFamily, /
    ) -> tuple[
        _TensorFamily, _TensorFamily, _TensorFamily, _TensorFamily, _TensorFamily
    ]: ...

@type_check_only
class TensorOpInference3To6(TensorOpMetadata, Protocol):
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

@type_check_only
class TensorOpInference4To1(TensorOpMetadata, Protocol):
    def __call__(
        self,
        x1: _TensorFamily,
        x2: _TensorFamily,
        x3: _TensorFamily,
        x4: _TensorFamily,
        /,
    ) -> _TensorFamily: ...

@type_check_only
class TensorOpInference4To2(TensorOpMetadata, Protocol):
    def __call__(
        self,
        x1: _TensorFamily,
        x2: _TensorFamily,
        x3: _TensorFamily,
        x4: _TensorFamily,
        /,
    ) -> tuple[_TensorFamily, _TensorFamily]: ...

@type_check_only
class TensorOpInference4To3(TensorOpMetadata, Protocol):
    def __call__(
        self,
        x1: _TensorFamily,
        x2: _TensorFamily,
        x3: _TensorFamily,
        x4: _TensorFamily,
        /,
    ) -> tuple[_TensorFamily, _TensorFamily, _TensorFamily]: ...

@type_check_only
class TensorOpInference4To4(TensorOpMetadata, Protocol):
    def __call__(
        self,
        x1: _TensorFamily,
        x2: _TensorFamily,
        x3: _TensorFamily,
        x4: _TensorFamily,
        /,
    ) -> tuple[_TensorFamily, _TensorFamily, _TensorFamily, _TensorFamily]: ...

@type_check_only
class TensorOpInference4To5(TensorOpMetadata, Protocol):
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

@type_check_only
class TensorOpInference4To6(TensorOpMetadata, Protocol):
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

@type_check_only
class TensorOpInference5To1(TensorOpMetadata, Protocol):
    def __call__(
        self,
        x1: _TensorFamily,
        x2: _TensorFamily,
        x3: _TensorFamily,
        x4: _TensorFamily,
        x5: _TensorFamily,
        /,
    ) -> _TensorFamily: ...

@type_check_only
class TensorOpInference5To2(TensorOpMetadata, Protocol):
    def __call__(
        self,
        x1: _TensorFamily,
        x2: _TensorFamily,
        x3: _TensorFamily,
        x4: _TensorFamily,
        x5: _TensorFamily,
        /,
    ) -> tuple[_TensorFamily, _TensorFamily]: ...

@type_check_only
class TensorOpInference5To3(TensorOpMetadata, Protocol):
    def __call__(
        self,
        x1: _TensorFamily,
        x2: _TensorFamily,
        x3: _TensorFamily,
        x4: _TensorFamily,
        x5: _TensorFamily,
        /,
    ) -> tuple[_TensorFamily, _TensorFamily, _TensorFamily]: ...

@type_check_only
class TensorOpInference5To4(TensorOpMetadata, Protocol):
    def __call__(
        self,
        x1: _TensorFamily,
        x2: _TensorFamily,
        x3: _TensorFamily,
        x4: _TensorFamily,
        x5: _TensorFamily,
        /,
    ) -> tuple[_TensorFamily, _TensorFamily, _TensorFamily, _TensorFamily]: ...

@type_check_only
class TensorOpInference5To5(TensorOpMetadata, Protocol):
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

@type_check_only
class TensorOpInference5To6(TensorOpMetadata, Protocol):
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

@type_check_only
class TensorOpInference6To1(TensorOpMetadata, Protocol):
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

@type_check_only
class TensorOpInference6To2(TensorOpMetadata, Protocol):
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

@type_check_only
class TensorOpInference6To3(TensorOpMetadata, Protocol):
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

@type_check_only
class TensorOpInference6To4(TensorOpMetadata, Protocol):
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

@type_check_only
class TensorOpInference6To5(TensorOpMetadata, Protocol):
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

@type_check_only
class TensorOpInference6To6(TensorOpMetadata, Protocol):
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

@type_check_only
class TensorOpInference1ToMany(TensorOpMetadata, Protocol):
    def __call__(self, x1: _TensorFamily, /) -> tuple[_TensorFamily, ...]: ...

@type_check_only
class TensorOpInferenceManyTo1(TensorOpMetadata, Protocol):
    def __call__(
        self, x1: _TensorFamily, x2: _TensorFamily, /, *xs: _TensorFamily
    ) -> _TensorFamily: ...

@type_check_only
class TensorOpInferenceManyToMany(TensorOpMetadata, Protocol):
    def __call__(
        self, x1: _TensorFamily, x2: _TensorFamily, /, *xs: _TensorFamily
    ) -> tuple[_TensorFamily, ...]: ...

@type_check_only
class EinopTensorOpInference1To1(
    TensorOpInference1To1, ReducerCustomization, Protocol
): ...

@type_check_only
class EinopTensorOpInference1To2(
    TensorOpInference1To2, ReducerCustomization, Protocol
): ...

@type_check_only
class EinopTensorOpInference1To3(
    TensorOpInference1To3, ReducerCustomization, Protocol
): ...

@type_check_only
class EinopTensorOpInference1To4(
    TensorOpInference1To4, ReducerCustomization, Protocol
): ...

@type_check_only
class EinopTensorOpInference1To5(
    TensorOpInference1To5, ReducerCustomization, Protocol
): ...

@type_check_only
class EinopTensorOpInference1To6(
    TensorOpInference1To6, ReducerCustomization, Protocol
): ...

@type_check_only
class EinopTensorOpInference2To1(
    TensorOpInference2To1, ReducerCustomization, Protocol
): ...

@type_check_only
class EinopTensorOpInference2To2(
    TensorOpInference2To2, ReducerCustomization, Protocol
): ...

@type_check_only
class EinopTensorOpInference2To3(
    TensorOpInference2To3, ReducerCustomization, Protocol
): ...

@type_check_only
class EinopTensorOpInference2To4(
    TensorOpInference2To4, ReducerCustomization, Protocol
): ...

@type_check_only
class EinopTensorOpInference2To5(
    TensorOpInference2To5, ReducerCustomization, Protocol
): ...

@type_check_only
class EinopTensorOpInference2To6(
    TensorOpInference2To6, ReducerCustomization, Protocol
): ...

@type_check_only
class EinopTensorOpInference3To1(
    TensorOpInference3To1, ReducerCustomization, Protocol
): ...

@type_check_only
class EinopTensorOpInference3To2(
    TensorOpInference3To2, ReducerCustomization, Protocol
): ...

@type_check_only
class EinopTensorOpInference3To3(
    TensorOpInference3To3, ReducerCustomization, Protocol
): ...

@type_check_only
class EinopTensorOpInference3To4(
    TensorOpInference3To4, ReducerCustomization, Protocol
): ...

@type_check_only
class EinopTensorOpInference3To5(
    TensorOpInference3To5, ReducerCustomization, Protocol
): ...

@type_check_only
class EinopTensorOpInference3To6(
    TensorOpInference3To6, ReducerCustomization, Protocol
): ...

@type_check_only
class EinopTensorOpInference4To1(
    TensorOpInference4To1, ReducerCustomization, Protocol
): ...

@type_check_only
class EinopTensorOpInference4To2(
    TensorOpInference4To2, ReducerCustomization, Protocol
): ...

@type_check_only
class EinopTensorOpInference4To3(
    TensorOpInference4To3, ReducerCustomization, Protocol
): ...

@type_check_only
class EinopTensorOpInference4To4(
    TensorOpInference4To4, ReducerCustomization, Protocol
): ...

@type_check_only
class EinopTensorOpInference4To5(
    TensorOpInference4To5, ReducerCustomization, Protocol
): ...

@type_check_only
class EinopTensorOpInference4To6(
    TensorOpInference4To6, ReducerCustomization, Protocol
): ...

@type_check_only
class EinopTensorOpInference5To1(
    TensorOpInference5To1, ReducerCustomization, Protocol
): ...

@type_check_only
class EinopTensorOpInference5To2(
    TensorOpInference5To2, ReducerCustomization, Protocol
): ...

@type_check_only
class EinopTensorOpInference5To3(
    TensorOpInference5To3, ReducerCustomization, Protocol
): ...

@type_check_only
class EinopTensorOpInference5To4(
    TensorOpInference5To4, ReducerCustomization, Protocol
): ...

@type_check_only
class EinopTensorOpInference5To5(
    TensorOpInference5To5, ReducerCustomization, Protocol
): ...

@type_check_only
class EinopTensorOpInference5To6(
    TensorOpInference5To6, ReducerCustomization, Protocol
): ...

@type_check_only
class EinopTensorOpInference6To1(
    TensorOpInference6To1, ReducerCustomization, Protocol
): ...

@type_check_only
class EinopTensorOpInference6To2(
    TensorOpInference6To2, ReducerCustomization, Protocol
): ...

@type_check_only
class EinopTensorOpInference6To3(
    TensorOpInference6To3, ReducerCustomization, Protocol
): ...

@type_check_only
class EinopTensorOpInference6To4(
    TensorOpInference6To4, ReducerCustomization, Protocol
): ...

@type_check_only
class EinopTensorOpInference6To5(
    TensorOpInference6To5, ReducerCustomization, Protocol
): ...

@type_check_only
class EinopTensorOpInference6To6(
    TensorOpInference6To6, ReducerCustomization, Protocol
): ...

@type_check_only
class EinopTensorOpInference1ToMany(
    TensorOpInference1ToMany, ReducerCustomization, Protocol
): ...

@type_check_only
class EinopTensorOpInferenceManyTo1(
    TensorOpInferenceManyTo1, ReducerCustomization, Protocol
): ...

@type_check_only
class EinopTensorOpInferenceManyToMany(
    TensorOpInferenceManyToMany, ReducerCustomization, Protocol
): ...

@overload
def view(lhs: AxisTerms, rhs: AxisTerms) -> TensorOpInference1To1: ...
@overload
def view(lhs: AxisTerms, rhs: tuple[AxisTerms, AxisTerms]) -> TensorOpInference1To2: ...
@overload
def view(
    lhs: AxisTerms, rhs: tuple[AxisTerms, AxisTerms, AxisTerms]
) -> TensorOpInference1To3: ...
@overload
def view(
    lhs: AxisTerms, rhs: tuple[AxisTerms, AxisTerms, AxisTerms, AxisTerms]
) -> TensorOpInference1To4: ...
@overload
def view(
    lhs: AxisTerms, rhs: tuple[AxisTerms, AxisTerms, AxisTerms, AxisTerms, AxisTerms]
) -> TensorOpInference1To5: ...
@overload
def view(
    lhs: AxisTerms,
    rhs: tuple[AxisTerms, AxisTerms, AxisTerms, AxisTerms, AxisTerms, AxisTerms],
) -> TensorOpInference1To6: ...
@overload
def view(
    lhs: AxisTerms,
    rhs: tuple[
        AxisTerms,
        AxisTerms,
        AxisTerms,
        AxisTerms,
        AxisTerms,
        AxisTerms,
        AxisTerms,
        Unpack[tuple[AxisTerms, ...]],
    ],
) -> TensorOpInference1ToMany: ...
@overload
def rearrange(lhs: AxisTerms, rhs: AxisTerms) -> TensorOpInference1To1: ...
@overload
def rearrange(
    lhs: AxisTerms, rhs: tuple[AxisTerms, AxisTerms]
) -> TensorOpInference1To2: ...
@overload
def rearrange(
    lhs: AxisTerms, rhs: tuple[AxisTerms, AxisTerms, AxisTerms]
) -> TensorOpInference1To3: ...
@overload
def rearrange(
    lhs: AxisTerms, rhs: tuple[AxisTerms, AxisTerms, AxisTerms, AxisTerms]
) -> TensorOpInference1To4: ...
@overload
def rearrange(
    lhs: AxisTerms, rhs: tuple[AxisTerms, AxisTerms, AxisTerms, AxisTerms, AxisTerms]
) -> TensorOpInference1To5: ...
@overload
def rearrange(
    lhs: AxisTerms,
    rhs: tuple[AxisTerms, AxisTerms, AxisTerms, AxisTerms, AxisTerms, AxisTerms],
) -> TensorOpInference1To6: ...
@overload
def rearrange(
    lhs: tuple[AxisTerms, AxisTerms], rhs: AxisTerms
) -> TensorOpInference2To1: ...
@overload
def rearrange(
    lhs: tuple[AxisTerms, AxisTerms], rhs: tuple[AxisTerms, AxisTerms]
) -> TensorOpInference2To2: ...
@overload
def rearrange(
    lhs: tuple[AxisTerms, AxisTerms], rhs: tuple[AxisTerms, AxisTerms, AxisTerms]
) -> TensorOpInference2To3: ...
@overload
def rearrange(
    lhs: tuple[AxisTerms, AxisTerms],
    rhs: tuple[AxisTerms, AxisTerms, AxisTerms, AxisTerms],
) -> TensorOpInference2To4: ...
@overload
def rearrange(
    lhs: tuple[AxisTerms, AxisTerms],
    rhs: tuple[AxisTerms, AxisTerms, AxisTerms, AxisTerms, AxisTerms],
) -> TensorOpInference2To5: ...
@overload
def rearrange(
    lhs: tuple[AxisTerms, AxisTerms],
    rhs: tuple[AxisTerms, AxisTerms, AxisTerms, AxisTerms, AxisTerms, AxisTerms],
) -> TensorOpInference2To6: ...
@overload
def rearrange(
    lhs: tuple[AxisTerms, AxisTerms, AxisTerms], rhs: AxisTerms
) -> TensorOpInference3To1: ...
@overload
def rearrange(
    lhs: tuple[AxisTerms, AxisTerms, AxisTerms], rhs: tuple[AxisTerms, AxisTerms]
) -> TensorOpInference3To2: ...
@overload
def rearrange(
    lhs: tuple[AxisTerms, AxisTerms, AxisTerms],
    rhs: tuple[AxisTerms, AxisTerms, AxisTerms],
) -> TensorOpInference3To3: ...
@overload
def rearrange(
    lhs: tuple[AxisTerms, AxisTerms, AxisTerms],
    rhs: tuple[AxisTerms, AxisTerms, AxisTerms, AxisTerms],
) -> TensorOpInference3To4: ...
@overload
def rearrange(
    lhs: tuple[AxisTerms, AxisTerms, AxisTerms],
    rhs: tuple[AxisTerms, AxisTerms, AxisTerms, AxisTerms, AxisTerms],
) -> TensorOpInference3To5: ...
@overload
def rearrange(
    lhs: tuple[AxisTerms, AxisTerms, AxisTerms],
    rhs: tuple[AxisTerms, AxisTerms, AxisTerms, AxisTerms, AxisTerms, AxisTerms],
) -> TensorOpInference3To6: ...
@overload
def rearrange(
    lhs: tuple[AxisTerms, AxisTerms, AxisTerms, AxisTerms], rhs: AxisTerms
) -> TensorOpInference4To1: ...
@overload
def rearrange(
    lhs: tuple[AxisTerms, AxisTerms, AxisTerms, AxisTerms],
    rhs: tuple[AxisTerms, AxisTerms],
) -> TensorOpInference4To2: ...
@overload
def rearrange(
    lhs: tuple[AxisTerms, AxisTerms, AxisTerms, AxisTerms],
    rhs: tuple[AxisTerms, AxisTerms, AxisTerms],
) -> TensorOpInference4To3: ...
@overload
def rearrange(
    lhs: tuple[AxisTerms, AxisTerms, AxisTerms, AxisTerms],
    rhs: tuple[AxisTerms, AxisTerms, AxisTerms, AxisTerms],
) -> TensorOpInference4To4: ...
@overload
def rearrange(
    lhs: tuple[AxisTerms, AxisTerms, AxisTerms, AxisTerms],
    rhs: tuple[AxisTerms, AxisTerms, AxisTerms, AxisTerms, AxisTerms],
) -> TensorOpInference4To5: ...
@overload
def rearrange(
    lhs: tuple[AxisTerms, AxisTerms, AxisTerms, AxisTerms],
    rhs: tuple[AxisTerms, AxisTerms, AxisTerms, AxisTerms, AxisTerms, AxisTerms],
) -> TensorOpInference4To6: ...
@overload
def rearrange(
    lhs: tuple[AxisTerms, AxisTerms, AxisTerms, AxisTerms, AxisTerms], rhs: AxisTerms
) -> TensorOpInference5To1: ...
@overload
def rearrange(
    lhs: tuple[AxisTerms, AxisTerms, AxisTerms, AxisTerms, AxisTerms],
    rhs: tuple[AxisTerms, AxisTerms],
) -> TensorOpInference5To2: ...
@overload
def rearrange(
    lhs: tuple[AxisTerms, AxisTerms, AxisTerms, AxisTerms, AxisTerms],
    rhs: tuple[AxisTerms, AxisTerms, AxisTerms],
) -> TensorOpInference5To3: ...
@overload
def rearrange(
    lhs: tuple[AxisTerms, AxisTerms, AxisTerms, AxisTerms, AxisTerms],
    rhs: tuple[AxisTerms, AxisTerms, AxisTerms, AxisTerms],
) -> TensorOpInference5To4: ...
@overload
def rearrange(
    lhs: tuple[AxisTerms, AxisTerms, AxisTerms, AxisTerms, AxisTerms],
    rhs: tuple[AxisTerms, AxisTerms, AxisTerms, AxisTerms, AxisTerms],
) -> TensorOpInference5To5: ...
@overload
def rearrange(
    lhs: tuple[AxisTerms, AxisTerms, AxisTerms, AxisTerms, AxisTerms],
    rhs: tuple[AxisTerms, AxisTerms, AxisTerms, AxisTerms, AxisTerms, AxisTerms],
) -> TensorOpInference5To6: ...
@overload
def rearrange(
    lhs: tuple[AxisTerms, AxisTerms, AxisTerms, AxisTerms, AxisTerms, AxisTerms],
    rhs: AxisTerms,
) -> TensorOpInference6To1: ...
@overload
def rearrange(
    lhs: tuple[AxisTerms, AxisTerms, AxisTerms, AxisTerms, AxisTerms, AxisTerms],
    rhs: tuple[AxisTerms, AxisTerms],
) -> TensorOpInference6To2: ...
@overload
def rearrange(
    lhs: tuple[AxisTerms, AxisTerms, AxisTerms, AxisTerms, AxisTerms, AxisTerms],
    rhs: tuple[AxisTerms, AxisTerms, AxisTerms],
) -> TensorOpInference6To3: ...
@overload
def rearrange(
    lhs: tuple[AxisTerms, AxisTerms, AxisTerms, AxisTerms, AxisTerms, AxisTerms],
    rhs: tuple[AxisTerms, AxisTerms, AxisTerms, AxisTerms],
) -> TensorOpInference6To4: ...
@overload
def rearrange(
    lhs: tuple[AxisTerms, AxisTerms, AxisTerms, AxisTerms, AxisTerms, AxisTerms],
    rhs: tuple[AxisTerms, AxisTerms, AxisTerms, AxisTerms, AxisTerms],
) -> TensorOpInference6To5: ...
@overload
def rearrange(
    lhs: tuple[AxisTerms, AxisTerms, AxisTerms, AxisTerms, AxisTerms, AxisTerms],
    rhs: tuple[AxisTerms, AxisTerms, AxisTerms, AxisTerms, AxisTerms, AxisTerms],
) -> TensorOpInference6To6: ...
@overload
def rearrange(
    lhs: AxisTerms,
    rhs: tuple[
        AxisTerms,
        AxisTerms,
        AxisTerms,
        AxisTerms,
        AxisTerms,
        AxisTerms,
        AxisTerms,
        Unpack[tuple[AxisTerms, ...]],
    ],
) -> TensorOpInference1ToMany: ...
@overload
def rearrange(
    lhs: tuple[
        AxisTerms,
        AxisTerms,
        AxisTerms,
        AxisTerms,
        AxisTerms,
        AxisTerms,
        AxisTerms,
        Unpack[tuple[AxisTerms, ...]],
    ],
    rhs: AxisTerms,
) -> TensorOpInferenceManyTo1: ...
@overload
def rearrange(
    lhs: tuple[
        AxisTerms,
        AxisTerms,
        AxisTerms,
        AxisTerms,
        AxisTerms,
        AxisTerms,
        AxisTerms,
        Unpack[tuple[AxisTerms, ...]],
    ],
    rhs: tuple[
        AxisTerms,
        AxisTerms,
        AxisTerms,
        AxisTerms,
        AxisTerms,
        AxisTerms,
        AxisTerms,
        Unpack[tuple[AxisTerms, ...]],
    ],
) -> TensorOpInferenceManyToMany: ...
def repeat(lhs: AxisTerms, rhs: AxisTerms) -> TensorOpInference1To1: ...
def reduce(lhs: AxisTerms, rhs: AxisTerms) -> ReduceTensorOpInference1To1: ...
@overload
def contract(lhs: AxisTerms, rhs: AxisTerms) -> TensorOpInference1To1: ...
@overload
def contract(
    lhs: tuple[AxisTerms, AxisTerms], rhs: AxisTerms
) -> TensorOpInference2To1: ...
@overload
def contract(
    lhs: tuple[AxisTerms, AxisTerms, AxisTerms], rhs: AxisTerms
) -> TensorOpInference3To1: ...
@overload
def contract(
    lhs: tuple[AxisTerms, AxisTerms, AxisTerms, AxisTerms], rhs: AxisTerms
) -> TensorOpInference4To1: ...
@overload
def contract(
    lhs: tuple[AxisTerms, AxisTerms, AxisTerms, AxisTerms, AxisTerms], rhs: AxisTerms
) -> TensorOpInference5To1: ...
@overload
def contract(
    lhs: tuple[AxisTerms, AxisTerms, AxisTerms, AxisTerms, AxisTerms, AxisTerms],
    rhs: AxisTerms,
) -> TensorOpInference6To1: ...
@overload
def contract(
    lhs: tuple[
        AxisTerms,
        AxisTerms,
        AxisTerms,
        AxisTerms,
        AxisTerms,
        AxisTerms,
        AxisTerms,
        Unpack[tuple[AxisTerms, ...]],
    ],
    rhs: AxisTerms,
) -> TensorOpInferenceManyTo1: ...
@overload
def einop(lhs: AxisTerms, rhs: AxisTerms) -> EinopTensorOpInference1To1: ...
@overload
def einop(
    lhs: AxisTerms, rhs: tuple[AxisTerms, AxisTerms]
) -> EinopTensorOpInference1To2: ...
@overload
def einop(
    lhs: AxisTerms, rhs: tuple[AxisTerms, AxisTerms, AxisTerms]
) -> EinopTensorOpInference1To3: ...
@overload
def einop(
    lhs: AxisTerms, rhs: tuple[AxisTerms, AxisTerms, AxisTerms, AxisTerms]
) -> EinopTensorOpInference1To4: ...
@overload
def einop(
    lhs: AxisTerms, rhs: tuple[AxisTerms, AxisTerms, AxisTerms, AxisTerms, AxisTerms]
) -> EinopTensorOpInference1To5: ...
@overload
def einop(
    lhs: AxisTerms,
    rhs: tuple[AxisTerms, AxisTerms, AxisTerms, AxisTerms, AxisTerms, AxisTerms],
) -> EinopTensorOpInference1To6: ...
@overload
def einop(
    lhs: tuple[AxisTerms, AxisTerms], rhs: AxisTerms
) -> EinopTensorOpInference2To1: ...
@overload
def einop(
    lhs: tuple[AxisTerms, AxisTerms], rhs: tuple[AxisTerms, AxisTerms]
) -> EinopTensorOpInference2To2: ...
@overload
def einop(
    lhs: tuple[AxisTerms, AxisTerms], rhs: tuple[AxisTerms, AxisTerms, AxisTerms]
) -> EinopTensorOpInference2To3: ...
@overload
def einop(
    lhs: tuple[AxisTerms, AxisTerms],
    rhs: tuple[AxisTerms, AxisTerms, AxisTerms, AxisTerms],
) -> EinopTensorOpInference2To4: ...
@overload
def einop(
    lhs: tuple[AxisTerms, AxisTerms],
    rhs: tuple[AxisTerms, AxisTerms, AxisTerms, AxisTerms, AxisTerms],
) -> EinopTensorOpInference2To5: ...
@overload
def einop(
    lhs: tuple[AxisTerms, AxisTerms],
    rhs: tuple[AxisTerms, AxisTerms, AxisTerms, AxisTerms, AxisTerms, AxisTerms],
) -> EinopTensorOpInference2To6: ...
@overload
def einop(
    lhs: tuple[AxisTerms, AxisTerms, AxisTerms], rhs: AxisTerms
) -> EinopTensorOpInference3To1: ...
@overload
def einop(
    lhs: tuple[AxisTerms, AxisTerms, AxisTerms], rhs: tuple[AxisTerms, AxisTerms]
) -> EinopTensorOpInference3To2: ...
@overload
def einop(
    lhs: tuple[AxisTerms, AxisTerms, AxisTerms],
    rhs: tuple[AxisTerms, AxisTerms, AxisTerms],
) -> EinopTensorOpInference3To3: ...
@overload
def einop(
    lhs: tuple[AxisTerms, AxisTerms, AxisTerms],
    rhs: tuple[AxisTerms, AxisTerms, AxisTerms, AxisTerms],
) -> EinopTensorOpInference3To4: ...
@overload
def einop(
    lhs: tuple[AxisTerms, AxisTerms, AxisTerms],
    rhs: tuple[AxisTerms, AxisTerms, AxisTerms, AxisTerms, AxisTerms],
) -> EinopTensorOpInference3To5: ...
@overload
def einop(
    lhs: tuple[AxisTerms, AxisTerms, AxisTerms],
    rhs: tuple[AxisTerms, AxisTerms, AxisTerms, AxisTerms, AxisTerms, AxisTerms],
) -> EinopTensorOpInference3To6: ...
@overload
def einop(
    lhs: tuple[AxisTerms, AxisTerms, AxisTerms, AxisTerms], rhs: AxisTerms
) -> EinopTensorOpInference4To1: ...
@overload
def einop(
    lhs: tuple[AxisTerms, AxisTerms, AxisTerms, AxisTerms],
    rhs: tuple[AxisTerms, AxisTerms],
) -> EinopTensorOpInference4To2: ...
@overload
def einop(
    lhs: tuple[AxisTerms, AxisTerms, AxisTerms, AxisTerms],
    rhs: tuple[AxisTerms, AxisTerms, AxisTerms],
) -> EinopTensorOpInference4To3: ...
@overload
def einop(
    lhs: tuple[AxisTerms, AxisTerms, AxisTerms, AxisTerms],
    rhs: tuple[AxisTerms, AxisTerms, AxisTerms, AxisTerms],
) -> EinopTensorOpInference4To4: ...
@overload
def einop(
    lhs: tuple[AxisTerms, AxisTerms, AxisTerms, AxisTerms],
    rhs: tuple[AxisTerms, AxisTerms, AxisTerms, AxisTerms, AxisTerms],
) -> EinopTensorOpInference4To5: ...
@overload
def einop(
    lhs: tuple[AxisTerms, AxisTerms, AxisTerms, AxisTerms],
    rhs: tuple[AxisTerms, AxisTerms, AxisTerms, AxisTerms, AxisTerms, AxisTerms],
) -> EinopTensorOpInference4To6: ...
@overload
def einop(
    lhs: tuple[AxisTerms, AxisTerms, AxisTerms, AxisTerms, AxisTerms], rhs: AxisTerms
) -> EinopTensorOpInference5To1: ...
@overload
def einop(
    lhs: tuple[AxisTerms, AxisTerms, AxisTerms, AxisTerms, AxisTerms],
    rhs: tuple[AxisTerms, AxisTerms],
) -> EinopTensorOpInference5To2: ...
@overload
def einop(
    lhs: tuple[AxisTerms, AxisTerms, AxisTerms, AxisTerms, AxisTerms],
    rhs: tuple[AxisTerms, AxisTerms, AxisTerms],
) -> EinopTensorOpInference5To3: ...
@overload
def einop(
    lhs: tuple[AxisTerms, AxisTerms, AxisTerms, AxisTerms, AxisTerms],
    rhs: tuple[AxisTerms, AxisTerms, AxisTerms, AxisTerms],
) -> EinopTensorOpInference5To4: ...
@overload
def einop(
    lhs: tuple[AxisTerms, AxisTerms, AxisTerms, AxisTerms, AxisTerms],
    rhs: tuple[AxisTerms, AxisTerms, AxisTerms, AxisTerms, AxisTerms],
) -> EinopTensorOpInference5To5: ...
@overload
def einop(
    lhs: tuple[AxisTerms, AxisTerms, AxisTerms, AxisTerms, AxisTerms],
    rhs: tuple[AxisTerms, AxisTerms, AxisTerms, AxisTerms, AxisTerms, AxisTerms],
) -> EinopTensorOpInference5To6: ...
@overload
def einop(
    lhs: tuple[AxisTerms, AxisTerms, AxisTerms, AxisTerms, AxisTerms, AxisTerms],
    rhs: AxisTerms,
) -> EinopTensorOpInference6To1: ...
@overload
def einop(
    lhs: tuple[AxisTerms, AxisTerms, AxisTerms, AxisTerms, AxisTerms, AxisTerms],
    rhs: tuple[AxisTerms, AxisTerms],
) -> EinopTensorOpInference6To2: ...
@overload
def einop(
    lhs: tuple[AxisTerms, AxisTerms, AxisTerms, AxisTerms, AxisTerms, AxisTerms],
    rhs: tuple[AxisTerms, AxisTerms, AxisTerms],
) -> EinopTensorOpInference6To3: ...
@overload
def einop(
    lhs: tuple[AxisTerms, AxisTerms, AxisTerms, AxisTerms, AxisTerms, AxisTerms],
    rhs: tuple[AxisTerms, AxisTerms, AxisTerms, AxisTerms],
) -> EinopTensorOpInference6To4: ...
@overload
def einop(
    lhs: tuple[AxisTerms, AxisTerms, AxisTerms, AxisTerms, AxisTerms, AxisTerms],
    rhs: tuple[AxisTerms, AxisTerms, AxisTerms, AxisTerms, AxisTerms],
) -> EinopTensorOpInference6To5: ...
@overload
def einop(
    lhs: tuple[AxisTerms, AxisTerms, AxisTerms, AxisTerms, AxisTerms, AxisTerms],
    rhs: tuple[AxisTerms, AxisTerms, AxisTerms, AxisTerms, AxisTerms, AxisTerms],
) -> EinopTensorOpInference6To6: ...
@overload
def einop(
    lhs: AxisTerms,
    rhs: tuple[
        AxisTerms,
        AxisTerms,
        AxisTerms,
        AxisTerms,
        AxisTerms,
        AxisTerms,
        AxisTerms,
        Unpack[tuple[AxisTerms, ...]],
    ],
) -> EinopTensorOpInference1ToMany: ...
@overload
def einop(
    lhs: tuple[
        AxisTerms,
        AxisTerms,
        AxisTerms,
        AxisTerms,
        AxisTerms,
        AxisTerms,
        AxisTerms,
        Unpack[tuple[AxisTerms, ...]],
    ],
    rhs: AxisTerms,
) -> EinopTensorOpInferenceManyTo1: ...
@overload
def einop(
    lhs: tuple[
        AxisTerms,
        AxisTerms,
        AxisTerms,
        AxisTerms,
        AxisTerms,
        AxisTerms,
        AxisTerms,
        Unpack[tuple[AxisTerms, ...]],
    ],
    rhs: tuple[
        AxisTerms,
        AxisTerms,
        AxisTerms,
        AxisTerms,
        AxisTerms,
        AxisTerms,
        AxisTerms,
        Unpack[tuple[AxisTerms, ...]],
    ],
) -> EinopTensorOpInferenceManyToMany: ...

__all__ = [
    "ReducibleTensorOp",
    "TensorOp",
    "contract",
    "einop",
    "rearrange",
    "reduce",
    "repeat",
    "view",
]
