from dataclasses import dataclass, field

from ..axis import AxisSide, AxisTerms
from ..diagnostics import ErrorCode, ValidationError
from ..einop_layout import EinopLayoutNormalization
from ..reduction.plan import ReducerPlanParser
from ..reduction.schema import Reducer, ReducerPlan
from ..signature import Signature
from .kind import OperationKind
from .policy import OpPolicy, resolve_op_policy
from .validation import validate_contract_atomic_terms


def _normalize_sizes_items(
    *,
    op_name: str,
    sizes_items: tuple[tuple[str, int], ...],
    axis_names: set[str],
) -> tuple[tuple[str, int], ...]:
    """Validate and normalize size bindings to one immutable sorted tuple."""
    merged: dict[str, int] = {}
    for key, value in sizes_items:
        if key not in axis_names:
            raise ValidationError(
                code=ErrorCode.INCONSISTENT_DIMS,
                message=(
                    f"inconsistent dims: with_sizes binding {key!r} "
                    "does not name a scalar axis in the signature"
                ),
                help="bind only scalar axes referenced by the operation signature",
                related=("with_sizes binding",),
                data={"operation": op_name, "dim": key},
            )
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"size binding for {key!r} must be an int")
        if value < 0:
            raise ValidationError(
                code=ErrorCode.INCONSISTENT_DIMS,
                message=f"inconsistent dims: negative with_sizes binding for {key!r}",
                help="provide non-negative with_sizes bindings",
                related=("with_sizes binding",),
                data={"operation": op_name, "dim": key, "value": value},
            )
        merged[key] = value
    return tuple(sorted(merged.items()))


@dataclass(frozen=True, slots=True)
class TensorOpDefinition:
    """Canonical operation semantics independent of runtime planning."""

    kind: OperationKind
    lhs: AxisSide
    rhs: AxisSide
    reducer_plan: ReducerPlan | None = None
    sizes_items: tuple[tuple[str, int], ...] = ()
    signature: Signature = field(init=False, repr=False)
    input_arity: int = field(init=False, repr=False)
    output_arity: int = field(init=False, repr=False)
    op_policy: OpPolicy = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Normalize inputs and enforce operation-semantic invariants."""
        normalized = Signature(inputs=self.lhs, outputs=self.rhs)
        op_policy = resolve_op_policy(self.kind)
        if self.reducer_plan is not None and not op_policy.supports_reducer:
            raise ValueError(f"{self.name} does not support reducer plans")
        op_policy.validate_constructor(
            op_name=self.name,
            lhs=normalized.inputs,
            rhs=normalized.outputs,
        )
        if self.kind is OperationKind.CONTRACT:
            validate_contract_atomic_terms(normalized)
        normalized_sizes_items = _normalize_sizes_items(
            op_name=self.name,
            sizes_items=self.sizes_items,
            axis_names=normalized.axis_names(),
        )
        object.__setattr__(self, "lhs", normalized.inputs)
        object.__setattr__(self, "rhs", normalized.outputs)
        object.__setattr__(self, "sizes_items", normalized_sizes_items)
        object.__setattr__(self, "signature", normalized)
        object.__setattr__(self, "input_arity", len(normalized.inputs))
        object.__setattr__(self, "output_arity", len(normalized.outputs))
        object.__setattr__(self, "op_policy", op_policy)

    @property
    def name(self) -> str:
        """Return the operation name used by execution and diagnostics."""
        return self.kind.value

    @property
    def supports_reducer(self) -> bool:
        """Return whether this operation accepts reducer customization."""
        return self.op_policy.supports_reducer

    def with_sizes(self, **sizes: int) -> "TensorOpDefinition":
        """Return a definition with merged explicit size bindings."""
        if not sizes:
            return self

        merged = dict(self.sizes_items)
        for key, value in sizes.items():
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"size binding for {key!r} must be an int")
            if value < 0:
                raise ValidationError(
                    code=ErrorCode.INCONSISTENT_DIMS,
                    message=(
                        f"inconsistent dims: negative with_sizes binding for {key!r}"
                    ),
                    help="provide non-negative with_sizes bindings",
                    related=("with_sizes binding",),
                    data={"operation": self.name, "dim": key, "value": value},
                )
            merged[key] = value

        normalized_sizes_items = tuple(sorted(merged.items()))
        if normalized_sizes_items == self.sizes_items:
            return self
        return TensorOpDefinition(
            kind=self.kind,
            lhs=self.lhs,
            rhs=self.rhs,
            reducer_plan=self.reducer_plan,
            sizes_items=normalized_sizes_items,
        )

    def reduce_by(
        self,
        reducer: Reducer | tuple[AxisTerms, Reducer],
        *phases: tuple[AxisTerms, Reducer],
    ) -> "TensorOpDefinition":
        """Return a definition with one canonical reducer plan."""
        if not self.supports_reducer:
            raise AttributeError(
                f"{self.name} does not support .reduce_by(...) in v0.1"
            )
        if isinstance(reducer, dict):
            raise TypeError(
                "dict reducer plans are not supported; "
                "use ordered phase tuples like "
                "reduce_by((ax[h], 'sum'), (ax[d], 'prod'))"
            )

        reducer_signature = self.signature
        if self.kind is OperationKind.EINOP:
            reducer_signature = EinopLayoutNormalization.from_signature(
                reducer_signature
            ).logical
        reducer_parser = ReducerPlanParser(
            lhs=reducer_signature.inputs,
            rhs=reducer_signature.outputs,
        )
        if self.kind is OperationKind.EINOP and not reducer_parser.reduced_terms():
            raise ValidationError(
                code=ErrorCode.INCONSISTENT_DIMS,
                message="inconsistent dims: reduce_by has no logical axes to reduce",
                help="remove reduce_by from einop signatures that preserve every axis",
                related=("einop reducer configuration",),
                data={"operation": "einop"},
            )
        reducer_plan = reducer_parser.parse(reducer=reducer, phases=phases)
        if reducer_plan == self.reducer_plan:
            return self
        return TensorOpDefinition(
            kind=self.kind,
            lhs=self.lhs,
            rhs=self.rhs,
            reducer_plan=reducer_plan,
            sizes_items=self.sizes_items,
        )

    def sizes(self) -> dict[str, int]:
        """Return explicit size bindings as one detached mapping."""
        return dict(self.sizes_items)


__all__ = ["TensorOpDefinition"]
