from einf.axis import AxisTerms, ScalarAxisTerms, expand_products_for_terms
from einf.diagnostics import ErrorCode, ValidationError
from einf.ir import IRProgram
from einf.plans.symbolic import SymbolicPlan
from einf.reduction.schema import ReducerPlan
from einf.steps.axis_slice import (
    AxisSliceSymbolicStep,
    build_axis_slice_symbolic_program,
    resolve_axis_slice_axis,
)
from einf.steps.base import SymbolicStep
from einf.steps.concat import (
    ConcatSymbolicStep,
    build_concat_symbolic_program,
    resolve_concat_axis,
)
from einf.steps.permute import (
    AxisPermuteSymbolicStep,
    PermuteSymbolicStep,
    build_axis_permute_symbolic_program,
    build_permute_symbolic_program,
)
from einf.steps.reshape import (
    ReshapeSymbolicStep,
    build_reshape_compiled_program,
    build_reshape_symbolic_program,
)


def _resolve_expanded_permutation(
    *,
    expanded_lhs_terms: AxisTerms,
    expanded_rhs_terms: AxisTerms,
) -> tuple[int, ...] | None:
    """Resolve rhs->lhs bijection over expanded unary terms."""
    lhs_term_to_index: dict[str, int] = {}
    for lhs_index, lhs_term in enumerate(expanded_lhs_terms):
        lhs_token = lhs_term.stable_token()
        if lhs_token in lhs_term_to_index:
            return None
        lhs_term_to_index[lhs_token] = lhs_index

    used_lhs_indices: set[int] = set()
    permutation: list[int] = []
    for rhs_term in expanded_rhs_terms:
        rhs_token = rhs_term.stable_token()
        lhs_index = lhs_term_to_index.get(rhs_token)
        if lhs_index is None:
            return None
        if lhs_index in used_lhs_indices:
            return None
        used_lhs_indices.add(lhs_index)
        permutation.append(lhs_index)

    if len(used_lhs_indices) != len(expanded_lhs_terms):
        return None
    return tuple(permutation)


def _build_unary_rearrange_plan(
    *,
    lhs_terms: AxisTerms,
    rhs_terms: AxisTerms,
    explicit_sizes_items: tuple[tuple[str, int], ...],
) -> SymbolicPlan | None:
    """Build unary rearrange symbolic plan from primitive chain."""
    compiled = build_reshape_compiled_program(lhs_terms, rhs_terms)
    if compiled is None:
        return None

    expanded_lhs_terms = AxisTerms(
        tuple(expand_products_for_terms(ScalarAxisTerms.from_spec(lhs_terms)))
    )
    expanded_rhs_terms = AxisTerms(
        tuple(expand_products_for_terms(ScalarAxisTerms.from_spec(rhs_terms)))
    )
    permutation = _resolve_expanded_permutation(
        expanded_lhs_terms=expanded_lhs_terms,
        expanded_rhs_terms=expanded_rhs_terms,
    )

    if expanded_lhs_terms == lhs_terms and expanded_rhs_terms == rhs_terms:
        if permutation is None:
            if len(lhs_terms) == len(rhs_terms):
                return None
            if lhs_terms == rhs_terms:
                return SymbolicPlan(
                    kind="route",
                    input_arity=1,
                    output_arity=1,
                    steps=(),
                )
            reshape_step = ReshapeSymbolicStep(
                program=build_reshape_symbolic_program(lhs_terms, rhs_terms),
                explicit_sizes_items=explicit_sizes_items,
            )
            return SymbolicPlan(
                kind="reshape",
                input_arity=1,
                output_arity=1,
                steps=(reshape_step,),
            )
        identity = tuple(range(len(permutation)))
        if permutation == identity:
            return SymbolicPlan(
                kind="route",
                input_arity=1,
                output_arity=1,
                steps=(),
            )
        permute_step = PermuteSymbolicStep(
            program=build_permute_symbolic_program(permutation)
        )
        return SymbolicPlan(
            kind="permute",
            input_arity=1,
            output_arity=1,
            steps=(permute_step,),
        )

    if permutation is not None:
        identity = tuple(range(len(permutation)))
        if permutation != identity:
            steps: list[SymbolicStep] = []
            if expanded_lhs_terms != lhs_terms:
                steps.append(
                    ReshapeSymbolicStep(
                        program=build_reshape_symbolic_program(
                            lhs_terms, expanded_lhs_terms
                        ),
                        explicit_sizes_items=explicit_sizes_items,
                    )
                )
            steps.append(
                PermuteSymbolicStep(program=build_permute_symbolic_program(permutation))
            )
            if expanded_rhs_terms != rhs_terms:
                steps.append(
                    ReshapeSymbolicStep(
                        program=build_reshape_symbolic_program(
                            expanded_rhs_terms, rhs_terms
                        ),
                        explicit_sizes_items=explicit_sizes_items,
                    )
                )
            return SymbolicPlan(
                kind="rearrange",
                input_arity=1,
                output_arity=1,
                steps=tuple(steps),
            )

    if lhs_terms == rhs_terms:
        return SymbolicPlan(
            kind="route",
            input_arity=1,
            output_arity=1,
            steps=(),
        )

    reshape_step = ReshapeSymbolicStep(
        program=build_reshape_symbolic_program(lhs_terms, rhs_terms),
        explicit_sizes_items=explicit_sizes_items,
    )
    return SymbolicPlan(
        kind="reshape",
        input_arity=1,
        output_arity=1,
        steps=(reshape_step,),
    )


def build_rearrange_symbolic_plan(
    ir_program: IRProgram,
    explicit_sizes_items: tuple[tuple[str, int], ...],
    reducer_plan: ReducerPlan | None,
) -> SymbolicPlan:
    """Build one symbolic plan for `rearrange` from canonical sides."""
    lhs = ir_program.lhs
    rhs = ir_program.rhs
    _ = reducer_plan
    if len(lhs) == 1 and len(rhs) == 1:
        unary_plan = _build_unary_rearrange_plan(
            lhs_terms=lhs[0],
            rhs_terms=rhs[0],
            explicit_sizes_items=explicit_sizes_items,
        )
        if unary_plan is not None:
            return unary_plan

    if resolve_axis_slice_axis(lhs, rhs) is not None:
        axis_slice_step = AxisSliceSymbolicStep(
            lhs=lhs,
            rhs=rhs,
            program=build_axis_slice_symbolic_program(lhs, rhs),
            explicit_sizes_items=explicit_sizes_items,
        )
        return SymbolicPlan(
            kind="axis_slice",
            input_arity=len(lhs),
            output_arity=len(rhs),
            steps=(axis_slice_step,),
        )

    if resolve_concat_axis(lhs, rhs) is not None:
        concat_step = ConcatSymbolicStep(
            lhs=lhs,
            rhs=rhs,
            program=build_concat_symbolic_program(lhs, rhs),
            explicit_sizes_items=explicit_sizes_items,
        )
        return SymbolicPlan(
            kind="concat",
            input_arity=len(lhs),
            output_arity=len(rhs),
            steps=(concat_step,),
        )

    if len(lhs) == 1 and len(rhs) == 1 and len(lhs[0]) == len(rhs[0]):
        axis_permute_step = AxisPermuteSymbolicStep(
            lhs=lhs,
            rhs=rhs,
            program=build_axis_permute_symbolic_program(lhs, rhs, explicit_sizes_items),
            explicit_sizes_items=explicit_sizes_items,
        )
        return SymbolicPlan(
            kind="permute",
            input_arity=len(lhs),
            output_arity=len(rhs),
            steps=(axis_permute_step,),
        )

    if len(lhs) == 1 and len(rhs) >= 2:
        axis_slice_step = AxisSliceSymbolicStep(
            lhs=lhs,
            rhs=rhs,
            program=build_axis_slice_symbolic_program(lhs, rhs),
            explicit_sizes_items=explicit_sizes_items,
        )
        return SymbolicPlan(
            kind="axis_slice",
            input_arity=len(lhs),
            output_arity=len(rhs),
            steps=(axis_slice_step,),
        )

    if len(lhs) >= 2 and len(rhs) == 1:
        output_rank = len(rhs[0])
        if all(len(input_terms) == output_rank for input_terms in lhs):
            raise ValidationError(
                code=ErrorCode.SEGMENT_STRADDLE,
                message=(
                    "segment straddle: concat segment ownership straddles "
                    "input boundaries"
                ),
                help=(
                    "use one concat signature whose output additive segments map "
                    "contiguously to input tensors"
                ),
                related=("rearrange lowering",),
                data={"operation": "rearrange"},
            )

    if len(lhs) == len(rhs):
        return SymbolicPlan(
            kind="route",
            input_arity=len(lhs),
            output_arity=len(rhs),
            steps=(),
        )

    raise ValidationError(
        code=ErrorCode.INCONSISTENT_DIMS,
        message=(
            "inconsistent dims: rearrange lowering could not resolve this signature "
            "into primitive steps"
        ),
        help=(
            "use signatures representable by permute/reshape/axis_slice/concat and deterministic tensor routing in v0.1"
        ),
        related=("rearrange lowering",),
        data={"operation": "rearrange"},
    )


__all__ = [
    "build_rearrange_symbolic_plan",
    "build_reshape_symbolic_program",
]
