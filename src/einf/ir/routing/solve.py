from einf.axis.matching import AtomicAxisToken, collect_axis_mappings
from einf.diagnostics import ErrorCode, ValidationError


def resolve_route_token_indices(
    *,
    input_tokens: tuple[tuple[AtomicAxisToken, ...], ...],
    output_tokens: tuple[tuple[AtomicAxisToken, ...], ...],
) -> tuple[int, ...]:
    """Resolve unique output->input tensor routing under identity-axis constraint."""
    identity_output_candidates: list[tuple[int, ...]] = []
    general_output_candidates: list[tuple[int, ...]] = []
    has_axis_mapping_ambiguity = False
    for output_index, output_token_group in enumerate(output_tokens):
        identity_matching_inputs: list[int] = []
        general_matching_inputs: list[int] = []
        for input_index, input_token_group in enumerate(input_tokens):
            mappings = collect_axis_mappings(
                input_token_group,
                output_token_group,
                allow_broadcast=False,
            )
            if not mappings:
                continue
            if len(mappings) > 1:
                has_axis_mapping_ambiguity = True
            general_matching_inputs.append(input_index)
            rank = len(output_token_group)
            identity = tuple(range(rank))
            if any(mapping == identity for mapping in mappings):
                identity_matching_inputs.append(input_index)
        if not general_matching_inputs:
            raise ValidationError(
                code=ErrorCode.INCONSISTENT_DIMS,
                message=(
                    "inconsistent dims: route could not match output tensor to "
                    f"any input tensor (output index {output_index})"
                ),
                help=(
                    "align lhs/rhs tensor groups so each output has one compatible "
                    "input source"
                ),
                related=("route lowering",),
                data={"operation": "rearrange"},
            )
        identity_output_candidates.append(tuple(identity_matching_inputs))
        general_output_candidates.append(tuple(general_matching_inputs))

    identity_solution_count, identity_solution = _count_route_solutions(
        output_candidates=identity_output_candidates
    )
    if identity_solution_count == 1 and identity_solution is not None:
        return identity_solution
    if identity_solution_count > 1:
        raise ValidationError(
            code=ErrorCode.AMBIGUOUS_DIMS,
            message=(
                "ambiguous dims: route has multiple valid identity tensor routings"
            ),
            help="add axis names or with_sizes constraints to make routing unique",
            related=("route lowering",),
            data={"operation": "rearrange"},
        )

    general_solution_count, general_solution = _count_route_solutions(
        output_candidates=general_output_candidates
    )
    if general_solution_count == 0:
        raise ValidationError(
            code=ErrorCode.INCONSISTENT_DIMS,
            message=(
                "inconsistent dims: route could not find a valid one-to-one "
                "tensor routing"
            ),
            help="align lhs/rhs tensor groups so each output maps to one unique input",
            related=("route lowering",),
            data={"operation": "rearrange"},
        )
    if general_solution_count > 1 or has_axis_mapping_ambiguity:
        raise ValidationError(
            code=ErrorCode.AMBIGUOUS_DIMS,
            message="ambiguous dims: route has multiple valid tensor routings",
            help="add axis names or with_sizes constraints to make routing unique",
            related=("route lowering",),
            data={"operation": "rearrange"},
        )
    _ = general_solution
    raise ValidationError(
        code=ErrorCode.INCONSISTENT_DIMS,
        message=(
            "inconsistent dims: route lowering requires identity axis mapping per "
            "output tensor"
        ),
        help=("use split/concat/permute/reshape lowering for axis-level remapping"),
        related=("route lowering",),
        data={"operation": "rearrange"},
    )


def _count_route_solutions(
    *,
    output_candidates: list[tuple[int, ...]],
) -> tuple[int, tuple[int, ...] | None]:
    """Count one-to-one routings up to two solutions and keep first solution."""
    solution_count = 0
    first_solution: tuple[int, ...] | None = None
    selected_inputs: list[int] = []

    def search(output_index: int, used_mask: int) -> bool:
        nonlocal first_solution, solution_count
        if output_index == len(output_candidates):
            solution_count += 1
            if solution_count == 1:
                first_solution = tuple(selected_inputs)
                return False
            return True

        for input_index in output_candidates[output_index]:
            input_bit = 1 << input_index
            if used_mask & input_bit:
                continue
            selected_inputs.append(input_index)
            should_stop = search(output_index + 1, used_mask | input_bit)
            selected_inputs.pop()
            if should_stop:
                return True
        return False

    _ = search(0, 0)
    return solution_count, first_solution


__all__ = ["resolve_route_token_indices"]
