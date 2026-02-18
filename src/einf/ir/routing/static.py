from einf.axis import Axis, AxisInt, AxisSide, AxisTerms


def precompute_route_output_indices(
    lhs: AxisSide,
    rhs: AxisSide,
) -> tuple[int, ...] | None:
    """Return deterministic output->input routing when statically unique."""
    for terms in lhs:
        if any(not isinstance(term, (Axis, AxisInt)) for term in terms):
            return None
    for terms in rhs:
        if any(not isinstance(term, (Axis, AxisInt)) for term in terms):
            return None

    lhs_by_terms: dict[AxisTerms, list[int]] = {}
    for input_index, input_terms in enumerate(lhs):
        indices = lhs_by_terms.get(input_terms)
        if indices is None:
            lhs_by_terms[input_terms] = [input_index]
            continue
        indices.append(input_index)

    mapping: list[int] = []
    used_inputs: set[int] = set()
    for output_terms in rhs:
        candidates = lhs_by_terms.get(output_terms)
        if candidates is None:
            return None
        available_candidates = tuple(
            input_index for input_index in candidates if input_index not in used_inputs
        )
        if len(available_candidates) != 1:
            return None
        selected_input = available_candidates[0]
        used_inputs.add(selected_input)
        mapping.append(selected_input)
    return tuple(mapping)


__all__ = ["precompute_route_output_indices"]
