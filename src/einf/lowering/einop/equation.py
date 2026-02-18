from einf.axis import AxisPack, AxisTermBase, AxisTerms
from einf.diagnostics import ErrorCode, ValidationError
from einf.signature import Signature


def has_nary_contraction_candidate(signature: Signature) -> bool:
    """Return whether n-ary signature has at least one contractible shared term."""
    output_terms = {term for axis_list in signature.outputs for term in axis_list}
    input_presence: dict[AxisTermBase, int] = {}
    for axis_list in signature.inputs:
        for term in set(axis_list):
            if term in input_presence:
                input_presence[term] += 1
                continue
            input_presence[term] = 1

    return any(
        count >= 2 and term not in output_terms
        for term, count in input_presence.items()
    )


def ordered_unique_axis_terms(*axis_lists: AxisTerms) -> AxisTerms:
    """Collect ordered unique terms across axis-term tuples."""
    seen: set[AxisTermBase] = set()
    ordered: list[AxisTermBase] = []
    for axis_list in axis_lists:
        for term in axis_list:
            if term in seen:
                continue
            seen.add(term)
            ordered.append(term)
    return AxisTerms(tuple(ordered))


def all_subset_axis_lists(
    *,
    ordered_terms: AxisTerms,
    target_terms: set[AxisTermBase],
    remaining_terms: set[AxisTermBase],
) -> tuple[AxisTerms, ...]:
    """Enumerate ordered subset axis-term tuples for one pairwise chain step."""
    candidate_rows: list[tuple[int, int, int, int, str, AxisTerms]] = []
    total_terms = len(ordered_terms)
    total_masks = 1 << total_terms
    for mask in range(total_masks):
        selected_terms: list[AxisTermBase] = []
        selected_set: set[AxisTermBase] = set()
        for index, term in enumerate(ordered_terms):
            if mask & (1 << index):
                selected_terms.append(term)
                selected_set.add(term)

        target_miss = sum(
            1
            for term in target_terms
            if term in ordered_terms and term not in selected_set
        )
        remaining_miss = sum(
            1
            for term in remaining_terms
            if term in ordered_terms and term not in selected_set
        )
        redundant_selected = sum(
            1
            for term in selected_set
            if term not in target_terms and term not in remaining_terms
        )
        candidate_rows.append(
            (
                target_miss,
                remaining_miss,
                redundant_selected,
                len(selected_terms),
                "|".join(term.stable_token() for term in selected_terms),
                AxisTerms(tuple(selected_terms)),
            )
        )

    candidate_rows.sort()
    return tuple(row[5] for row in candidate_rows)


def build_einop_equations(
    *,
    input_axis_lists: tuple[AxisTerms, ...],
    output_axis_lists: tuple[AxisTerms, ...],
) -> tuple[str, ...]:
    """Build deterministic einsum equations for each output axis-term tuple."""
    equations: list[str] = []
    for output_axis_list in output_axis_lists:
        key_to_symbol: dict[str, str] = {}
        symbols = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"

        if sum(
            len(axis_list) for axis_list in input_axis_lists + (output_axis_list,)
        ) > len(symbols):
            raise ValidationError(
                code=ErrorCode.INCONSISTENT_DIMS,
                message="inconsistent dims: too many atomic axes for einsum symbol budget",
                help="use fewer distinct atomic axis symbols in one einop output",
                related=("einop equation",),
                data={"operation": "einop"},
            )

        def symbol_for(key: str) -> str:
            existing = key_to_symbol.get(key)
            if existing is not None:
                return existing
            assigned = symbols[len(key_to_symbol)]
            key_to_symbol[key] = assigned
            return assigned

        input_subscripts: list[str] = []
        input_term_keys: set[str] = set()
        for input_axis_list in input_axis_lists:
            seen_keys: set[str] = set()
            input_keys: list[str] = []
            for term in input_axis_list:
                if isinstance(term, AxisPack):
                    raise ValidationError(
                        code=ErrorCode.INCONSISTENT_DIMS,
                        message="inconsistent dims: unresolved axis pack in einsum equation build",
                        help="ensure axis packs are expanded before lowering to einsum",
                        related=("einop equation",),
                        data={"operation": "einop"},
                    )
                term_key = term.stable_token()
                if term_key in seen_keys:
                    raise ValidationError(
                        code=ErrorCode.INCONSISTENT_DIMS,
                        message=(
                            "inconsistent dims: einop input axis terms must be unique "
                            "within each input tensor"
                        ),
                        help="declare each input axis term at most once per input axis-list",
                        related=("einop schema",),
                        data={"operation": "einop"},
                    )
                seen_keys.add(term_key)
                input_keys.append(term_key)
                input_term_keys.add(term_key)
            input_subscripts.append(
                "".join(symbol_for(term_key) for term_key in input_keys)
            )

        output_keys: list[str] = []
        seen_output_keys: set[str] = set()
        for term in output_axis_list:
            if isinstance(term, AxisPack):
                raise ValidationError(
                    code=ErrorCode.INCONSISTENT_DIMS,
                    message="inconsistent dims: unresolved axis pack in einsum equation build",
                    help="ensure axis packs are expanded before lowering to einsum",
                    related=("einop equation",),
                    data={"operation": "einop"},
                )
            term_key = term.stable_token()
            if term_key in seen_output_keys:
                raise ValidationError(
                    code=ErrorCode.INCONSISTENT_DIMS,
                    message="inconsistent dims: einop output axis terms must be unique",
                    help="declare each output axis term at most once per output tensor",
                    related=("einop schema",),
                    data={"operation": "einop"},
                )
            seen_output_keys.add(term_key)
            output_keys.append(term_key)

        missing_keys = tuple(
            term_key for term_key in output_keys if term_key not in input_term_keys
        )
        if missing_keys:
            raise ValidationError(
                code=ErrorCode.INCONSISTENT_DIMS,
                message="inconsistent dims: einop output axis terms must appear in input terms",
                help="ensure every output axis term appears in at least one input axis-list",
                related=("einop schema",),
                data={"operation": "einop", "missing": len(missing_keys)},
            )

        output_subscript = "".join(symbol_for(term_key) for term_key in output_keys)
        equations.append(f"{','.join(input_subscripts)}->{output_subscript}")

    return tuple(equations)


__all__ = [
    "all_subset_axis_lists",
    "build_einop_equations",
    "has_nary_contraction_candidate",
    "ordered_unique_axis_terms",
]
