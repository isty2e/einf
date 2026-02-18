from dataclasses import dataclass

from einf.axis import Axis, AxisExpr, AxisInt, ScalarAxisTerms, term_size


@dataclass(frozen=True, slots=True)
class AtomicAxisToken:
    """One atomic axis token used by plan-level axis matching."""

    label: str | None
    size: int


PieceMapping = tuple[int | None, ...]


def scalar_terms_to_atomic_tokens(
    terms: ScalarAxisTerms,
    axis_sizes: dict[str, int],
) -> tuple[AtomicAxisToken, ...]:
    """Convert one scalar-axis tuple into atomic axis tokens."""
    tokens: list[AtomicAxisToken] = []
    for term in terms:
        if isinstance(term, Axis):
            tokens.append(
                AtomicAxisToken(
                    label=term.name,
                    size=axis_sizes[term.name],
                )
            )
            continue
        if isinstance(term, AxisInt):
            tokens.append(AtomicAxisToken(label=None, size=term.value))
            continue
        if isinstance(term, AxisExpr):
            tokens.append(AtomicAxisToken(label=None, size=term_size(term, axis_sizes)))
            continue
        raise TypeError("unsupported scalar term")
    return tuple(tokens)


def collect_axis_mappings(
    input_tokens: tuple[AtomicAxisToken, ...],
    output_tokens: tuple[AtomicAxisToken, ...],
    *,
    allow_broadcast: bool,
) -> tuple[PieceMapping, ...]:
    """Collect up to two valid output-axis source mappings."""
    input_count = len(input_tokens)
    cache: dict[tuple[int, int], tuple[PieceMapping, ...]] = {}
    full_unused_mask = (1 << input_count) - 1
    remaining_label_requirements: list[dict[tuple[str, int], int]] = [
        {} for _ in range(len(output_tokens))
    ]
    running_requirements: dict[tuple[str, int], int] = {}
    for output_index in range(len(output_tokens) - 1, -1, -1):
        remaining_label_requirements[output_index] = dict(running_requirements)
        output_token = output_tokens[output_index]
        if output_token.label is not None:
            requirement_key = (output_token.label, output_token.size)
            running_requirements[requirement_key] = (
                running_requirements.get(requirement_key, 0) + 1
            )

    def search(
        output_index: int,
        unused_mask: int,
    ) -> tuple[PieceMapping, ...]:
        cache_key = (output_index, unused_mask)
        if cache_key in cache:
            return cache[cache_key]

        if output_index == len(output_tokens):
            if unused_mask != 0:
                cache[cache_key] = ()
                return ()
            cache[cache_key] = ((),)
            return ((),)

        output_token = output_tokens[output_index]
        exact_candidates: list[int] = []
        fallback_candidates: list[int] = []
        available_label_counts: dict[tuple[str, int], int] = {}
        for input_index in range(input_count):
            if unused_mask & (1 << input_index) == 0:
                continue
            input_token = input_tokens[input_index]
            if input_token.label is None:
                continue
            key = (input_token.label, input_token.size)
            available_label_counts[key] = available_label_counts.get(key, 0) + 1

        required_labels = remaining_label_requirements[output_index]
        for input_index in range(input_count):
            if unused_mask & (1 << input_index) == 0:
                continue
            input_token = input_tokens[input_index]
            if input_token.size != output_token.size:
                continue
            if input_token.label == output_token.label:
                exact_candidates.append(input_index)
                continue
            if input_token.label is None or output_token.label is None:
                if output_token.label is None and input_token.label is not None:
                    key = (input_token.label, input_token.size)
                    required = required_labels.get(key, 0)
                    available = available_label_counts.get(key, 0)
                    if required > 0 and available <= required:
                        continue
                fallback_candidates.append(input_index)

        if output_token.label is not None and exact_candidates:
            fallback_candidates = []

        results: list[PieceMapping] = []
        seen: set[PieceMapping] = set()
        for input_index in exact_candidates + fallback_candidates:
            next_unused_mask = unused_mask & ~(1 << input_index)
            tails = search(output_index + 1, next_unused_mask)
            for tail in tails:
                result = (input_index, *tail)
                if result in seen:
                    continue
                seen.add(result)
                results.append(result)
                if len(results) >= 2:
                    cache[cache_key] = tuple(results)
                    return tuple(results)

        if allow_broadcast:
            tails = search(output_index + 1, unused_mask)
            for tail in tails:
                result = (None, *tail)
                if result in seen:
                    continue
                seen.add(result)
                results.append(result)
                if len(results) >= 2:
                    cache[cache_key] = tuple(results)
                    return tuple(results)

        cache[cache_key] = tuple(results)
        return tuple(results)

    return search(0, full_unused_mask)


__all__ = [
    "AtomicAxisToken",
    "collect_axis_mappings",
    "PieceMapping",
    "scalar_terms_to_atomic_tokens",
]
