from dataclasses import dataclass

from einf.axis import (
    AxisExpr,
    AxisSide,
    AxisTermBase,
    AxisTerms,
    ScalarAxisTermBase,
    flatten_mul_children,
)
from einf.signature import Signature

_ExtentKey = tuple[str, ...]
_LayoutKey = tuple[str, ...]
_CompositeKey = tuple[_LayoutKey, str]


def _multiplicative_layout(
    term: ScalarAxisTermBase,
) -> tuple[_ExtentKey, _LayoutKey] | None:
    if not isinstance(term, AxisExpr) or term.operator != "*":
        return None

    layout_key = tuple(factor.stable_token() for factor in flatten_mul_children(term))
    return tuple(sorted(layout_key)), layout_key


def _collect_expanded_extent_keys(signature: Signature) -> frozenset[_ExtentKey]:
    composites_by_extent: dict[_ExtentKey, set[_CompositeKey]] = {}
    extent_keys_by_factor: dict[str, set[_ExtentKey]] = {}
    atomic_factor_tokens: set[str] = set()
    for side in (signature.inputs, signature.outputs):
        for axis_terms in side:
            for term in axis_terms:
                if not isinstance(term, ScalarAxisTermBase):
                    continue
                layout = _multiplicative_layout(term)
                if layout is None:
                    atomic_factor_tokens.add(term.stable_token())
                    continue
                extent_key, layout_key = layout
                composite_key = (layout_key, term.stable_token())
                composites_by_extent.setdefault(extent_key, set()).add(composite_key)
                for factor_token in extent_key:
                    extent_keys_by_factor.setdefault(factor_token, set()).add(
                        extent_key
                    )

    expanded_extent_keys = {
        extent_key
        for extent_key, composite_keys in composites_by_extent.items()
        if len(composite_keys) > 1
    }
    for extent_key in composites_by_extent:
        for factor_token in extent_key:
            sibling_extent_keys = extent_keys_by_factor[factor_token] - {extent_key}
            if factor_token in atomic_factor_tokens or sibling_extent_keys:
                expanded_extent_keys.add(extent_key)
                break
    return frozenset(expanded_extent_keys)


def _expand_selected_products(
    side: AxisSide,
    *,
    extent_keys: frozenset[_ExtentKey],
    side_name: str,
) -> AxisSide:
    normalized_terms: list[AxisTerms] = []
    for axis_terms in side:
        normalized_terms.append(
            _expand_selected_terms(axis_terms, extent_keys=extent_keys)
        )
    return AxisSide.from_spec(tuple(normalized_terms), side_name=side_name)


def _expand_selected_terms(
    axis_terms: AxisTerms,
    *,
    extent_keys: frozenset[_ExtentKey],
) -> AxisTerms:
    expanded: list[AxisTermBase] = []
    for term in axis_terms:
        if isinstance(term, ScalarAxisTermBase):
            layout = _multiplicative_layout(term)
            if layout is not None and layout[0] in extent_keys:
                expanded.extend(flatten_mul_children(term))
                continue
        expanded.append(term)
    return AxisTerms.from_spec(tuple(expanded))


@dataclass(frozen=True, slots=True)
class EinopLayoutNormalization:
    """Separate requested composite layouts from logical factor axes."""

    requested: Signature
    logical: Signature
    expanded_extent_keys: frozenset[_ExtentKey]

    def __post_init__(self) -> None:
        if len(self.requested.inputs) != len(self.logical.inputs):
            raise ValueError("einop layout normalization changed input arity")
        if len(self.requested.outputs) != len(self.logical.outputs):
            raise ValueError("einop layout normalization changed output arity")
        if self.requested != self.logical and not self.expanded_extent_keys:
            raise ValueError("einop layout normalization requires expanded extent keys")

    @classmethod
    def from_signature(cls, signature: Signature) -> "EinopLayoutNormalization":
        """Expand products whose factors cross a top-level layout boundary."""
        extent_keys = _collect_expanded_extent_keys(signature)
        if not extent_keys:
            return cls(
                requested=signature,
                logical=signature,
                expanded_extent_keys=frozenset(),
            )

        logical = Signature(
            inputs=_expand_selected_products(
                signature.inputs,
                extent_keys=extent_keys,
                side_name="lhs",
            ),
            outputs=_expand_selected_products(
                signature.outputs,
                extent_keys=extent_keys,
                side_name="rhs",
            ),
        )
        return cls(
            requested=signature,
            logical=logical,
            expanded_extent_keys=extent_keys,
        )

    def normalize_terms(self, axis_terms: AxisTerms) -> AxisTerms:
        """Project requested composite terms into logical factor axes."""
        return _expand_selected_terms(
            axis_terms,
            extent_keys=self.expanded_extent_keys,
        )

    def generated_duplicate_terms(self) -> tuple[str, ...]:
        """Return repeated logical terms introduced by product expansion."""
        duplicates: dict[str, str] = {}
        for requested_side, logical_side in (
            (self.requested.inputs, self.logical.inputs),
            (self.requested.outputs, self.logical.outputs),
        ):
            for requested_terms, logical_terms in zip(
                requested_side,
                logical_side,
                strict=True,
            ):
                requested_counts = requested_terms.term_counts()
                for term, logical_count in logical_terms.term_counts().items():
                    if logical_count > 1 and logical_count > requested_counts.get(
                        term, 0
                    ):
                        duplicates[term.stable_token()] = term.to_dsl()
        return tuple(duplicates[token] for token in sorted(duplicates))

    @property
    def is_required(self) -> bool:
        """Return whether requested and logical factor layouts differ."""
        return self.requested != self.logical


__all__ = ["EinopLayoutNormalization"]
