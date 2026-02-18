from .rules import (
    EINSUM_AXIS_SLICE_RULE,
    EINSUM_BINARY_RULE,
    PERMUTE_EXPAND_RULE,
    PERMUTE_PERMUTE_RULE,
    RESHAPE_RESHAPE_RULE,
)
from .types import TupleFusionRule

TUPLE_FUSION_RULES: tuple[TupleFusionRule, ...] = (
    EINSUM_AXIS_SLICE_RULE,
    PERMUTE_EXPAND_RULE,
    PERMUTE_PERMUTE_RULE,
    RESHAPE_RESHAPE_RULE,
    EINSUM_BINARY_RULE,
)

TUPLE_FUSION_RULES_BY_WINDOW: dict[int, tuple[TupleFusionRule, ...]] = {}
for tuple_fusion_rule in TUPLE_FUSION_RULES:
    existing_rules = TUPLE_FUSION_RULES_BY_WINDOW.get(tuple_fusion_rule.window_size)
    if existing_rules is None:
        TUPLE_FUSION_RULES_BY_WINDOW[tuple_fusion_rule.window_size] = (
            tuple_fusion_rule,
        )
    else:
        TUPLE_FUSION_RULES_BY_WINDOW[tuple_fusion_rule.window_size] = (
            *existing_rules,
            tuple_fusion_rule,
        )


__all__ = ["TUPLE_FUSION_RULES_BY_WINDOW"]
