from .rules import (
    EINSUM_AXIS_SLICE_RULE,
    EINSUM_BINARY_RULE,
    PERMUTE_EXPAND_RULE,
    PERMUTE_PERMUTE_RULE,
    RESHAPE_RESHAPE_RULE,
)
from .types import RuntimeStepFusionRule

RUNTIME_STEP_FUSION_RULES: tuple[RuntimeStepFusionRule, ...] = (
    EINSUM_AXIS_SLICE_RULE,
    PERMUTE_EXPAND_RULE,
    PERMUTE_PERMUTE_RULE,
    RESHAPE_RESHAPE_RULE,
    EINSUM_BINARY_RULE,
)

RUNTIME_STEP_FUSION_RULES_BY_WINDOW: dict[int, tuple[RuntimeStepFusionRule, ...]] = {}
for runtime_step_fusion_rule in RUNTIME_STEP_FUSION_RULES:
    existing_rules = RUNTIME_STEP_FUSION_RULES_BY_WINDOW.get(
        runtime_step_fusion_rule.window_size
    )
    if existing_rules is None:
        RUNTIME_STEP_FUSION_RULES_BY_WINDOW[runtime_step_fusion_rule.window_size] = (
            runtime_step_fusion_rule,
        )
    else:
        RUNTIME_STEP_FUSION_RULES_BY_WINDOW[runtime_step_fusion_rule.window_size] = (
            *existing_rules,
            runtime_step_fusion_rule,
        )


__all__ = ["RUNTIME_STEP_FUSION_RULES_BY_WINDOW"]
