from dataclasses import dataclass


@dataclass(frozen=True, slots=True, order=True)
class SymbolicPlanScore:
    """Deterministic symbolic-plan score."""

    peak_einsum_numel: int
    pre_einsum_materialize_numel: int
    post_einsum_materialize_numel: int
    allocation_count: int
    kernel_count: int
    step_count: int


__all__ = ["SymbolicPlanScore"]
