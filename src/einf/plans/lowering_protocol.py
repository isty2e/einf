from abc import ABC, abstractmethod

from einf.axis import AxisSide
from einf.ir import IRProgram

from .symbolic import SymbolicPlan


class LoweringProgram(ABC):
    """Lowering policy that emits symbolic candidates from abstract operation form."""

    @abstractmethod
    def ir_program(
        self,
        *,
        op_name: str,
        lhs: AxisSide,
        rhs: AxisSide,
        explicit_sizes_items: tuple[tuple[str, int], ...],
    ) -> IRProgram:
        """Return canonical lowering IR program for one abstract operation."""
        raise NotImplementedError

    @abstractmethod
    def symbolic_candidates(
        self,
        *,
        ir_program: IRProgram,
        explicit_sizes_items: tuple[tuple[str, int], ...],
    ) -> tuple[SymbolicPlan, ...]:
        """Return ordered symbolic plan candidates."""
        raise NotImplementedError


__all__ = ["LoweringProgram"]
