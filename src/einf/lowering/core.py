from abc import ABC, abstractmethod
from dataclasses import dataclass

try:
    from typing import Self
except ImportError:  # pragma: no cover
    from typing_extensions import Self

from einf.axis import AxisSide
from einf.ir import IRProgram, build_default_ir_program
from einf.plans.symbolic import SymbolicPlan
from einf.reduction.schema import ReducerPlan

from .compile import build_symbolic_candidates_from_ir


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


@dataclass(frozen=True, slots=True)
class EmptyLoweringProgram(LoweringProgram):
    """Lowering policy with no symbolic candidate output."""

    def ir_program(
        self,
        *,
        op_name: str,
        lhs: AxisSide,
        rhs: AxisSide,
        explicit_sizes_items: tuple[tuple[str, int], ...],
    ) -> IRProgram:
        _ = explicit_sizes_items
        return build_default_ir_program(
            op_name=op_name,
            lhs=lhs,
            rhs=rhs,
        )

    def symbolic_candidates(
        self,
        *,
        ir_program: IRProgram,
        explicit_sizes_items: tuple[tuple[str, int], ...],
    ) -> tuple[SymbolicPlan, ...]:
        return ()


@dataclass(frozen=True, slots=True)
class StaticLoweringProgram(LoweringProgram):
    """Lowering policy that always returns the same symbolic candidates."""

    candidates: tuple[SymbolicPlan, ...]
    ir: IRProgram | None = None

    def ir_program(
        self,
        *,
        op_name: str,
        lhs: AxisSide,
        rhs: AxisSide,
        explicit_sizes_items: tuple[tuple[str, int], ...],
    ) -> IRProgram:
        _ = explicit_sizes_items
        if self.ir is not None:
            return self.ir
        return build_default_ir_program(
            op_name=op_name,
            lhs=lhs,
            rhs=rhs,
        )

    def symbolic_candidates(
        self,
        *,
        ir_program: IRProgram,
        explicit_sizes_items: tuple[tuple[str, int], ...],
    ) -> tuple[SymbolicPlan, ...]:
        return self.candidates


@dataclass(frozen=True, slots=True)
class DefaultLoweringProgram(LoweringProgram):
    """Deterministic default Abstract->Symbolic lowering rules."""

    reducer_plan: ReducerPlan | None = None

    def with_reducer_plan(self, reducer_plan: ReducerPlan | None) -> Self:
        """Return a configured lowering program with fixed reducer plan."""
        if reducer_plan == self.reducer_plan:
            return self
        return type(self)(reducer_plan=reducer_plan)

    def symbolic_candidates(
        self,
        *,
        ir_program: IRProgram,
        explicit_sizes_items: tuple[tuple[str, int], ...],
    ) -> tuple[SymbolicPlan, ...]:
        return build_symbolic_candidates_from_ir(
            ir_program=ir_program,
            explicit_sizes_items=explicit_sizes_items,
            reducer_plan=self.reducer_plan,
        )

    def ir_program(
        self,
        *,
        op_name: str,
        lhs: AxisSide,
        rhs: AxisSide,
        explicit_sizes_items: tuple[tuple[str, int], ...],
    ) -> IRProgram:
        _ = explicit_sizes_items
        return build_default_ir_program(
            op_name=op_name,
            lhs=lhs,
            rhs=rhs,
        )


__all__ = [
    "LoweringProgram",
    "DefaultLoweringProgram",
    "EmptyLoweringProgram",
    "StaticLoweringProgram",
]
