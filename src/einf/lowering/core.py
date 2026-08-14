from dataclasses import dataclass

try:
    from typing import Self
except ImportError:  # pragma: no cover
    from typing_extensions import Self

from einf.ir import IRProgram, LoweringSignature
from einf.plans.lowering_protocol import LoweringProgram
from einf.plans.symbolic import SymbolicPlan
from einf.reduction.schema import ReducerPlan

from .compile import build_symbolic_candidates_from_ir


@dataclass(frozen=True, slots=True)
class EmptyLoweringProgram(LoweringProgram):
    """Lowering policy with no symbolic candidate output."""

    def ir_program(
        self,
        source: LoweringSignature,
        /,
    ) -> IRProgram:
        """Build default IR for a source with no candidate lowering.

        Parameters
        ----------
        source : LoweringSignature
            Canonical structural input.

        Returns
        -------
        IRProgram
            Source-bound IR with the default trace.
        """
        return IRProgram.from_source(source)

    def symbolic_candidates(
        self,
        *,
        ir_program: IRProgram,
    ) -> tuple[SymbolicPlan, ...]:
        """Return no candidates for an IR program.

        Parameters
        ----------
        ir_program : IRProgram
            Source-bound lowering IR.

        Returns
        -------
        tuple[SymbolicPlan, ...]
            Empty candidate sequence.
        """
        return ()


@dataclass(frozen=True, slots=True)
class StaticLoweringProgram(LoweringProgram):
    """Lowering policy that always returns the same symbolic candidates."""

    candidates: tuple[SymbolicPlan, ...]
    ir: IRProgram | None = None

    def ir_program(
        self,
        source: LoweringSignature,
        /,
    ) -> IRProgram:
        """Return the configured IR or build one for the source.

        Parameters
        ----------
        source : LoweringSignature
            Canonical structural input.

        Returns
        -------
        IRProgram
            Configured or default source-bound IR.
        """
        if self.ir is not None:
            return self.ir
        return IRProgram.from_source(source)

    def symbolic_candidates(
        self,
        *,
        ir_program: IRProgram,
    ) -> tuple[SymbolicPlan, ...]:
        """Return the configured symbolic candidates.

        Parameters
        ----------
        ir_program : IRProgram
            Source-bound lowering IR supplied by the caller.

        Returns
        -------
        tuple[SymbolicPlan, ...]
            Configured candidate sequence.
        """
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
    ) -> tuple[SymbolicPlan, ...]:
        """Compile default symbolic candidates for an IR program.

        Parameters
        ----------
        ir_program : IRProgram
            Source-bound lowering IR.

        Returns
        -------
        tuple[SymbolicPlan, ...]
            Ordered default candidates.
        """
        return build_symbolic_candidates_from_ir(
            ir_program=ir_program,
            reducer_plan=self.reducer_plan,
        )

    def ir_program(
        self,
        source: LoweringSignature,
        /,
    ) -> IRProgram:
        """Build default IR for a structural source.

        Parameters
        ----------
        source : LoweringSignature
            Canonical structural input.

        Returns
        -------
        IRProgram
            Source-bound IR with the default trace.
        """
        return IRProgram.from_source(source)


__all__ = [
    "DefaultLoweringProgram",
    "EmptyLoweringProgram",
    "LoweringProgram",
    "StaticLoweringProgram",
]
