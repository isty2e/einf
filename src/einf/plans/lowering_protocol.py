from abc import ABC, abstractmethod

from einf.ir import IRProgram, LoweringSignature

from .symbolic import SymbolicPlan


class LoweringProgram(ABC):
    """Lowering policy that emits symbolic candidates from abstract operation form."""

    @abstractmethod
    def ir_program(
        self,
        source: LoweringSignature,
        /,
    ) -> IRProgram:
        """Return the IR program for one structural operation.

        Parameters
        ----------
        source : LoweringSignature
            Canonical structural input to lowering.

        Returns
        -------
        IRProgram
            IR program whose source matches ``source``.
        """
        raise NotImplementedError

    @abstractmethod
    def symbolic_candidates(
        self,
        *,
        ir_program: IRProgram,
    ) -> tuple[SymbolicPlan, ...]:
        """Return ordered symbolic plans for an IR program.

        Parameters
        ----------
        ir_program : IRProgram
            Canonical lowering input and observability trace.

        Returns
        -------
        tuple[SymbolicPlan, ...]
            Ordered candidates whose sources match ``ir_program.source``.
        """
        raise NotImplementedError


__all__ = ["LoweringProgram"]
