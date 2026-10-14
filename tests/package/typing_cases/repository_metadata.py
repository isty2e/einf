from dataclasses import dataclass
from pathlib import Path

from einf.analysis.checkers import (
    CheckerAdapter,
    CheckerRequest,
    CheckerResult,
)
from einf.analysis.checkers.pyright import PyrightAdapter
from einf.steps.base import (
    RuntimeSpecializationContext,
    RuntimeStep,
    StepProgram,
    SymbolicStep,
    SymbolicStepScore,
)
from einf.steps.context import PlanSelectionContext
from einf.tensor_types import TensorLike


class IdentityProgram(StepProgram):
    pass


@dataclass(frozen=True, slots=True)
class IdentityRuntimeStep(RuntimeStep[IdentityProgram]):
    name: str
    input_arity: int
    output_arity: int
    program: IdentityProgram

    def run(self, tensors: tuple[TensorLike, ...], /) -> tuple[TensorLike, ...]:
        return tensors


@dataclass(frozen=True, slots=True)
class IdentitySymbolicStep(SymbolicStep[IdentityProgram]):
    name: str
    input_arity: int
    output_arity: int
    program: IdentityProgram

    def score(self, context: PlanSelectionContext, /) -> SymbolicStepScore:
        return SymbolicStepScore(0, 0, 0, 0)

    def specialize(
        self, context: RuntimeSpecializationContext, /
    ) -> IdentityRuntimeStep:
        return IdentityRuntimeStep(
            self.name, self.input_arity, self.output_arity, self.program
        )


class MutableAdapter(CheckerAdapter):
    name: str
    executable: str

    def __init__(self) -> None:
        self.name = "mutable"
        self.executable = "initial"

    def build_command(self, request: CheckerRequest, /) -> tuple[str, ...]:
        return (self.executable, *(str(path) for path in request.targets))

    def parse_output(
        self, *, stdout: str, stderr: str, request: CheckerRequest
    ) -> CheckerResult:
        return CheckerResult(diagnostics=(), failures=())


program = IdentityProgram()
concrete_runtime = IdentityRuntimeStep("identity", 0, 0, program)
runtime: RuntimeStep[StepProgram] = concrete_runtime
assert runtime.run(()) == ()
assert runtime.program is program

symbolic: SymbolicStep[StepProgram] = IdentitySymbolicStep("identity", 0, 0, program)
specialized = symbolic.specialize(RuntimeSpecializationContext((), None))
assert specialized.run(()) == ()
assert specialized.program is program

request = CheckerRequest(targets=(Path("example.py"),), project_root=Path("."))
mutable = MutableAdapter()
mutable.executable = "updated"
consumer: CheckerAdapter = mutable
assert consumer.build_command(request) == ("updated", "example.py")
assert consumer.normalize_output(
    returncode=0, stdout="", stderr="", request=request
) == CheckerResult(diagnostics=(), failures=())

frozen: CheckerAdapter = PyrightAdapter("pyright", "pyright")
assert frozen.build_command(request) == ("pyright", "--outputjson", "example.py")
