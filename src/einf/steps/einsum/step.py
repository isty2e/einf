from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache

import opt_einsum

from einf.axis import AxisSide, ScalarAxisTerms
from einf.backend import BackendProfile, load_backend_module
from einf.diagnostics import ErrorCode, ValidationError
from einf.plans.context import PlanSelectionContext, build_runtime_execution_context
from einf.plans.scoring import einsum_output_shape, einsum_peak_numel
from einf.signature import Signature
from einf.steps.base import (
    RuntimeProgram,
    RuntimeSpecializationContext,
    RuntimeStep,
    SymbolicProgram,
    SymbolicStep,
    SymbolicStepScore,
)
from einf.tensor_types import TensorLike

from .equation import build_contract_equation
from .native import try_native_contract_einsum

_EINSUM_SYMBOLS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _parse_binary_einsum_equation(
    equation: str,
    /,
) -> tuple[str, str, str] | None:
    """Parse one binary einsum equation into input/output subscripts."""
    normalized = equation.replace(" ", "")
    left_right = normalized.split("->")
    if len(left_right) != 2:
        return None
    inputs_part, output_subscript = left_right
    input_subscripts = inputs_part.split(",")
    if len(input_subscripts) != 2:
        return None
    lhs_subscript, rhs_subscript = input_subscripts
    if not lhs_subscript or not rhs_subscript or not output_subscript:
        return None
    return lhs_subscript, rhs_subscript, output_subscript


def _prefer_native_matmul(
    *,
    operands: tuple[TensorLike, ...],
) -> bool:
    """Return whether direct matmul is preferred over einsum for one binary call."""
    return len(operands) == 2


def _operand_shapes_key(
    operands: tuple[TensorLike, ...],
    /,
) -> tuple[tuple[int, ...], ...] | None:
    """Build hashable operand shape key when every shape is tuple[int, ...]."""
    shape_key: list[tuple[int, ...]] = []
    for operand in operands:
        operand_shape = getattr(operand, "shape", None)
        if not isinstance(operand_shape, tuple):
            return None
        if any(type(dim) is not int for dim in operand_shape):
            return None
        shape_key.append(operand_shape)
    return tuple(shape_key)


@lru_cache(maxsize=4096)
def _cached_contract_expression(
    equation: str,
    operand_shapes: tuple[tuple[int, ...], ...],
):
    """Return cached opt_einsum contract expression for one equation/shape set."""
    return opt_einsum.contract_expression(
        equation,
        *operand_shapes,
        optimize="auto",
    )


@lru_cache(maxsize=256)
def _is_binary_matmul_equation(equation: str, /) -> bool:
    """Return whether one binary einsum equation is exactly matmul-shaped."""
    parsed = _parse_binary_einsum_equation(equation)
    if parsed is None:
        return False
    lhs_subscript, rhs_subscript, output_subscript = parsed
    if len(lhs_subscript) < 1 or len(rhs_subscript) < 2:
        return False
    if (
        len(set(lhs_subscript)) != len(lhs_subscript)
        or len(set(rhs_subscript)) != len(rhs_subscript)
        or len(set(output_subscript)) != len(output_subscript)
    ):
        return False

    contracted_label = lhs_subscript[-1]
    if rhs_subscript[0] != contracted_label:
        return False
    if contracted_label in output_subscript:
        return False
    if lhs_subscript.count(contracted_label) != 1:
        return False
    if rhs_subscript.count(contracted_label) != 1:
        return False

    expected_output = f"{lhs_subscript[:-1]}{rhs_subscript[1:]}"
    return output_subscript == expected_output


@dataclass(frozen=True, slots=True)
class EinsumSymbolicProgram(SymbolicProgram):
    """Precompiled einsum symbolic program."""

    input_arity: int
    output_arity: int
    equations: tuple[str, ...]
    chain_order: tuple[int, ...]
    carrier_index: int | None
    lhs: AxisSide | None
    rhs: AxisSide | None
    explicit_sizes_items: tuple[tuple[str, int], ...]
    allow_native_matmul: bool


@dataclass(frozen=True, slots=True)
class EinsumRuntimeProgram(RuntimeProgram):
    """Runtime einsum program with fully resolved equation set."""

    equations: tuple[str, ...]
    chain_order: tuple[int, ...]
    carrier_index: int | None
    allow_native_matmul: bool


def build_einsum_symbolic_program_from_equations(
    *,
    input_arity: int,
    output_arity: int,
    equations: tuple[str, ...],
    chain_order: tuple[int, ...] = (),
    carrier_index: int | None = None,
    allow_native_matmul: bool = False,
) -> EinsumSymbolicProgram:
    """Build one equation-driven einsum program."""
    program = EinsumSymbolicProgram(
        input_arity=input_arity,
        output_arity=output_arity,
        equations=equations,
        chain_order=chain_order,
        carrier_index=carrier_index,
        lhs=None,
        rhs=None,
        explicit_sizes_items=(),
        allow_native_matmul=allow_native_matmul,
    )
    _validate_einsum_program(program)
    return program


def build_einsum_symbolic_program_from_sides(
    *,
    lhs: AxisSide,
    rhs: AxisSide,
    explicit_sizes_items: tuple[tuple[str, int], ...],
    allow_native_matmul: bool = False,
) -> EinsumSymbolicProgram:
    """Build one side-driven einsum program."""
    program = EinsumSymbolicProgram(
        input_arity=len(lhs),
        output_arity=len(rhs),
        equations=(),
        chain_order=(),
        carrier_index=None,
        lhs=lhs,
        rhs=rhs,
        explicit_sizes_items=explicit_sizes_items,
        allow_native_matmul=allow_native_matmul,
    )
    _validate_einsum_program(program)
    return program


def _build_equation_from_scalar_terms(
    *,
    input_axis_terms: tuple[ScalarAxisTerms, ...],
    output_axis_terms: ScalarAxisTerms,
) -> str:
    """Build one deterministic einsum equation from scalarized axis terms."""
    total_terms = sum(len(terms) for terms in input_axis_terms) + len(output_axis_terms)
    if total_terms > len(_EINSUM_SYMBOLS):
        raise ValidationError(
            code=ErrorCode.INCONSISTENT_DIMS,
            message="inconsistent dims: too many scalar axes for einsum symbol budget",
            help="use fewer distinct scalar axes in one contraction step",
            related=("einsum equation",),
            data={},
        )

    key_to_symbol: dict[str, str] = {}

    def symbol_for(term_key: str) -> str:
        existing = key_to_symbol.get(term_key)
        if existing is not None:
            return existing
        assigned = _EINSUM_SYMBOLS[len(key_to_symbol)]
        key_to_symbol[term_key] = assigned
        return assigned

    input_term_keys: set[str] = set()
    input_subscripts: list[str] = []
    for axis_terms in input_axis_terms:
        chars: list[str] = []
        for term in axis_terms:
            token = term.stable_token()
            chars.append(symbol_for(token))
            input_term_keys.add(token)
        input_subscripts.append("".join(chars))

    seen_output_terms: set[str] = set()
    output_chars: list[str] = []
    for term in output_axis_terms:
        token = term.stable_token()
        if token in seen_output_terms:
            raise ValidationError(
                code=ErrorCode.INCONSISTENT_DIMS,
                message="inconsistent dims: contract output axis names must be unique",
                help="declare each output axis at most once",
                related=("contract schema",),
                data={"operation": "contract"},
            )
        seen_output_terms.add(token)
        if token not in input_term_keys:
            raise ValidationError(
                code=ErrorCode.INCONSISTENT_DIMS,
                message=(
                    "inconsistent dims: contract output axis terms must appear in "
                    "input terms"
                ),
                help="ensure every output axis appears in at least one input axis-list",
                related=("contract equation",),
                data={},
            )
        output_chars.append(symbol_for(token))

    return f"{','.join(input_subscripts)}->{''.join(output_chars)}"


def _equation_from_sides(
    *,
    lhs: AxisSide,
    rhs: AxisSide,
    explicit_sizes_items: tuple[tuple[str, int], ...],
    input_shapes: tuple[tuple[int, ...], ...],
) -> str:
    """Build one einsum equation from `(lhs, rhs)` with scalar fallback."""
    try:
        return build_contract_equation(
            input_axis_lists=lhs,
            output_axis_list=rhs[0],
        )
    except ValidationError as error:
        if error.code != ErrorCode.CONTRACT_NON_ATOMIC_AXIS.value:
            raise

    signature = Signature(inputs=lhs, outputs=rhs)
    context_explicit_sizes = signature.filter_explicit_sizes(dict(explicit_sizes_items))
    normalized_context = build_runtime_execution_context(
        signature=signature,
        tensors=(),
        explicit_sizes=context_explicit_sizes,
        input_shapes=input_shapes,
    )
    return _build_equation_from_scalar_terms(
        input_axis_terms=normalized_context.lhs_terms,
        output_axis_terms=normalized_context.rhs_terms[0],
    )


def _validate_einsum_program(program: EinsumSymbolicProgram) -> None:
    """Validate one einsum symbolic program."""
    has_equations = len(program.equations) > 0
    has_side_spec = program.lhs is not None or program.rhs is not None

    if has_equations and has_side_spec:
        raise ValueError(
            "einsum symbolic step must define either equations or side spec"
        )
    if not has_equations and not has_side_spec:
        raise ValueError("einsum symbolic step requires equations or side spec")

    if has_equations:
        if program.chain_order:
            if program.output_arity != 1:
                raise ValueError("chain einsum symbolic step must be N->1")
            if len(program.chain_order) != len(program.equations):
                raise ValueError(
                    "chain einsum step requires one equation per chain edge"
                )
            if program.carrier_index is None:
                raise ValueError("chain einsum step requires a carrier index")
            if (
                program.carrier_index < 0
                or program.carrier_index >= program.input_arity
            ):
                raise ValueError("chain einsum step carrier index is out of bounds")
            return

        if program.carrier_index is not None:
            raise ValueError("direct einsum symbolic step cannot have a carrier index")
        if program.output_arity != 1:
            raise ValueError("direct einsum symbolic step must be N->1")
        if len(program.equations) != 1:
            raise ValueError(
                "direct einsum symbolic step requires exactly one equation"
            )
        return

    if program.chain_order:
        raise ValueError("side-based einsum symbolic step cannot be chain mode")
    if program.carrier_index is not None:
        raise ValueError("side-based einsum symbolic step cannot set carrier index")
    if program.output_arity != 1:
        raise ValueError("side-based einsum symbolic step must be N->1")

    lhs = program.lhs
    rhs = program.rhs
    if lhs is None or rhs is None:
        raise ValueError("side-based einsum symbolic step requires lhs and rhs")
    if len(lhs) != program.input_arity:
        raise ValueError("side-based einsum symbolic step input arity mismatch")
    if len(rhs) != program.output_arity:
        raise ValueError("side-based einsum symbolic step output arity mismatch")


@dataclass(frozen=True, slots=True)
class EinsumSymbolicStep(SymbolicStep[EinsumSymbolicProgram]):
    """Symbolic einsum step with precompiled direct or side-driven program."""

    program: EinsumSymbolicProgram
    name: str = "einsum"
    input_arity: int = 0
    output_arity: int = 0

    def __post_init__(self) -> None:
        _validate_einsum_program(self.program)
        object.__setattr__(self, "input_arity", self.program.input_arity)
        object.__setattr__(self, "output_arity", self.program.output_arity)

    def _resolved_equations(
        self,
        *,
        input_shapes: tuple[tuple[int, ...], ...],
    ) -> tuple[str, ...]:
        if self.program.equations:
            return self.program.equations

        lhs = self.program.lhs
        rhs = self.program.rhs
        if lhs is None or rhs is None:
            raise ValueError("side-based einsum symbolic step is missing lhs/rhs")
        return (
            _equation_from_sides(
                lhs=lhs,
                rhs=rhs,
                explicit_sizes_items=self.program.explicit_sizes_items,
                input_shapes=input_shapes,
            ),
        )

    def preview_equations(self) -> tuple[str, ...]:
        """Return deterministic preview equations if available without runtime shapes."""
        if self.program.equations:
            return self.program.equations

        lhs = self.program.lhs
        rhs = self.program.rhs
        if lhs is None or rhs is None:
            return ()

        try:
            return (
                build_contract_equation(
                    input_axis_lists=lhs,
                    output_axis_list=rhs[0],
                ),
            )
        except ValidationError:
            return ()

    def specialize(
        self,
        context: RuntimeSpecializationContext,
        /,
    ) -> RuntimeStep:
        equations = self._resolved_equations(input_shapes=context.input_shapes)
        runtime_program = EinsumRuntimeProgram(
            equations=equations,
            chain_order=self.program.chain_order,
            carrier_index=self.program.carrier_index,
            allow_native_matmul=self.program.allow_native_matmul,
        )
        backend_profile = context.backend_profile
        executor = (
            None if backend_profile is None else _build_einsum_executor(backend_profile)
        )
        return EinsumRuntimeStep(
            name=self.name,
            input_arity=self.input_arity,
            output_arity=self.output_arity,
            program=runtime_program,
            backend_profile=backend_profile,
            executor=executor,
        )

    def score(self, context: PlanSelectionContext, /) -> SymbolicStepScore:
        tensor_shapes = context.input_shapes
        equations = self._resolved_equations(input_shapes=tensor_shapes)
        peak_numel = 0
        if self.program.chain_order:
            carrier_index = self.program.carrier_index
            if carrier_index is None or carrier_index >= len(tensor_shapes):
                return SymbolicStepScore(
                    peak_einsum_numel=0,
                    materialize_numel=0,
                    allocation_count=0,
                    kernel_count=len(self.program.equations),
                )

            carrier_shape = tensor_shapes[carrier_index]
            for equation, next_index in zip(
                equations,
                self.program.chain_order,
                strict=True,
            ):
                if next_index >= len(tensor_shapes):
                    continue
                local_peak = einsum_peak_numel(
                    equation=equation,
                    operand_shapes=(carrier_shape, tensor_shapes[next_index]),
                )
                if local_peak is not None:
                    peak_numel = max(peak_numel, local_peak)
                next_shape = einsum_output_shape(
                    equation=equation,
                    operand_shapes=(carrier_shape, tensor_shapes[next_index]),
                )
                if next_shape is not None:
                    carrier_shape = next_shape
        else:
            for equation in equations:
                local_peak = einsum_peak_numel(
                    equation=equation,
                    operand_shapes=tensor_shapes,
                )
                if local_peak is not None:
                    peak_numel = max(peak_numel, local_peak)

        return SymbolicStepScore(
            peak_einsum_numel=peak_numel,
            materialize_numel=0,
            allocation_count=1 if self.program.chain_order else len(equations),
            kernel_count=len(equations),
        )

    def specialization_depends_on_input_shapes(self) -> bool:
        """Return whether specialization depends on runtime input shapes."""
        return not self.program.equations


@dataclass(frozen=True, slots=True)
class _EinsumEquationExecutor:
    """Run one lowered einsum equation sequence under one backend profile."""

    profile: BackendProfile
    native_namespace_einsum: Callable[..., TensorLike] | None = None
    native_module_einsum: Callable[..., TensorLike] | None = None
    native_module_matmul: Callable[[TensorLike, TensorLike], TensorLike] | None = None

    def run(
        self,
        equation: str,
        operands: tuple[TensorLike, ...],
        chain_mode: bool,
        allow_native_matmul: bool,
    ) -> TensorLike:
        """Execute one equation with native-fallback dispatch and error mapping."""
        operand_shapes = _operand_shapes_key(operands)
        if operand_shapes is not None:
            should_use_cached_expression = False
            if len(operands) > 2 and not chain_mode:
                should_use_cached_expression = True
            elif len(operands) == 2 and not allow_native_matmul:
                should_use_cached_expression = True
            if should_use_cached_expression:
                try:
                    expression = _cached_contract_expression(equation, operand_shapes)
                    return expression(*operands)
                except Exception:
                    pass

        if (
            allow_native_matmul
            and len(operands) == 2
            and self.native_module_matmul is not None
            and _is_binary_matmul_equation(equation)
            and _prefer_native_matmul(operands=operands)
        ):
            try:
                return self.native_module_matmul(
                    operands[0],
                    operands[1],
                )
            except Exception:
                pass

        module_einsum = self.native_module_einsum
        if module_einsum is not None:
            try:
                return module_einsum(equation, *operands)
            except Exception:
                pass

        namespace_einsum = self.native_namespace_einsum
        if namespace_einsum is not None and len(operands) == 2:
            try:
                native_output = namespace_einsum(equation, *operands)
            except Exception:
                native_output = try_native_contract_einsum(
                    equation=equation,
                    tensors=operands,
                    namespace=self.profile.namespace,
                )
            if native_output is not None:
                return native_output

        try:
            return opt_einsum.contract(
                equation,
                *operands,
                optimize="auto",
            )
        except Exception as error:
            if chain_mode:
                message = f"inconsistent dims: chain einsum execution failed: {error}"
                help_message = (
                    "ensure input shapes satisfy chain einsum lowering constraints"
                )
            else:
                message = f"inconsistent dims: einsum execution failed: {error}"
                help_message = "ensure input shapes satisfy einsum lowering constraints"
            raise ValidationError(
                code=ErrorCode.INCONSISTENT_DIMS,
                message=message,
                help=help_message,
                related=("einsum execution",),
                data={"operation": "einsum"},
            ) from error


def _build_einsum_executor(profile: BackendProfile, /) -> _EinsumEquationExecutor:
    """Build one runtime einsum executor for one backend profile."""
    module_einsum: Callable[..., TensorLike] | None = None
    module_matmul: Callable[[TensorLike, TensorLike], TensorLike] | None = None
    backend_family = profile.backend_family
    if backend_family is not None:
        try:
            backend_module = load_backend_module(backend_family)
        except (ModuleNotFoundError, ValueError):
            backend_module = None
        if backend_module is not None:
            module_einsum_candidate = getattr(backend_module, "einsum", None)
            if callable(module_einsum_candidate):
                module_einsum = module_einsum_candidate
            module_matmul_candidate = getattr(backend_module, "matmul", None)
            if callable(module_matmul_candidate):
                module_matmul = module_matmul_candidate

    namespace_einsum = getattr(profile.namespace, "einsum", None)
    if not callable(namespace_einsum):
        namespace_einsum = None
    return _EinsumEquationExecutor(
        profile=profile,
        native_namespace_einsum=namespace_einsum,
        native_module_einsum=module_einsum,
        native_module_matmul=module_matmul,
    )


@dataclass(frozen=True, slots=True)
class EinsumRuntimeStep(RuntimeStep[EinsumRuntimeProgram]):
    """Runtime step for precomputed direct or chain einsum equations."""

    name: str
    input_arity: int
    output_arity: int
    program: EinsumRuntimeProgram
    backend_profile: BackendProfile | None
    executor: _EinsumEquationExecutor | None = None

    def _build_executor(self, profile: BackendProfile, /) -> _EinsumEquationExecutor:
        """Build one runtime einsum executor for one backend profile."""
        return _build_einsum_executor(profile)

    def _resolve_executor(
        self,
        /,
    ) -> _EinsumEquationExecutor:
        executor = self.executor
        if executor is not None:
            return executor
        profile = self.backend_profile
        if profile is None:
            raise ValidationError(
                code=ErrorCode.BACKEND_DISPATCH_UNSUPPORTED_INPUT,
                message=(
                    "backend dispatch unsupported input: einsum runtime requires "
                    "a resolved backend profile"
                ),
                help="execute through AbstractPlan/TensorOp call path to resolve backend profile",
                related=("backend dispatch",),
                data={"operation": "einsum"},
            )
        return self._build_executor(profile)

    def _run_direct(
        self,
        tensors: tuple[TensorLike, ...],
        /,
    ) -> tuple[TensorLike, ...]:
        executor = self._resolve_executor()
        equations = self.program.equations
        if len(equations) == 1:
            return (
                executor.run(
                    equations[0],
                    tensors,
                    False,
                    self.program.allow_native_matmul,
                ),
            )

        outputs: list[TensorLike] = []
        for equation in equations:
            outputs.append(
                executor.run(
                    equation,
                    tensors,
                    False,
                    self.program.allow_native_matmul,
                )
            )
        return tuple(outputs)

    def _run_chain(
        self,
        tensors: tuple[TensorLike, ...],
        /,
    ) -> tuple[TensorLike, ...]:
        carrier_index = self.program.carrier_index
        if carrier_index is None:
            raise ValidationError(
                code=ErrorCode.INCONSISTENT_DIMS,
                message="inconsistent dims: chain einsum runtime step is missing carrier index",
                help="rebuild TensorOp symbolic plan and retry",
                related=("einsum runtime",),
                data={"operation": "einsum"},
            )

        if len(self.program.chain_order) != len(self.program.equations):
            raise ValidationError(
                code=ErrorCode.INCONSISTENT_DIMS,
                message=(
                    "inconsistent dims: chain einsum runtime step has mismatched "
                    "equation/chain lengths"
                ),
                help="rebuild TensorOp symbolic plan with aligned chain metadata",
                related=("einsum runtime",),
                data={"operation": "einsum"},
            )

        executor = self._resolve_executor()
        carrier_tensor = tensors[carrier_index]
        for equation, next_index in zip(
            self.program.equations,
            self.program.chain_order,
            strict=True,
        ):
            carrier_tensor = executor.run(
                equation,
                (carrier_tensor, tensors[next_index]),
                True,
                self.program.allow_native_matmul,
            )

        return (carrier_tensor,)

    def run(
        self,
        tensors: tuple[TensorLike, ...],
        /,
    ) -> tuple[TensorLike, ...]:
        outputs = (
            self._run_chain(tensors)
            if self.program.chain_order
            else self._run_direct(tensors)
        )
        if len(outputs) != self.output_arity:
            raise ValidationError(
                code=ErrorCode.INCONSISTENT_DIMS,
                message=(
                    "inconsistent dims: einsum runtime output arity mismatch: "
                    f"expected {self.output_arity}, got {len(outputs)}"
                ),
                help="ensure symbolic output arity matches resolved runtime outputs",
                related=("einsum execution",),
                data={"operation": "einsum"},
            )
        return outputs


__all__ = [
    "build_einsum_symbolic_program_from_equations",
    "build_einsum_symbolic_program_from_sides",
    "EinsumSymbolicProgram",
    "EinsumRuntimeProgram",
    "EinsumRuntimeStep",
    "EinsumSymbolicStep",
]
