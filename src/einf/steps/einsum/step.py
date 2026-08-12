from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal, cast

import opt_einsum
from array_api_compat import array_namespace

from einf.axis import AxisSide, ScalarAxisTerms
from einf.backend import BACKEND_POLICY, BackendProfile
from einf.backend.namespace import output_namespace_matches
from einf.backend.runtime import (
    BackendRuntimeUnavailable,
    is_backend_runtime_uninterposed,
    load_backend_module,
    supports_native_array_ops,
)
from einf.diagnostics import ErrorCode, ExecutionError, TensorOpError, ValidationError
from einf.signature import Signature
from einf.steps.base import (
    RuntimeProgram,
    RuntimeSpecializationContext,
    RuntimeStep,
    SymbolicProgram,
    SymbolicStep,
    SymbolicStepScore,
)
from einf.steps.context import PlanSelectionContext, build_runtime_execution_context
from einf.steps.runtime import (
    FALLBACK_ELIGIBLE_BACKEND_ERRORS,
    validate_runtime_output_shape,
)
from einf.steps.scoring import einsum_output_shape, einsum_peak_numel
from einf.tensor_types import TensorLike, TrustedTensorFamily, trusted_tensor_family

from .equation import build_contract_equation

_EINSUM_SYMBOLS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
_CONTRACT_EXPRESSION_CACHE_MAXSIZE = 2_048


def _parse_binary_einsum_equation(
    equation: str,
    /,
) -> tuple[str, str, str] | None:
    """Parse one binary einsum equation into input/output subscripts."""
    if equation.count("->") != 1:
        return None
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
    if any(
        symbol not in _EINSUM_SYMBOLS
        for subscript in (lhs_subscript, rhs_subscript, output_subscript)
        for symbol in subscript
    ):
        return None
    return lhs_subscript, rhs_subscript, output_subscript


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


@lru_cache(maxsize=_CONTRACT_EXPRESSION_CACHE_MAXSIZE)
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
    if len(lhs_subscript) < 1 or len(rhs_subscript) != 2:
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
    """Runtime equations with specialization-proven matmul admissions.

    Parameters
    ----------
    equations : tuple[str, ...]
        Fully resolved einsum equations.
    chain_order : tuple[int, ...]
        Operand indices consumed by a chain program.
    carrier_index : int or None
        Initial carrier operand for a chain program.
    native_matmul_equations : frozenset[str]
        Equations proven equivalent to the native matmul primitive.

    Raises
    ------
    ValueError
        A native matmul admission is absent from ``equations`` or is not
        semantically matmul-shaped.
    """

    equations: tuple[str, ...]
    chain_order: tuple[int, ...]
    carrier_index: int | None
    native_matmul_equations: frozenset[str]

    def __post_init__(self) -> None:
        if not self.native_matmul_equations.issubset(self.equations):
            raise ValueError(
                "native matmul admissions must belong to the runtime equations"
            )
        if any(
            not _is_binary_matmul_equation(equation)
            for equation in self.native_matmul_equations
        ):
            raise ValueError(
                "native matmul admissions must be semantically matmul-shaped"
            )


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
    key_to_symbol: dict[str, str] = {}

    def symbol_for(term_key: str) -> str:
        existing = key_to_symbol.get(term_key)
        if existing is not None:
            return existing
        if len(key_to_symbol) >= len(_EINSUM_SYMBOLS):
            raise ValidationError(
                code=ErrorCode.INCONSISTENT_DIMS,
                message="inconsistent dims: too many scalar axes for einsum symbol budget",
                help="use fewer distinct scalar axes in one contraction step",
                related=("einsum equation",),
                data={"operation": "contract"},
            )
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
                data={"operation": "contract"},
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
        if program.output_arity != len(program.equations):
            raise ValueError(
                "direct einsum symbolic step requires one equation per output"
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

    def requires_einsum_backend(self) -> bool:
        """Return that einsum steps require an einsum-capable backend."""
        return True

    def specialize(
        self,
        context: RuntimeSpecializationContext,
        /,
    ) -> RuntimeStep:
        equations = self._resolved_equations(input_shapes=context.input_shapes)
        native_matmul_equations = (
            frozenset(
                equation
                for equation in equations
                if _is_binary_matmul_equation(equation)
            )
            if self.program.allow_native_matmul
            else frozenset()
        )
        runtime_program = EinsumRuntimeProgram(
            equations=equations,
            chain_order=self.program.chain_order,
            carrier_index=self.program.carrier_index,
            native_matmul_equations=native_matmul_equations,
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
class EinsumEquationExecutor:
    """Run lowered einsum equations through specialized backend routes.

    Parameters
    ----------
    profile
        Resolved namespace profile for the operands.
    namespace_einsum
        Einsum primitive provided by the resolved namespace, if available.
    native_module_einsum
        Einsum primitive from a supported native backend module, if available.
    native_module_matmul
        Matmul primitive from a supported native backend module, if available.
    opt_einsum_supported
        Whether ``opt_einsum`` routes are admitted for the backend family.
    """

    profile: BackendProfile
    namespace_einsum: Callable[..., TensorLike] | None = None
    native_module_einsum: Callable[..., TensorLike] | None = None
    native_module_matmul: Callable[[TensorLike, TensorLike], TensorLike] | None = None
    opt_einsum_supported: bool = True

    def run(
        self,
        equation: str,
        operands: tuple[TensorLike, ...],
        chain_mode: bool,
        native_matmul_admitted: bool,
    ) -> TensorLike:
        """Execute one equation with native-fallback dispatch and error mapping."""
        native_operand_family = self._trusted_native_operand_family(operands)
        namespace_error: Exception | None = None
        if native_operand_family is None:
            namespace_output, namespace_error = self._try_namespace_einsum(
                equation=equation,
                operands=operands,
            )
            if namespace_output is not None:
                return namespace_output

        operand_shapes = _operand_shapes_key(operands)
        if operand_shapes is not None:
            should_use_cached_expression = False
            if (len(operands) > 2 and not chain_mode) or (
                len(operands) == 2 and not native_matmul_admitted
            ):
                should_use_cached_expression = True
            if should_use_cached_expression and self.opt_einsum_supported:
                try:
                    expression = _cached_contract_expression(equation, operand_shapes)
                    output = expression(*operands)
                except TensorOpError:
                    raise
                except FALLBACK_ELIGIBLE_BACKEND_ERRORS:
                    pass
                except Exception as error:
                    raise _project_einsum_route_error(error) from error
                else:
                    return _validate_einsum_output(
                        equation=equation,
                        operands=operands,
                        output=output,
                        profile=self.profile,
                        native_operand_family=native_operand_family,
                    )

        if (
            native_operand_family is not None
            and native_matmul_admitted
            and len(operands) == 2
            and self.native_module_matmul is not None
        ):
            try:
                output = self.native_module_matmul(
                    operands[0],
                    operands[1],
                )
            except TensorOpError:
                raise
            except FALLBACK_ELIGIBLE_BACKEND_ERRORS:
                pass
            except Exception as error:
                raise _project_einsum_route_error(error) from error
            else:
                return _trust_or_validate_native_einsum_output(
                    equation=equation,
                    operands=operands,
                    output=output,
                    profile=self.profile,
                    route_callable=self.native_module_matmul,
                    module_op_name="matmul",
                    native_operand_family=native_operand_family,
                )

        module_einsum = self.native_module_einsum
        if native_operand_family is not None and module_einsum is not None:
            try:
                output = module_einsum(equation, *operands)
            except TensorOpError:
                raise
            except FALLBACK_ELIGIBLE_BACKEND_ERRORS:
                pass
            except Exception as error:
                raise _project_einsum_route_error(error) from error
            else:
                return _trust_or_validate_native_einsum_output(
                    equation=equation,
                    operands=operands,
                    output=output,
                    profile=self.profile,
                    route_callable=module_einsum,
                    module_op_name="einsum",
                    native_operand_family=native_operand_family,
                )

        if native_operand_family is not None:
            namespace_output, namespace_error = self._try_namespace_einsum(
                equation=equation,
                operands=operands,
            )
            if namespace_output is not None:
                return namespace_output

        if not self.opt_einsum_supported:
            if namespace_error is not None:
                raise _project_einsum_route_error(namespace_error) from namespace_error
            BACKEND_POLICY.validate_einsum_capability(
                profile=self.profile,
                op_name="einsum",
            )
            raise RuntimeError("einsum executor has no admitted execution route")

        try:
            output = opt_einsum.contract(
                equation,
                *operands,
                optimize="auto",
            )
        except TensorOpError:
            raise
        except Exception as error:
            if chain_mode:
                message = f"backend execution failed: chain einsum failed: {error}"
                help_message = (
                    "ensure input shapes satisfy chain einsum lowering constraints"
                )
            else:
                message = f"backend execution failed: einsum failed: {error}"
                help_message = "ensure input shapes satisfy einsum lowering constraints"
            raise ExecutionError(
                code=ErrorCode.BACKEND_EXECUTION_FAILED,
                message=message,
                help=help_message,
                related=("einsum execution",),
                data={"operation": "einsum"},
            ) from error
        return _validate_einsum_output(
            equation=equation,
            operands=operands,
            output=output,
            profile=self.profile,
            native_operand_family=native_operand_family,
        )

    def _try_namespace_einsum(
        self,
        *,
        equation: str,
        operands: tuple[TensorLike, ...],
    ) -> tuple[TensorLike | None, Exception | None]:
        """Run the selected namespace route once when it is available."""
        namespace_einsum = self.namespace_einsum
        if namespace_einsum is None:
            return None, None
        try:
            output = namespace_einsum(equation, *operands)
        except TensorOpError:
            raise
        except FALLBACK_ELIGIBLE_BACKEND_ERRORS as error:
            return None, error
        except Exception as error:
            raise _project_einsum_route_error(error) from error
        return (
            _validate_einsum_output(
                equation=equation,
                operands=operands,
                output=output,
                profile=self.profile,
                native_operand_family=None,
            ),
            None,
        )

    def _trusted_native_operand_family(
        self,
        operands: tuple[TensorLike, ...],
        /,
    ) -> TrustedTensorFamily | None:
        """Return the native family proven by exact operands and runtime state."""
        if not operands:
            return None
        operand_type = type(operands[0])
        backend_family = trusted_tensor_family(operand_type)
        if backend_family is None or self.profile.backend_family != backend_family:
            return None
        if any(type(operand) is not operand_type for operand in operands[1:]):
            return None
        if not is_backend_runtime_uninterposed(backend_family):
            return None
        return backend_family


def _trust_or_validate_native_einsum_output(
    *,
    equation: str,
    operands: tuple[TensorLike, ...],
    output: TensorLike,
    profile: BackendProfile,
    route_callable: Callable[..., TensorLike],
    module_op_name: Literal["einsum", "matmul"],
    native_operand_family: TrustedTensorFamily,
) -> TensorLike:
    """Skip semantic validation only for exact native tensor execution."""
    if operands and type(output) is type(operands[0]):
        try:
            backend_module = load_backend_module(native_operand_family)
            canonical_callable = getattr(
                backend_module,
                module_op_name,
                None,
            )
        except Exception:  # noqa: BLE001 - missing proof selects validation
            canonical_callable = None
        if route_callable is canonical_callable:
            return output
    return _validate_einsum_output(
        equation=equation,
        operands=operands,
        output=output,
        profile=profile,
        native_operand_family=native_operand_family,
    )


def _validate_einsum_output(
    *,
    equation: str,
    operands: tuple[TensorLike, ...],
    output: TensorLike,
    profile: BackendProfile,
    native_operand_family: TrustedTensorFamily | None,
) -> TensorLike:
    """Validate one output from an untrusted einsum execution route."""
    operand_shapes = _operand_shapes_key(operands)
    if operand_shapes is None:
        raise ExecutionError(
            code=ErrorCode.OP_OUTPUT_PROTOCOL_VIOLATION,
            message="einsum output protocol violation: operand shapes are not canonical",
            help="use TensorLike operands with tuple[int, ...] shape",
            related=("einsum execution", "TensorLike input protocol"),
            data={"operation": "einsum"},
        )
    expected_shape = einsum_output_shape(
        equation=equation,
        operand_shapes=operand_shapes,
    )
    if expected_shape is None:
        raise ExecutionError(
            code=ErrorCode.INCONSISTENT_DIMS,
            message="inconsistent dims: einsum output shape could not be derived",
            help="ensure the lowered equation matches all operand shapes",
            related=("einsum execution", "einsum equation"),
            data={"operation": "einsum"},
        )
    validated_output = validate_runtime_output_shape(
        output,
        expected_shape,
        operation="einsum",
    )
    if (
        native_operand_family is not None
        and operands
        and type(validated_output) is type(operands[0])
    ):
        return validated_output
    try:
        output_namespace = array_namespace(validated_output)
    except TensorOpError:
        raise
    except Exception as error:
        raise ExecutionError(
            code=ErrorCode.OP_OUTPUT_PROTOCOL_VIOLATION,
            message=(
                "einsum output protocol violation: output namespace could not "
                f"be resolved: {error}"
            ),
            help="return an einsum output with a valid array namespace",
            related=("einsum execution", "TensorOp output protocol"),
            data={"operation": "einsum"},
        ) from error
    if not output_namespace_matches(profile.namespace, output_namespace):
        raise ExecutionError(
            code=ErrorCode.OP_OUTPUT_PROTOCOL_VIOLATION,
            message=(
                "einsum output protocol violation: output belongs to a different "
                "backend namespace"
            ),
            help="return an einsum output from the selected input namespace",
            related=("einsum execution", "TensorOp output protocol"),
            data={"operation": "einsum"},
        )
    return validated_output


def _project_einsum_route_error(error: Exception) -> ExecutionError:
    """Build one structured error for an unexpected einsum route failure."""
    return ExecutionError(
        code=ErrorCode.BACKEND_EXECUTION_FAILED,
        message=f"backend execution failed: einsum execution failed: {error}",
        help="ensure input shapes satisfy einsum lowering constraints",
        related=("einsum execution",),
        data={"operation": "einsum"},
    )


def _build_einsum_executor(profile: BackendProfile, /) -> EinsumEquationExecutor:
    """Build one runtime einsum executor for one backend profile."""
    module_einsum: Callable[..., TensorLike] | None = None
    module_matmul: Callable[[TensorLike, TensorLike], TensorLike] | None = None
    backend_family = profile.backend_family
    if backend_family is not None and supports_native_array_ops(backend_family):
        try:
            backend_module = load_backend_module(backend_family)
        except BackendRuntimeUnavailable:
            backend_module = None
        except TensorOpError:
            raise
        except Exception as error:
            raise _project_einsum_route_error(error) from error
        if backend_module is not None:
            try:
                module_einsum_candidate = getattr(backend_module, "einsum", None)
                if callable(module_einsum_candidate):
                    module_einsum = cast(
                        Callable[..., TensorLike],
                        module_einsum_candidate,
                    )
                module_matmul_candidate = getattr(backend_module, "matmul", None)
                if callable(module_matmul_candidate):
                    module_matmul = cast(
                        Callable[[TensorLike, TensorLike], TensorLike],
                        module_matmul_candidate,
                    )
            except TensorOpError:
                raise
            except Exception as error:
                raise _project_einsum_route_error(error) from error

    namespace_einsum = BACKEND_POLICY.resolve_namespace_einsum(profile)
    if namespace_einsum is module_einsum:
        namespace_einsum = None
    return EinsumEquationExecutor(
        profile=profile,
        namespace_einsum=namespace_einsum,
        native_module_einsum=module_einsum,
        native_module_matmul=module_matmul,
        opt_einsum_supported=BACKEND_POLICY.supports_opt_einsum(profile.backend_family),
    )


@dataclass(frozen=True, slots=True)
class EinsumRuntimeStep(RuntimeStep[EinsumRuntimeProgram]):
    """Runtime step for precomputed direct or chain einsum equations."""

    name: str
    input_arity: int
    output_arity: int
    program: EinsumRuntimeProgram
    backend_profile: BackendProfile | None
    executor: EinsumEquationExecutor | None = None

    def _build_executor(self, profile: BackendProfile, /) -> EinsumEquationExecutor:
        """Build one runtime einsum executor for one backend profile."""
        return _build_einsum_executor(profile)

    def _resolve_executor(
        self,
        /,
    ) -> EinsumEquationExecutor:
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
                    equations[0] in self.program.native_matmul_equations,
                ),
            )

        outputs: list[TensorLike] = []
        for equation in equations:
            outputs.append(
                executor.run(
                    equation,
                    tensors,
                    False,
                    equation in self.program.native_matmul_equations,
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
                equation in self.program.native_matmul_equations,
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
    "EinsumRuntimeProgram",
    "EinsumRuntimeStep",
    "EinsumSymbolicProgram",
    "EinsumSymbolicStep",
    "build_einsum_symbolic_program_from_equations",
    "build_einsum_symbolic_program_from_sides",
]
