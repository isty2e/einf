from einf.axis import Axis, AxisSide, AxisTerms
from einf.diagnostics import ErrorCode, ValidationError
from einf.signature import Signature


def _collect_contract_axis_names(signature: Signature) -> tuple[list[str], list[str]]:
    """Validate contract atomic terms and collect input/output axis names."""
    input_axis_names: list[str] = []
    for input_index, axis_list in enumerate(signature.inputs):
        for axis_index, term in enumerate(axis_list):
            if not isinstance(term, Axis):
                raise ValidationError(
                    code=ErrorCode.CONTRACT_NON_ATOMIC_AXIS,
                    message=(
                        "contract non-atomic axis: "
                        "contract only supports atomic axis names in v0.1"
                    ),
                    help="use atomic axis names in contract",
                    related=("contract axis expression",),
                    data={
                        "operation": "contract",
                        "side": "lhs",
                        "input_index": input_index,
                        "axis_index": axis_index,
                        "term": term.to_dsl(),
                    },
                )
            input_axis_names.append(term.name)

    output_axis_names: list[str] = []
    for axis_index, term in enumerate(signature.outputs[0]):
        if not isinstance(term, Axis):
            raise ValidationError(
                code=ErrorCode.CONTRACT_NON_ATOMIC_AXIS,
                message=(
                    "contract non-atomic axis: "
                    "contract only supports atomic axis names in v0.1"
                ),
                help="use atomic axis names in contract",
                related=("contract axis expression",),
                data={
                    "operation": "contract",
                    "side": "rhs",
                    "axis_index": axis_index,
                    "term": term.to_dsl(),
                },
            )
        output_axis_names.append(term.name)
    return input_axis_names, output_axis_names


def validate_contract_atomic_terms(signature: Signature) -> None:
    """Validate that contract signature terms are atomic axis names."""
    _collect_contract_axis_names(signature)


def validate_contract_atomic(signature: Signature) -> None:
    """Validate contract atomic-term and output-axis consistency constraints."""
    input_axis_names, output_axis_names = _collect_contract_axis_names(signature)

    if len(set(output_axis_names)) != len(output_axis_names):
        raise ValidationError(
            code=ErrorCode.INCONSISTENT_DIMS,
            message="inconsistent dims: contract output axis names must be unique",
            help="declare each output axis at most once",
            related=("contract schema",),
            data={"operation": "contract"},
        )

    input_axis_set = set(input_axis_names)
    missing_output_axes = [
        name for name in output_axis_names if name not in input_axis_set
    ]
    if missing_output_axes:
        raise ValidationError(
            code=ErrorCode.INCONSISTENT_DIMS,
            message="inconsistent dims: contract output axes must appear in input axes",
            help="ensure every output axis name appears in at least one input",
            related=("contract schema",),
            data={
                "operation": "contract",
                "missing": ",".join(missing_output_axes),
            },
        )


def build_contract_equation(
    *,
    input_axis_lists: AxisSide,
    output_axis_list: AxisTerms,
) -> str:
    """Build deterministic einsum equation for one atomic contract."""
    key_to_symbol: dict[str, str] = {}
    symbols = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    if sum(
        len(axis_list) for axis_list in input_axis_lists + (output_axis_list,)
    ) > len(symbols):
        raise ValidationError(
            code=ErrorCode.INCONSISTENT_DIMS,
            message="inconsistent dims: too many atomic axes for einsum symbol budget",
            help="use fewer distinct atomic axis symbols in one contract call",
            related=("contract equation",),
            data={},
        )

    def symbol_for(key: str) -> str:
        existing = key_to_symbol.get(key)
        if existing is not None:
            return existing

        assigned = symbols[len(key_to_symbol)]
        key_to_symbol[key] = assigned
        return assigned

    input_subscripts: list[str] = []
    input_term_keys: set[str] = set()
    for axis_list in input_axis_lists:
        subscript_chars: list[str] = []
        for term in axis_list:
            if not isinstance(term, Axis):
                raise ValidationError(
                    code=ErrorCode.CONTRACT_NON_ATOMIC_AXIS,
                    message=(
                        "contract non-atomic axis: contract only supports "
                        "atomic axis symbols"
                    ),
                    help=(
                        "rewrite product/concat expressions into explicit axes "
                        "before using contract"
                    ),
                    related=("contract equation",),
                    data={},
                )
            term_key = term.stable_token()
            subscript_chars.append(symbol_for(term_key))
            input_term_keys.add(term_key)
        input_subscripts.append("".join(subscript_chars))

    output_subscript_chars: list[str] = []
    seen_output_terms: set[str] = set()
    for term in output_axis_list:
        if not isinstance(term, Axis):
            raise ValidationError(
                code=ErrorCode.CONTRACT_NON_ATOMIC_AXIS,
                message=(
                    "contract non-atomic axis: contract only supports "
                    "atomic axis symbols"
                ),
                help=(
                    "rewrite product/concat expressions into explicit axes "
                    "before using contract"
                ),
                related=("contract equation",),
                data={},
            )
        term_key = term.stable_token()
        if term_key in seen_output_terms:
            raise ValidationError(
                code=ErrorCode.INCONSISTENT_DIMS,
                message="inconsistent dims: contract output axis names must be unique",
                help="declare each output axis at most once",
                related=("contract schema",),
                data={"operation": "contract"},
            )
        seen_output_terms.add(term_key)
        if term_key not in input_term_keys:
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
        output_subscript_chars.append(symbol_for(term_key))

    return f"{','.join(input_subscripts)}->{''.join(output_subscript_chars)}"
