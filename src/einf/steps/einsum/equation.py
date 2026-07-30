from einf.axis import Axis, AxisSide, AxisTerms
from einf.diagnostics import ErrorCode, ValidationError


def build_contract_equation(
    *,
    input_axis_lists: AxisSide,
    output_axis_list: AxisTerms,
) -> str:
    """Build deterministic einsum equation for one atomic contract."""
    key_to_symbol: dict[str, str] = {}
    symbols = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"

    def symbol_for(key: str) -> str:
        existing = key_to_symbol.get(key)
        if existing is not None:
            return existing

        if len(key_to_symbol) >= len(symbols):
            raise ValidationError(
                code=ErrorCode.INCONSISTENT_DIMS,
                message="inconsistent dims: too many atomic axes for einsum symbol budget",
                help="use fewer distinct atomic axis symbols in one contract call",
                related=("contract equation",),
                data={"operation": "contract"},
            )
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
                    data={"operation": "contract"},
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
                data={"operation": "contract"},
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
                data={"operation": "contract"},
            )
        output_subscript_chars.append(symbol_for(term_key))

    return f"{','.join(input_subscripts)}->{''.join(output_subscript_chars)}"
