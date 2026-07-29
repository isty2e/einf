from ..axis import Axis
from ..diagnostics import ErrorCode, ValidationError
from ..signature import Signature


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


__all__ = ["validate_contract_atomic_terms"]
