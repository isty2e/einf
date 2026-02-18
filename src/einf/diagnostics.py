from enum import Enum
from typing import Literal, TypeAlias

DiagnosticSeverity: TypeAlias = Literal["error"]
DiagnosticValue: TypeAlias = str | int | bool


class ErrorCode(str, Enum):
    """Canonical internal diagnostic codes."""

    DIM_SOLVE_ERROR = "dim_solve_error"
    INCONSISTENT_DIMS = "inconsistent_dims"
    AMBIGUOUS_DIMS = "ambiguous_dims"
    NUMEL_MISMATCH_GROW = "numel_mismatch_grow"
    NUMEL_MISMATCH_SHRINK = "numel_mismatch_shrink"
    CONTRACT_NON_ATOMIC_AXIS = "contract_non_atomic_axis"
    OP_ARITY_MISMATCH = "op_arity_mismatch"
    OP_OUTPUT_PROTOCOL_VIOLATION = "op_output_protocol_violation"
    MULTI_INPUT_NOT_ALLOWED = "multi_input_not_allowed"
    NOT_A_VIEW = "not_a_view"
    SEGMENT_STRADDLE = "segment_straddle"
    BACKEND_DISPATCH_UNSUPPORTED_INPUT = "backend_dispatch_unsupported_input"
    BACKEND_DISPATCH_MIXED_FAMILY = "backend_dispatch_mixed_family"
    BACKEND_REQUIRED_EXTENSION_MISSING = "backend_required_extension_missing"


class TensorOpError(ValueError):
    """Structured base error for validation/runtime diagnostics."""

    channel = "error"
    severity: DiagnosticSeverity
    code: str
    external_code: str
    help: str | None
    related: tuple[str, ...]
    data: dict[str, DiagnosticValue]
    message: str

    @staticmethod
    def _normalize_code(code: str | ErrorCode) -> str:
        """Normalize code to canonical internal `snake_case` form."""
        if isinstance(code, ErrorCode):
            return code.value

        if not isinstance(code, str):
            raise TypeError("diagnostic code must be a string or ErrorCode")
        if not code:
            raise ValueError("diagnostic code cannot be empty")
        if any(char.isspace() for char in code):
            raise ValueError("diagnostic code cannot contain whitespace")

        upper_allowed = all(
            char.isupper() or char.isdigit() or char == "_" for char in code
        )
        lower_allowed = all(
            char.islower() or char.isdigit() or char == "_" for char in code
        )
        if upper_allowed and code[0].isalpha():
            return code.lower()
        if lower_allowed and code[0].isalpha():
            return code

        raise ValueError(
            "diagnostic code must be snake_case (or UPPER_SNAKE for compatibility)"
        )

    @staticmethod
    def _normalize_related(related: tuple[str, ...]) -> tuple[str, ...]:
        """Validate related notes."""
        normalized_related: list[str] = []
        for note in related:
            if not isinstance(note, str):
                raise TypeError("related diagnostics must be tuple[str, ...]")
            if not note.strip():
                raise ValueError("related diagnostic note cannot be empty")
            normalized_related.append(note)
        return tuple(normalized_related)

    @staticmethod
    def _normalize_data(data: dict[str, DiagnosticValue]) -> dict[str, DiagnosticValue]:
        """Validate and copy diagnostic payload data."""
        normalized_data: dict[str, DiagnosticValue] = {}
        for key, value in data.items():
            if not isinstance(key, str):
                raise TypeError("diagnostic data keys must be strings")
            if not isinstance(value, str | int | bool):
                raise TypeError(
                    "diagnostic data values must be str, int, or bool entries"
                )
            normalized_data[key] = value
        return normalized_data

    def __init__(
        self,
        *,
        code: str | ErrorCode,
        message: str,
        help: str | None = None,
        related: tuple[str, ...] = (),
        data: dict[str, DiagnosticValue] | None = None,
    ) -> None:
        """Build one structured TensorOp error."""
        normalized_code = self._normalize_code(code)
        if not isinstance(message, str):
            raise TypeError("diagnostic message must be a string")
        if not message.strip():
            raise ValueError("diagnostic message cannot be empty")
        if help is not None and not isinstance(help, str):
            raise TypeError("diagnostic help must be a string or None")

        payload_data = {} if data is None else data
        self.code = normalized_code
        self.external_code = normalized_code.upper()
        self.severity = "error"
        self.help = help
        self.related = self._normalize_related(related)
        self.data = self._normalize_data(payload_data)
        self.message = message
        super().__init__(message)


class ValidationError(TensorOpError):
    """Structured validation-phase error."""

    channel = "validation_error"


class ExecutionError(TensorOpError):
    """Structured execution-phase error."""

    channel = "execution_error"


__all__ = ["ErrorCode", "TensorOpError", "ValidationError", "ExecutionError"]
