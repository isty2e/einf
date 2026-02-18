from dataclasses import dataclass

import numpy as np
import pytest

from einf import (
    ErrorCode,
    ValidationError,
    ax,
    axes,
    contract,
    einop,
    rearrange,
    reduce,
    repeat,
    view,
)


class _NoEinsumNamespace:
    __name__ = "fake.noeinsum"


@dataclass(frozen=True, slots=True)
class _NoEinsumTensor:
    shape: tuple[int, ...]

    def __array_namespace__(
        self, api_version: str | None = None
    ) -> type[_NoEinsumNamespace]:
        _ = api_version
        return _NoEinsumNamespace

    def __getitem__(self, key: object) -> "_NoEinsumTensor":
        _ = key
        return self


def _assert_validation_error(
    *,
    error: ValidationError,
    code: ErrorCode,
    message_contains: tuple[str, ...],
    help_contains: tuple[str, ...],
    related_contains: tuple[str, ...],
    data_keys: tuple[str, ...] = (),
) -> None:
    """Assert one emitted validation error against expected conformance tokens."""
    assert error.code == code.value

    normalized_message = error.message.lower()
    normalized_help = (error.help or "").lower()
    normalized_related = " ".join(error.related).lower()

    for token in message_contains:
        assert token in normalized_message
    for token in help_contains:
        assert token in normalized_help
    for token in related_contains:
        assert token in normalized_related

    observed_data_keys = tuple(sorted(error.data.keys())) if error.data else ()
    assert observed_data_keys == tuple(sorted(data_keys))


def test_axis_expr_precedence_parentheses_then_multiply_then_plus() -> None:
    h, w, r, c = axes("h", "w", "r", "c")
    _ = rearrange(ax[(h + (w * r)), c], ax[(h + (w * r)), c]).with_sizes(
        h=2,
        w=3,
        r=4,
        c=1,
    )


def test_reduce_non_unique_dim_solve_allows_h_times_w_to_one_when_output_deterministic() -> (
    None
):
    h, w = axes("h", "w")
    op = reduce(ax[(h * w)], ax[1])

    tensor = np.arange(6, dtype=np.float64)
    result = op(tensor)
    np.testing.assert_array_equal(result, np.array([tensor.sum()]))


def test_inflate_allows_zero_literal_factor() -> None:
    b, c = axes("b", "c")
    op = repeat(ax[b, c], ax[b, c, 0])

    tensor = np.arange(6).reshape(2, 3)
    result = op(tensor)

    assert result.shape == (2, 3, 0)


def test_inflate_literal_and_symbolic_factor_are_equivalent() -> None:
    b, c, r = axes("b", "c", "r")
    literal = repeat(ax[b, c], ax[b, c, 3])
    symbolic = repeat(ax[b, c], ax[b, c, r]).with_sizes(r=3)

    tensor = np.arange(6).reshape(2, 3)
    literal_result = literal(tensor)
    symbolic_result = symbolic(tensor)

    np.testing.assert_array_equal(literal_result, symbolic_result)


def test_einop_reduce_by_requires_unary_signature() -> None:
    i, k, j = axes("i", "k", "j")
    op = einop((ax[i, k], ax[k, j]), ax[i, j])

    with pytest.raises(ValidationError) as error:
        _ = op.reduce_by(lambda x: x)

    _assert_validation_error(
        error=error.value,
        code=ErrorCode.OP_ARITY_MISMATCH,
        message_contains=("reduce_by", "unary"),
        help_contains=("unary",),
        related_contains=("reduce_by contract",),
        data_keys=("lhs_arity", "rhs_arity"),
    )


def test_view_requires_unary_lhs_signature_in_v01() -> None:
    (b,) = axes("b")

    with pytest.raises(ValidationError) as error:
        _ = getattr(view, "__call__")((ax[b], ax[b]), ax[b])

    _assert_validation_error(
        error=error.value,
        code=ErrorCode.MULTI_INPUT_NOT_ALLOWED,
        message_contains=("multi-input", "view"),
        help_contains=("unary lhs",),
        related_contains=("view schema",),
        data_keys=("expected", "got", "operation"),
    )


def test_contract_requires_einsum_capable_backend_extension() -> None:
    (i,) = axes("i")
    op = contract((ax[i], ax[i]), ax[i])

    with pytest.raises(ValidationError) as error:
        _ = op(_NoEinsumTensor(shape=(3,)), _NoEinsumTensor(shape=(3,)))

    _assert_validation_error(
        error=error.value,
        code=ErrorCode.BACKEND_REQUIRED_EXTENSION_MISSING,
        message_contains=("backend required extension missing", "contract"),
        help_contains=("einsum-capable backend family",),
        related_contains=("backend capability",),
        data_keys=("operation",),
    )
