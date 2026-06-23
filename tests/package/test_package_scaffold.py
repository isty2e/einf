import pytest

from einf import Signature, ax, axes, contract, packs, reduce, symbols, view
from einf.reduction.schema import ReducerPhase


def test_axes_creates_named_axes() -> None:
    b, h, w = axes("b", "h", "w")
    assert b.name == "b"
    assert h.name == "h"
    assert w.name == "w"


def test_axes_rejects_duplicates() -> None:
    with pytest.raises(ValueError):
        axes("b", "b")


def test_axes_rejects_whitespace_delimited_arguments() -> None:
    with pytest.raises(ValueError):
        axes("b h")


def test_packs_create_named_axis_packs() -> None:
    t1, t2 = packs("T1", "T2")
    assert t1.name == "T1"
    assert t2.name == "T2"


def test_symbols_creates_axes_and_packs_from_tuple_inputs() -> None:
    axis_group, pack_group = symbols(axes=("b", "h", "w"), packs=("T1", "T2"))
    assert tuple(axis.name for axis in axis_group) == ("b", "h", "w")
    assert tuple(pack.name for pack in pack_group) == ("T1", "T2")


def test_symbols_rejects_overlap_between_axes_and_packs() -> None:
    with pytest.raises(ValueError):
        symbols(axes=("b", "T1"), packs=("T1",))


def test_symbols_rejects_non_tuple_inputs() -> None:
    with pytest.raises(TypeError):
        symbols(axes=["b", "h"], packs=("T1",))  # type: ignore[arg-type]


def test_view_factory_sets_expected_arity_from_lhs_rhs() -> None:
    b, c = axes("b", "c")
    op = view(ax[b, c], ax[b, c])
    assert op.signature.input_arity == 1
    assert op.signature.output_arity == 1
    assert len(op.lhs) == 1
    assert len(op.rhs) == 1


def test_reduce_supports_custom_reducer() -> None:
    b, c = axes("b", "c")
    op = reduce(ax[b, c], ax[b])
    configured = op.reduce_by("sum")
    assert configured.reducer_plan == (ReducerPhase(axes=(c,), reducer="sum"),)


def test_contract_lhs_rhs_tuple_defines_multi_input_arity() -> None:
    i, k, j = axes("i", "k", "j")
    op = contract((ax[i, k], ax[k, j]), ax[i, j])
    assert op.signature.input_arity == 2
    assert op.signature.output_arity == 1


def test_typed_signature_dsl_builds_multi_input_signature() -> None:
    i, k, j = axes("i", "k", "j")
    sig = Signature(inputs=(ax[i, k], ax[k, j]), outputs=(ax[i, j],))
    assert sig.input_arity == 2
    assert sig.output_arity == 1


def test_typed_signature_dsl_builds_multi_output_signature() -> None:
    b, h1, h2, c = axes("b", "h1", "h2", "c")
    sig = Signature(inputs=(ax[b, (h1 + h2), c],), outputs=(ax[b, h1, c], ax[b, h2, c]))
    assert sig.input_arity == 1
    assert sig.output_arity == 2


def test_typed_signature_dsl_rejects_invalid_output_tuple() -> None:
    b, c = axes("b", "c")
    with pytest.raises(TypeError):
        _ = getattr(Signature, "__call__")(inputs=(ax[b, c],), outputs=(ax[b], 3))


def test_typed_signature_dsl_supports_axis_pack_terms() -> None:
    t1, t2 = packs("T1", "T2")
    sig = Signature(inputs=(ax[t1, 3, t2],), outputs=(ax[t2, 3, t1],))
    assert sig.inputs[0][0] == t1
    assert sig.outputs[0][-1] == t1
