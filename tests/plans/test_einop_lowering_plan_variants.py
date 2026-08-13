from dataclasses import fields

import pytest

from einf import ax, axes
from einf.einop_layout import EinopLayoutNormalization
from einf.lowering.einop import (
    CarrierEinopLoweringPlan,
    ChainEinopLoweringPlan,
    DirectEinsumEinopLoweringPlan,
    EinopChainSearchRequest,
    EinopLoweringPlan,
    EinopPrimitiveRoute,
    LayoutNormalizedEinopLoweringPlan,
    PrimitiveEinopLoweringPlan,
    build_einop_execution_plan,
)
from einf.lowering.einop.search_plan import build_symbolic_einsum_chain_plan
from einf.signature import Signature


def test_einop_lowering_variants_expose_only_owned_facts() -> None:
    a, b, c = axes("a", "b", "c")
    primitive = PrimitiveEinopLoweringPlan(route=EinopPrimitiveRoute.REARRANGE)
    direct = DirectEinsumEinopLoweringPlan(equations=("ab,bc->ac",))
    normalization = EinopLayoutNormalization.from_signature(
        Signature(inputs=(ax[a * b],), outputs=(ax[b * a],))
    )
    layout = LayoutNormalizedEinopLoweringPlan(normalization=normalization)
    carrier = CarrierEinopLoweringPlan(
        equation="ab,bc->ac",
        intermediate=ax[a, c],
        tail=primitive,
    )
    chain = ChainEinopLoweringPlan(
        equations=("ab,bc->ac", "ac,cd->ad"),
        intermediate=ax[a, c],
        carrier_index=1,
        chain_order=(2, 0),
        tail=primitive,
    )

    assert primitive.symbolic_kind == "rearrange"
    assert direct.symbolic_kind == "einsum"
    assert layout.symbolic_kind == "layout_normalized"
    assert carrier.symbolic_kind == "einsum_carrier_then_unary"
    assert chain.symbolic_kind == "einsum_chain_then_unary"
    assert tuple(field.name for field in fields(primitive)) == ("route",)
    assert tuple(field.name for field in fields(direct)) == ("equations",)
    assert tuple(field.name for field in fields(layout)) == ("normalization",)
    assert tuple(field.name for field in fields(carrier)) == (
        "equation",
        "intermediate",
        "tail",
    )
    assert tuple(field.name for field in fields(chain)) == (
        "equations",
        "intermediate",
        "carrier_index",
        "chain_order",
        "tail",
    )
    assert all(
        not hasattr(plan, "__dict__")
        for plan in (primitive, direct, layout, carrier, chain)
    )


def test_einop_primitive_routes_are_closed() -> None:
    assert tuple(EinopPrimitiveRoute) == (
        EinopPrimitiveRoute.ROUTE,
        EinopPrimitiveRoute.REARRANGE,
        EinopPrimitiveRoute.REPEAT,
        EinopPrimitiveRoute.REDUCE,
        EinopPrimitiveRoute.REDUCE_REPEAT,
        EinopPrimitiveRoute.CONTRACT,
    )


def test_einop_chain_search_request_is_not_executable() -> None:
    request = EinopChainSearchRequest()

    assert not isinstance(request, EinopLoweringPlan)
    assert fields(request) == ()


def test_direct_einsum_variant_rejects_empty_equations() -> None:
    with pytest.raises(ValueError, match="requires at least one equation"):
        DirectEinsumEinopLoweringPlan(equations=())


def test_unary_multi_output_plan_remains_direct() -> None:
    a, b = axes("a", "b")
    signature = Signature(
        inputs=(ax[a, b],),
        outputs=(ax[a, b], ax[b, a]),
    )

    plan = build_einop_execution_plan(
        analysis_signature=signature,
        has_reducer_plan=False,
    )

    assert plan == DirectEinsumEinopLoweringPlan(equations=("ab->ab", "ab->ba"))


def test_chain_search_does_not_invoke_tail_builder_for_unary_input() -> None:
    a, b = axes("a", "b")
    signature = Signature(
        inputs=(ax[a, b],),
        outputs=(ax[a, b], ax[b, a]),
    )

    def fail_if_called(_signature: Signature) -> EinopLoweringPlan:
        raise AssertionError("unary input must not enter carrier-chain search")

    plan = build_symbolic_einsum_chain_plan(
        analysis_signature=signature,
        tail_builder=fail_if_called,
    )

    assert plan is None


def test_layout_variant_rejects_identity_normalization() -> None:
    (a,) = axes("a")
    signature = Signature(inputs=(ax[a],), outputs=(ax[a],))

    with pytest.raises(ValueError, match="requires a layout change"):
        LayoutNormalizedEinopLoweringPlan(
            normalization=EinopLayoutNormalization.from_signature(signature)
        )


def test_composite_variants_require_unary_executable_tail_plans() -> None:
    a, b = axes("a", "b")

    with pytest.raises(TypeError, match="requires a unary executable tail plan"):
        CarrierEinopLoweringPlan(
            equation="ab->ab",
            intermediate=ax[a, b],
            tail=EinopChainSearchRequest(),  # type: ignore[arg-type]
        )


def test_composite_variants_reject_subclassed_executable_plans() -> None:
    a, b = axes("a", "b")
    extended_plan_type = type(
        "ExtendedPrimitiveEinopLoweringPlan",
        (PrimitiveEinopLoweringPlan,),
        {},
    )
    extended_plan = extended_plan_type(route=EinopPrimitiveRoute.REARRANGE)

    with pytest.raises(TypeError, match="requires a unary executable tail plan"):
        CarrierEinopLoweringPlan(
            equation="ab->ab",
            intermediate=ax[a, b],
            tail=extended_plan,
        )


def test_composite_variants_reject_composite_tail_plans() -> None:
    a, c = axes("a", "c")
    primitive_tail = PrimitiveEinopLoweringPlan(route=EinopPrimitiveRoute.REARRANGE)
    carrier_tail = CarrierEinopLoweringPlan(
        equation="ab,bc->ac",
        intermediate=ax[a, c],
        tail=primitive_tail,
    )
    chain_tail = ChainEinopLoweringPlan(
        equations=("ab,bc->ac",),
        intermediate=ax[a, c],
        carrier_index=0,
        chain_order=(1,),
        tail=primitive_tail,
    )

    for composite_tail in (carrier_tail, chain_tail):
        with pytest.raises(TypeError, match="requires a unary executable tail plan"):
            CarrierEinopLoweringPlan(
                equation="ab,bc->ac",
                intermediate=ax[a, c],
                tail=composite_tail,  # type: ignore[arg-type]
            )

        with pytest.raises(TypeError, match="requires a unary executable tail plan"):
            ChainEinopLoweringPlan(
                equations=("ab,bc->ac",),
                intermediate=ax[a, c],
                carrier_index=0,
                chain_order=(1,),
                tail=composite_tail,  # type: ignore[arg-type]
            )


@pytest.mark.parametrize(
    ("carrier_index", "chain_order", "equations", "message"),
    [
        (0, (1,), ("ab,bc->ac", "ac,cd->ad"), "one equation per edge"),
        (0, (1, 1), ("ab,bc->ac", "ac,cd->ad"), "duplicate inputs"),
        (3, (0, 1), ("ab,bc->ac", "ac,cd->ad"), "out of range"),
        (1, (0, 3), ("ab,bc->ac", "ac,cd->ad"), "every non-carrier input"),
    ],
)
def test_chain_variant_rejects_incomplete_input_partitions(
    carrier_index: int,
    chain_order: tuple[int, ...],
    equations: tuple[str, ...],
    message: str,
) -> None:
    a, c = axes("a", "c")

    with pytest.raises(ValueError, match=message):
        ChainEinopLoweringPlan(
            equations=equations,
            intermediate=ax[a, c],
            carrier_index=carrier_index,
            chain_order=chain_order,
            tail=PrimitiveEinopLoweringPlan(route=EinopPrimitiveRoute.REARRANGE),
        )


def test_carrier_planner_owns_its_canonical_tail() -> None:
    b, h, w, d, j, k = axes("b", "h", "w", "d", "j", "k")
    signature = Signature(
        inputs=(ax[b, h + w, d], ax[d, j], ax[j, k]),
        outputs=(ax[b, h, k], ax[b, w, k]),
    )

    plan = build_einop_execution_plan(
        analysis_signature=signature,
        has_reducer_plan=False,
    )

    assert isinstance(plan, CarrierEinopLoweringPlan)
    assert plan.intermediate == ax[b, h + w, k]
    assert plan.tail == PrimitiveEinopLoweringPlan(route=EinopPrimitiveRoute.REARRANGE)


def test_chain_search_owns_order_carrier_and_canonical_tail() -> None:
    b, h, w, d, j, k = axes("b", "h", "w", "d", "j", "k")
    signature = Signature(
        inputs=(ax[b, h + w, d], ax[d, j], ax[j, k]),
        outputs=(ax[b, h, k], ax[b, w, k]),
    )

    plan = build_symbolic_einsum_chain_plan(
        analysis_signature=signature,
        tail_builder=lambda tail_signature: build_einop_execution_plan(
            analysis_signature=tail_signature,
            has_reducer_plan=False,
        ),
    )

    assert isinstance(plan, ChainEinopLoweringPlan)
    assert plan.carrier_index == 0
    assert plan.chain_order == (1, 2)
    assert plan.intermediate == ax[b, h + w, k]
    assert plan.tail == PrimitiveEinopLoweringPlan(route=EinopPrimitiveRoute.REARRANGE)
