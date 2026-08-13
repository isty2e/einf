from dataclasses import dataclass
from enum import Enum
from typing import TypeAlias, final, get_args

from einf.axis import AxisTerms
from einf.einop_layout import EinopLayoutNormalization


class EinopPrimitiveRoute(str, Enum):
    """Closed primitive routes available to einop lowering."""

    ROUTE = "route"
    REARRANGE = "rearrange"
    REPEAT = "repeat"
    REDUCE = "reduce"
    REDUCE_REPEAT = "reduce_repeat"
    CONTRACT = "contract"


def _validate_equations(equations: tuple[str, ...], *, owner: str) -> None:
    if type(equations) is not tuple:
        raise TypeError(f"{owner} equations must be a tuple")
    if not equations:
        raise ValueError(f"{owner} requires at least one equation")
    if any(type(equation) is not str or not equation for equation in equations):
        raise TypeError(f"{owner} equations must be non-empty strings")


@dataclass(frozen=True, slots=True)
@final
class PrimitiveEinopLoweringPlan:
    """Lower one einop through a primitive route.

    Parameters
    ----------
    route : EinopPrimitiveRoute
        Primitive route selected for symbolic lowering.

    Raises
    ------
    TypeError
        If ``route`` is not an :class:`EinopPrimitiveRoute`.
    """

    route: EinopPrimitiveRoute

    def __post_init__(self) -> None:
        if not isinstance(self.route, EinopPrimitiveRoute):
            raise TypeError("primitive einop lowering requires a primitive route")

    @property
    def symbolic_kind(self) -> str:
        """Return the primitive route name.

        Returns
        -------
        str
            Primitive route value.
        """
        return self.route.value


@dataclass(frozen=True, slots=True)
@final
class DirectEinsumEinopLoweringPlan:
    """Lower one einop through independent einsum equations.

    Parameters
    ----------
    equations : tuple[str, ...]
        Non-empty equations, one for each symbolic output.

    Raises
    ------
    TypeError
        If ``equations`` is not a tuple of non-empty strings.
    ValueError
        If ``equations`` is empty.
    """

    equations: tuple[str, ...]

    def __post_init__(self) -> None:
        _validate_equations(self.equations, owner="direct einsum einop lowering")

    @property
    def symbolic_kind(self) -> str:
        """Return the direct einsum symbolic kind.

        Returns
        -------
        str
            ``"einsum"``.
        """
        return "einsum"


@dataclass(frozen=True, slots=True)
@final
class LayoutNormalizedEinopLoweringPlan:
    """Lower one einop through a normalized logical layout.

    Parameters
    ----------
    normalization : EinopLayoutNormalization
        Required mapping between requested and logical signatures.

    Raises
    ------
    TypeError
        If ``normalization`` is not an
        :class:`~einf.einop_layout.EinopLayoutNormalization`.
    ValueError
        If the supplied normalization does not change the signature.
    """

    normalization: EinopLayoutNormalization

    def __post_init__(self) -> None:
        if type(self.normalization) is not EinopLayoutNormalization:
            raise TypeError(
                "layout-normalized einop lowering requires layout normalization"
            )
        if not self.normalization.is_required:
            raise ValueError(
                "layout-normalized einop lowering requires a layout change"
            )

    @property
    def symbolic_kind(self) -> str:
        """Return the layout-normalized symbolic kind.

        Returns
        -------
        str
            ``"layout_normalized"``.
        """
        return "layout_normalized"


@dataclass(frozen=True, slots=True)
@final
class CarrierEinopLoweringPlan:
    """Lower inputs to one carrier before a canonical unary tail.

    Parameters
    ----------
    equation : str
        Equation that materializes the carrier tensor.
    intermediate : AxisTerms
        Logical axis terms of the carrier tensor.
    tail : EinopLoweringPlan
        Canonical unary plan consuming the carrier.

    Raises
    ------
    TypeError
        If a field has the wrong type.
    """

    equation: str
    intermediate: AxisTerms
    tail: "EinopLoweringPlan"

    def __post_init__(self) -> None:
        if type(self.equation) is not str or not self.equation:
            raise TypeError("carrier einop lowering requires a non-empty equation")
        if not isinstance(self.intermediate, AxisTerms):
            raise TypeError("carrier einop lowering requires intermediate axis terms")
        if not _is_einop_lowering_plan(self.tail):
            raise TypeError("carrier einop lowering requires an executable tail plan")

    @property
    def symbolic_kind(self) -> str:
        """Return the carrier composite symbolic kind.

        Returns
        -------
        str
            ``"einsum_carrier_then_unary"``.
        """
        return "einsum_carrier_then_unary"


@dataclass(frozen=True, slots=True)
@final
class ChainEinopLoweringPlan:
    """Lower all inputs through an ordered carrier chain and unary tail.

    Parameters
    ----------
    equations : tuple[str, ...]
        Binary equations applied along the carrier chain.
    intermediate : AxisTerms
        Logical axis terms produced by the final chain edge.
    carrier_index : int
        Input index used as the initial carrier.
    chain_order : tuple[int, ...]
        Remaining input indices in execution order.
    tail : EinopLoweringPlan
        Canonical unary plan consuming the final carrier.

    Raises
    ------
    TypeError
        If a field has the wrong type.
    ValueError
        If there is not one equation per edge, or the carrier and chain order
        do not consume every input exactly once.
    """

    equations: tuple[str, ...]
    intermediate: AxisTerms
    carrier_index: int
    chain_order: tuple[int, ...]
    tail: "EinopLoweringPlan"

    def __post_init__(self) -> None:
        _validate_equations(self.equations, owner="chain einop lowering")
        if not isinstance(self.intermediate, AxisTerms):
            raise TypeError("chain einop lowering requires intermediate axis terms")
        if type(self.carrier_index) is not int:
            raise TypeError("chain einop lowering carrier index must be an integer")
        if type(self.chain_order) is not tuple:
            raise TypeError("chain einop lowering order must be a tuple")
        if any(type(index) is not int for index in self.chain_order):
            raise TypeError("chain einop lowering order must contain integer indices")
        if not _is_einop_lowering_plan(self.tail):
            raise TypeError("chain einop lowering requires an executable tail plan")
        if len(self.chain_order) != len(self.equations):
            raise ValueError("chain einop lowering requires one equation per edge")

        input_arity = len(self.chain_order) + 1
        if self.carrier_index < 0 or self.carrier_index >= input_arity:
            raise ValueError("chain einop lowering carrier index is out of range")
        expected_order = set(range(input_arity)) - {self.carrier_index}
        if len(set(self.chain_order)) != len(self.chain_order):
            raise ValueError("chain einop lowering order contains duplicate inputs")
        if set(self.chain_order) != expected_order:
            raise ValueError(
                "chain einop lowering must consume every non-carrier input"
            )

    @property
    def symbolic_kind(self) -> str:
        """Return the chain composite symbolic kind.

        Returns
        -------
        str
            ``"einsum_chain_then_unary"``.
        """
        return "einsum_chain_then_unary"


@dataclass(frozen=True, slots=True)
@final
class EinopChainSearchRequest:
    """Signal that base lowering requires exhaustive carrier-chain search."""


EinopLoweringPlan: TypeAlias = (
    PrimitiveEinopLoweringPlan
    | DirectEinsumEinopLoweringPlan
    | LayoutNormalizedEinopLoweringPlan
    | CarrierEinopLoweringPlan
    | ChainEinopLoweringPlan
)


def _is_einop_lowering_plan(candidate: EinopLoweringPlan) -> bool:
    return type(candidate) in get_args(EinopLoweringPlan)


__all__ = [
    "CarrierEinopLoweringPlan",
    "ChainEinopLoweringPlan",
    "DirectEinsumEinopLoweringPlan",
    "EinopChainSearchRequest",
    "EinopLoweringPlan",
    "EinopPrimitiveRoute",
    "LayoutNormalizedEinopLoweringPlan",
    "PrimitiveEinopLoweringPlan",
]
