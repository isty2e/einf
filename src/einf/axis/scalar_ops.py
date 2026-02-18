from .base import ScalarAxisTermBase
from .collections import ScalarAxisTerms
from .terms import Axis, AxisExpr, AxisInt


def term_size(term: ScalarAxisTermBase, axis_sizes: dict[str, int]) -> int:
    """Evaluate one scalar term size under solved axis sizes."""
    if isinstance(term, AxisInt):
        return term.value
    if isinstance(term, Axis):
        return axis_sizes[term.name]
    if isinstance(term, AxisExpr):
        left = term_size(term.left, axis_sizes)
        right = term_size(term.right, axis_sizes)
        if term.operator == "+":
            return left + right
        return left * right
    raise TypeError("unsupported scalar term")


def first_add_index(terms: ScalarAxisTerms) -> int | None:
    """Return first concat-term index in one term list, if any."""
    for index, term in enumerate(terms):
        if isinstance(term, AxisExpr) and term.operator == "+":
            return index
    return None


def flatten_add_children(term: ScalarAxisTermBase) -> ScalarAxisTerms:
    """Flatten one nested `+` expression into ordered child terms."""
    if isinstance(term, AxisExpr) and term.operator == "+":
        left = flatten_add_children(term.left)
        right = flatten_add_children(term.right)
        return ScalarAxisTerms(tuple((*left, *right)))
    return ScalarAxisTerms((term,))


def split_add_children(term: ScalarAxisTermBase) -> ScalarAxisTerms:
    """Return immediate children of one `+` expression."""
    if isinstance(term, AxisExpr) and term.operator == "+":
        return ScalarAxisTerms((term.left, term.right))
    return ScalarAxisTerms((term,))


def flatten_mul_children(term: ScalarAxisTermBase) -> ScalarAxisTerms:
    """Flatten one nested `*` expression into ordered child terms."""
    if isinstance(term, AxisExpr) and term.operator == "*":
        left = flatten_mul_children(term.left)
        right = flatten_mul_children(term.right)
        return ScalarAxisTerms(tuple((*left, *right)))
    return ScalarAxisTerms((term,))


def expand_products_for_terms(terms: ScalarAxisTerms) -> ScalarAxisTerms:
    """Expand top-level product terms to scalar-factor lists."""
    expanded: list[ScalarAxisTermBase] = []
    for term in terms:
        if isinstance(term, AxisExpr) and term.operator == "*":
            for child in flatten_mul_children(term):
                expanded.append(child)
            continue
        expanded.append(term)
    return ScalarAxisTerms(tuple(expanded))
