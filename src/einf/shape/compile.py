from einf.axis import Axis, AxisExpr, AxisInt, ScalarAxisTermBase

from .nodes import ShapeAxisName, ShapeBinary, ShapeDimRef, ShapeLiteral, ShapeNode


def compile_shape_node(
    *,
    term: ScalarAxisTermBase,
    axis_index_by_name: dict[str, int],
) -> ShapeNode | None:
    """Compile one scalar axis term into one shape-node expression tree."""
    if isinstance(term, AxisInt):
        return ShapeLiteral(term.value)
    if isinstance(term, Axis):
        axis_index = axis_index_by_name.get(term.name)
        if axis_index is None:
            return ShapeAxisName(term.name)
        return ShapeDimRef(axis_index)
    if not isinstance(term, AxisExpr):
        return None

    left_node = compile_shape_node(
        term=term.left,
        axis_index_by_name=axis_index_by_name,
    )
    if left_node is None:
        return None
    right_node = compile_shape_node(
        term=term.right,
        axis_index_by_name=axis_index_by_name,
    )
    if right_node is None:
        return None
    if term.operator not in {"+", "*"}:
        return None
    return ShapeBinary(term.operator, left_node, right_node)


def shape_node_axis_names(node: ShapeNode) -> set[str]:
    """Collect explicit axis-name dependencies from one compiled shape node."""
    if isinstance(node, ShapeAxisName):
        return {node.name}
    if isinstance(node, ShapeBinary):
        return shape_node_axis_names(node.left) | shape_node_axis_names(node.right)
    return set()


__all__ = [
    "compile_shape_node",
    "shape_node_axis_names",
]
