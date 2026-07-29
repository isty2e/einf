from pathlib import Path

import pytest

from einf.analysis.model import TextPosition, TextSpan
from einf.analysis.parser import (
    AstParserBackend,
    ParsedModule,
    ParsedNode,
    ParserSyntaxError,
    TextEdit,
)


def test_ast_backend_parse_graph_invariants() -> None:
    backend = AstParserBackend()
    module = backend.parse(source="x = 1\nprint(x)\n", path=Path("sample.py"))
    root = module.root()
    children = module.children(root)

    assert root.kind == "Module"
    assert root.node_id == module.root_id
    assert tuple(child.node_id for child in children) == root.child_ids
    assert all(module.node(child.node_id) == child for child in children)


def test_parsed_module_rejects_invalid_initialization() -> None:
    root = ParsedNode(
        node_id=0,
        kind="Module",
        span=None,
        value=None,
        child_ids=(),
    )
    with pytest.raises(ValueError, match="at least one node"):
        ParsedModule(
            path=Path("sample.py"),
            source="",
            nodes=(),
            root_id=0,
        )

    with pytest.raises(ValueError, match="root_id is out of bounds"):
        ParsedModule(
            path=Path("sample.py"),
            source="",
            nodes=(root,),
            root_id=1,
        )

    with pytest.raises(ValueError, match="node ids must be contiguous"):
        ParsedModule(
            path=Path("sample.py"),
            source="",
            nodes=(
                ParsedNode(
                    node_id=1,
                    kind="Module",
                    span=None,
                    value=None,
                    child_ids=(),
                ),
            ),
            root_id=0,
        )

    with pytest.raises(ValueError, match="child id 1 is out of bounds"):
        ParsedModule(
            path=Path("sample.py"),
            source="",
            nodes=(
                ParsedNode(
                    node_id=0,
                    kind="Module",
                    span=None,
                    value=None,
                    child_ids=(1,),
                ),
            ),
            root_id=0,
        )


def test_parsed_module_node_rejects_out_of_bounds() -> None:
    module = ParsedModule(
        path=Path("sample.py"),
        source="",
        nodes=(
            ParsedNode(
                node_id=0,
                kind="Module",
                span=None,
                value=None,
                child_ids=(),
            ),
        ),
        root_id=0,
    )
    with pytest.raises(ValueError, match="out of bounds"):
        module.node(-1)
    with pytest.raises(ValueError, match="out of bounds"):
        module.node(1)


def test_ast_backend_extracts_values_from_names_and_constants() -> None:
    backend = AstParserBackend()
    source = "name = 'txt'\na = True\nb = 7\nc = 3.5\nd = call()\n"
    module = backend.parse(source=source, path=Path("sample.py"))

    name_values = [node.value for node in module.nodes if node.kind == "Name"]
    constant_values = [node.value for node in module.nodes if node.kind == "Constant"]
    call_values = [node.value for node in module.nodes if node.kind == "Call"]

    assert "name" in name_values
    assert "a" in name_values
    assert "b" in name_values
    assert "c" in name_values
    assert "txt" in constant_values
    assert "True" in constant_values
    assert "7" in constant_values
    assert "3.5" in constant_values
    assert call_values == [None]


def test_ast_backend_records_multiline_call_spans() -> None:
    backend = AstParserBackend()
    source = "f(\n    x,\n)\n"
    module = backend.parse(source=source, path=Path("sample.py"))
    call_nodes = [node for node in module.nodes if node.kind == "Call"]
    assert len(call_nodes) == 1
    call_span = call_nodes[0].span
    assert call_span is not None
    assert call_span.start.line == 1
    assert call_span.end.line == 3


def test_ast_backend_reparse_uses_new_source_and_keeps_path() -> None:
    backend = AstParserBackend()
    path = Path("sample.py")
    previous = backend.parse(source="x = 1\n", path=path)
    reparsed = backend.reparse(
        previous=previous,
        edits=(
            TextEdit(
                span=TextSpan(
                    start=TextPosition(line=1, column=0),
                    end=TextPosition(line=1, column=5),
                ),
                replacement="x = 2\ny = 3\n",
            ),
        ),
        new_source="x = 2\ny = 3\n",
    )

    assert reparsed.path == previous.path
    assert reparsed.source == "x = 2\ny = 3\n"
    assert reparsed.nodes is not previous.nodes
    assert reparsed.root_id == 0
    assert len(reparsed.nodes) > len(previous.nodes)


def test_ast_backend_root_module_span_is_none() -> None:
    backend = AstParserBackend()
    module = backend.parse(source="x = 1\n", path=Path("sample.py"))
    assert module.root().kind == "Module"
    assert module.root().span is None


def test_ast_backend_parse_invalid_source_raises_canonical_syntax_error() -> None:
    backend = AstParserBackend()
    with pytest.raises(ParserSyntaxError) as error_info:
        backend.parse(source="x =\n", path=Path("sample.py"))
    assert error_info.value.span is not None


def test_ast_backend_reparse_preserves_graph_navigation_contract() -> None:
    backend = AstParserBackend()
    previous = backend.parse(source="x = 1\n", path=Path("sample.py"))
    reparsed = backend.reparse(
        previous=previous,
        edits=(),
        new_source="x = 2\nprint(x)\n",
    )

    root = reparsed.root()
    children = reparsed.children(root)
    assert tuple(child.node_id for child in children) == root.child_ids
    assert all(reparsed.node(child.node_id) == child for child in children)


def test_ast_backend_multiline_call_span_has_expected_column_order() -> None:
    backend = AstParserBackend()
    module = backend.parse(source="fn(\n  x,\n)\n", path=Path("sample.py"))
    call_node = next(node for node in module.nodes if node.kind == "Call")
    assert call_node.span is not None
    assert call_node.span.start.column == 0
    assert call_node.span.end.column >= 1


def test_ast_backend_parse_empty_source_has_root_without_children() -> None:
    backend = AstParserBackend()
    module = backend.parse(source="", path=Path("sample.py"))
    assert module.root().kind == "Module"
    assert module.root().child_ids == ()
    assert module.children(module.root()) == ()


def test_ast_backend_does_not_extract_bytes_constant_value() -> None:
    backend = AstParserBackend()
    module = backend.parse(source="x = b'abc'\n", path=Path("sample.py"))
    byte_constant_nodes = [
        node for node in module.nodes if node.kind == "Constant" and node.value is None
    ]
    assert len(byte_constant_nodes) >= 1


def test_ast_backend_negative_float_constant_keeps_positive_literal_node_value() -> (
    None
):
    backend = AstParserBackend()
    module = backend.parse(source="x = -0.5\n", path=Path("sample.py"))
    constant_values = [node.value for node in module.nodes if node.kind == "Constant"]
    assert "0.5" in constant_values


def test_ast_backend_reparse_invalid_source_raises_canonical_syntax_error() -> None:
    backend = AstParserBackend()
    previous = backend.parse(source="x = 1\n", path=Path("sample.py"))
    with pytest.raises(ParserSyntaxError) as error_info:
        backend.reparse(
            previous=previous,
            edits=(),
            new_source="x =\n",
        )
    assert error_info.value.span is not None


def test_ast_backend_constant_leaf_node_has_no_children() -> None:
    backend = AstParserBackend()
    module = backend.parse(source="x = 1\n", path=Path("sample.py"))
    constant_node = next(node for node in module.nodes if node.kind == "Constant")
    assert module.children(constant_node) == ()


def test_ast_backend_node_spans_are_non_inverted() -> None:
    backend = AstParserBackend()
    module = backend.parse(
        source="f(\n    x,\n)\nvalue = 3\n",
        path=Path("sample.py"),
    )

    for node in module.nodes:
        span = node.span
        if span is None:
            continue
        starts_after_end = span.start.line > span.end.line or (
            span.start.line == span.end.line and span.start.column > span.end.column
        )
        assert not starts_after_end


def test_parsed_module_children_use_constructor_validated_graph() -> None:
    module = ParsedModule(
        path=Path("sample.py"),
        source="x = 1\n",
        nodes=(
            ParsedNode(
                node_id=0,
                kind="Module",
                span=None,
                value=None,
                child_ids=(1,),
            ),
            ParsedNode(
                node_id=1,
                kind="Constant",
                span=None,
                value="1",
                child_ids=(),
            ),
        ),
        root_id=0,
    )

    assert module.children(module.root()) == (module.node(1),)


def test_ast_backend_node_ids_are_contiguous() -> None:
    backend = AstParserBackend()
    module = backend.parse(source="x = 1\nprint(x)\n", path=Path("sample.py"))
    assert tuple(node.node_id for node in module.nodes) == tuple(
        range(len(module.nodes))
    )


def test_ast_backend_child_ids_reference_existing_nodes() -> None:
    backend = AstParserBackend()
    module = backend.parse(source="x = 1\nprint(x)\n", path=Path("sample.py"))
    valid_ids = set(range(len(module.nodes)))
    for node in module.nodes:
        assert set(node.child_ids).issubset(valid_ids)


def test_ast_backend_negative_integer_constant_node_keeps_positive_literal_value() -> (
    None
):
    backend = AstParserBackend()
    module = backend.parse(source="x = -7\n", path=Path("sample.py"))
    constant_values = [node.value for node in module.nodes if node.kind == "Constant"]
    assert "7" in constant_values


def test_ast_backend_extracts_true_and_false_constant_values() -> None:
    backend = AstParserBackend()
    module = backend.parse(source="a = True\nb = False\n", path=Path("sample.py"))
    constant_values = [node.value for node in module.nodes if node.kind == "Constant"]
    assert "True" in constant_values
    assert "False" in constant_values


def test_ast_backend_parse_preserves_given_path() -> None:
    backend = AstParserBackend()
    target_path = Path("pkg/sample.py")
    module = backend.parse(source="x = 1\n", path=target_path)
    assert module.path == target_path


def test_ast_backend_call_span_for_attribute_chain_is_ordered() -> None:
    backend = AstParserBackend()
    module = backend.parse(source="pkg.sub.fn(x)\n", path=Path("sample.py"))
    call_node = next(node for node in module.nodes if node.kind == "Call")
    assert call_node.span is not None
    assert call_node.span.start.line == 1
    assert call_node.span.end.line == 1
    assert call_node.span.start.column < call_node.span.end.column


def test_ast_backend_unicode_prefix_uses_character_columns() -> None:
    backend = AstParserBackend()
    source = "é = 1; print(é)\n"
    module = backend.parse(source=source, path=Path("sample.py"))
    call_node = next(node for node in module.nodes if node.kind == "Call")
    assert call_node.span is not None
    line = source.splitlines(keepends=True)[0]
    assert line[call_node.span.start.column : call_node.span.end.column] == "print(é)"
