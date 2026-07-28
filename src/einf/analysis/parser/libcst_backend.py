import importlib
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

from einf.analysis.model import TextPosition, TextSpan

from .base import ParsedModule, ParsedNode, TextEdit


def _load_libcst_dependencies() -> tuple[ModuleType, type, type]:
    """Load libcst and metadata providers on demand."""
    try:
        libcst = importlib.import_module("libcst")
        libcst_metadata = importlib.import_module("libcst.metadata")
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "libcst parser backend requires libcst; install einf[analysis] to enable it"
        ) from error

    metadata_wrapper = getattr(libcst_metadata, "MetadataWrapper", None)
    position_provider = getattr(libcst_metadata, "PositionProvider", None)
    if not isinstance(metadata_wrapper, type) or not isinstance(
        position_provider, type
    ):
        raise RuntimeError(
            "libcst parser backend requires MetadataWrapper and PositionProvider"
        )

    return libcst, metadata_wrapper, position_provider


def _to_text_span(code_range) -> TextSpan | None:
    """Convert libcst CodeRange-like objects to canonical TextSpan."""
    if code_range is None:
        return None

    start = getattr(code_range, "start", None)
    end = getattr(code_range, "end", None)
    if start is None or end is None:
        return None

    start_line = getattr(start, "line", None)
    start_column = getattr(start, "column", None)
    end_line = getattr(end, "line", None)
    end_column = getattr(end, "column", None)
    if (
        type(start_line) is not int
        or type(start_column) is not int
        or type(end_line) is not int
        or type(end_column) is not int
    ):
        return None

    return TextSpan(
        start=TextPosition(line=start_line, column=start_column),
        end=TextPosition(line=end_line, column=end_column),
    )


def _node_value(cst_node) -> str | None:
    """Extract stable string value from value-bearing libcst nodes."""
    value = getattr(cst_node, "value", None)
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    return None


@dataclass(frozen=True, slots=True)
class LibCstParserBackend:
    """Parser backend that lowers libcst trees into canonical ParsedModule."""

    name: str = "libcst"

    def parse(self, source: str, path: Path) -> ParsedModule:
        """Parse source text using libcst and normalize tree shape."""
        libcst, metadata_wrapper_type, position_provider_type = (
            _load_libcst_dependencies()
        )
        module_node = libcst.parse_module(source)
        wrapper = metadata_wrapper_type(module_node)
        positions = wrapper.resolve(position_provider_type)

        nodes: list[ParsedNode] = []

        def visit(cst_node) -> int:
            node_id = len(nodes)
            nodes.append(
                ParsedNode(
                    node_id=node_id,
                    kind="pending",
                    span=None,
                    value=None,
                    child_ids=(),
                )
            )

            child_values = getattr(cst_node, "children", ())
            child_ids = tuple(visit(child_node) for child_node in child_values)
            nodes[node_id] = ParsedNode(
                node_id=node_id,
                kind=type(cst_node).__name__,
                span=_to_text_span(positions.get(cst_node)),
                value=_node_value(cst_node),
                child_ids=child_ids,
            )
            return node_id

        root_id = visit(wrapper.module)
        return ParsedModule(
            path=path,
            source=source,
            nodes=tuple(nodes),
            root_id=root_id,
        )

    def reparse(
        self,
        previous: ParsedModule,
        edits: tuple[TextEdit, ...],
        new_source: str,
    ) -> ParsedModule:
        """Reparse by falling back to full parse for now."""
        _ = edits
        return self.parse(source=new_source, path=previous.path)
