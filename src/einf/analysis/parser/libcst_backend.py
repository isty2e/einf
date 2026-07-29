import importlib
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

from einf.analysis.model import TextPosition, TextSpan

from .base import (
    ParsedModule,
    ParsedNode,
    ParserSyntaxError,
    ParserUnavailableError,
    TextEdit,
)


@dataclass(frozen=True, slots=True)
class _LibCstDependencies:
    module: ModuleType
    metadata_wrapper_type: type
    position_provider_type: type
    syntax_error_type: type[BaseException]


def _load_libcst_dependencies() -> _LibCstDependencies:
    """Load libcst and metadata providers on demand."""
    try:
        libcst = importlib.import_module("libcst")
        libcst_metadata = importlib.import_module("libcst.metadata")
    except ModuleNotFoundError as error:
        raise ParserUnavailableError(
            backend="libcst",
            message=(
                "libcst parser backend requires libcst; "
                "install einf[analysis] to enable it"
            ),
        ) from error

    metadata_wrapper = getattr(libcst_metadata, "MetadataWrapper", None)
    position_provider = getattr(libcst_metadata, "PositionProvider", None)
    parse_module = getattr(libcst, "parse_module", None)
    syntax_error = getattr(libcst, "ParserSyntaxError", None)
    if (
        not isinstance(metadata_wrapper, type)
        or not isinstance(position_provider, type)
        or not callable(parse_module)
        or not (
            isinstance(syntax_error, type) and issubclass(syntax_error, BaseException)
        )
    ):
        raise ParserUnavailableError(
            backend="libcst",
            message="libcst parser backend dependencies are incomplete",
        )

    return _LibCstDependencies(
        module=libcst,
        metadata_wrapper_type=metadata_wrapper,
        position_provider_type=position_provider,
        syntax_error_type=syntax_error,
    )


def _normalize_syntax_error(error: BaseException) -> ParserSyntaxError:
    message = getattr(error, "message", None)
    normalized_message = message if isinstance(message, str) else str(error)
    line = getattr(error, "raw_line", None)
    column = getattr(error, "raw_column", None)
    if type(line) is not int or type(column) is not int or line < 1 or column < 0:
        return ParserSyntaxError(message=normalized_message, span=None)

    return ParserSyntaxError(
        message=normalized_message,
        span=TextSpan(
            start=TextPosition(line=line, column=column),
            end=TextPosition(line=line, column=column + 1),
        ),
    )


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

    def validate_available(self) -> None:
        """Confirm that the optional LibCST parser dependencies are available."""
        _load_libcst_dependencies()

    def parse(self, source: str, path: Path) -> ParsedModule:
        """Parse source text using libcst and normalize tree shape."""
        dependencies = _load_libcst_dependencies()
        try:
            module_node = dependencies.module.parse_module(source)
        except dependencies.syntax_error_type as error:
            raise _normalize_syntax_error(error) from error

        wrapper = dependencies.metadata_wrapper_type(module_node)
        positions = wrapper.resolve(dependencies.position_provider_type)

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
