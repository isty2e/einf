from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from einf.analysis.model import TextSpan


class ParserSyntaxError(Exception):
    """Backend-neutral syntax failure raised while parsing source text."""

    def __init__(self, *, message: str, span: TextSpan | None) -> None:
        self.message = message
        self.span = span
        super().__init__(message)


class ParserUnavailableError(RuntimeError):
    """Parser configuration failure caused by an unavailable backend."""

    def __init__(self, *, backend: str, message: str) -> None:
        self.backend = backend
        self.message = message
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class TextEdit:
    """One incremental text edit."""

    span: TextSpan
    replacement: str


@dataclass(frozen=True, slots=True)
class ParsedNode:
    """Backend-neutral parsed node record."""

    node_id: int
    kind: str
    span: TextSpan | None
    value: str | None
    child_ids: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class ParsedModule:
    """Backend-neutral parsed module graph."""

    path: Path
    source: str
    nodes: tuple[ParsedNode, ...]
    root_id: int

    def __post_init__(self) -> None:
        if not self.nodes:
            raise ValueError("parsed module requires at least one node")
        if self.root_id < 0 or self.root_id >= len(self.nodes):
            raise ValueError("parsed module root_id is out of bounds")
        for expected_node_id, node in enumerate(self.nodes):
            if node.node_id != expected_node_id:
                raise ValueError(
                    "parsed module node ids must be contiguous and match tuple order"
                )
            for child_id in node.child_ids:
                if child_id < 0 or child_id >= len(self.nodes):
                    raise ValueError(
                        f"parsed module child id {child_id} is out of bounds"
                    )

    def node(self, node_id: int) -> ParsedNode:
        """Return one parsed node by id."""
        if node_id < 0 or node_id >= len(self.nodes):
            raise ValueError(f"parsed node id {node_id} is out of bounds")
        return self.nodes[node_id]

    def root(self) -> ParsedNode:
        """Return the root parsed node."""
        return self.node(self.root_id)

    def children(self, node: ParsedNode) -> tuple[ParsedNode, ...]:
        """Return child nodes for one parsed node."""
        return tuple(self.node(child_id) for child_id in node.child_ids)


class ParserBackend(Protocol):
    """Protocol for parser backends usable by static analyzer passes."""

    @property
    def name(self) -> str:
        """Return parser backend identifier."""
        ...

    def validate_available(self) -> None:
        """Raise ParserUnavailableError if this backend cannot run."""
        ...

    def parse(self, source: str, path: Path) -> ParsedModule:
        """Parse one source text into a backend-neutral parsed module."""
        ...

    def reparse(
        self,
        previous: ParsedModule,
        edits: tuple[TextEdit, ...],
        new_source: str,
    ) -> ParsedModule:
        """Reparse one module from incremental edits."""
        ...
