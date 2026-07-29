import asyncio
from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path
from typing import cast

import pytest
from lsprotocol import types as lsp

from einf.analysis.lsp import (
    LspDocumentState,
    LspService,
    encode_semantic_tokens,
)
from einf.analysis.lsp.position_codec import LspPositionCodec
from einf.analysis.lsp.server import _publish_document_state, build_server
from einf.analysis.model import AnalysisDiagnostic, AxisToken, TextPosition, TextSpan
from einf.analysis.validator.model import ValidationFileReport

_SOURCE = "한😋axis\n"
_AXIS_SPAN = TextSpan(
    start=TextPosition(line=1, column=2),
    end=TextPosition(line=1, column=6),
)
_ENCODING_CASES = (
    (lsp.PositionEncodingKind.Utf8, 7, 11),
    (lsp.PositionEncodingKind.Utf16, 3, 7),
    (lsp.PositionEncodingKind.Utf32, 2, 6),
)


def _initialize_server(
    position_encodings: Sequence[lsp.PositionEncodingKind | str] | None,
):
    server = build_server()
    general = (
        lsp.GeneralClientCapabilities(position_encodings=position_encodings)
        if position_encodings is not None
        else None
    )
    params = lsp.InitializeParams(
        capabilities=lsp.ClientCapabilities(general=general),
        root_uri="file:///workspace",
    )
    initialize = server.protocol.lsp_initialize(params)
    handler, args, kwargs = next(initialize)
    handler(*args, **(kwargs or {}))
    with pytest.raises(StopIteration) as completed:
        next(initialize)
    return server, completed.value.value


@pytest.mark.parametrize(
    ("encoding", "wire_start", "wire_end"),
    _ENCODING_CASES,
)
def test_position_codec_round_trips_multibyte_and_astral_prefixes(
    encoding: lsp.PositionEncodingKind,
    wire_start: int,
    wire_end: int,
) -> None:
    codec = LspPositionCodec(
        lines=tuple(_SOURCE.splitlines(keepends=True)),
        encoding=encoding,
    )

    lsp_range = codec.to_lsp_range(_AXIS_SPAN)

    assert lsp_range == lsp.Range(
        start=lsp.Position(line=0, character=wire_start),
        end=lsp.Position(line=0, character=wire_end),
    )
    assert codec.from_lsp_range(lsp_range) == _AXIS_SPAN


def test_position_codec_rejects_unsupported_encoding() -> None:
    with pytest.raises(ValueError, match="unsupported LSP position encoding"):
        LspPositionCodec(
            lines=tuple(_SOURCE.splitlines(keepends=True)),
            encoding="utf-7",
        )


@pytest.mark.parametrize(
    ("encoding", "wire_start"),
    (
        (lsp.PositionEncodingKind.Utf8, 33),
        (lsp.PositionEncodingKind.Utf16, 29),
        (lsp.PositionEncodingKind.Utf32, 28),
    ),
)
def test_analyzed_unicode_source_encodes_canonical_axis_span(
    encoding: lsp.PositionEncodingKind,
    wire_start: int,
) -> None:
    source = (
        "from einf import ax, axes, rearrange\n"
        'b = axes("b")[0]\n'
        'marker = "한😋"; rearrange(ax[b], ax[b])\n'
    )
    state = LspService().open_document(
        uri="file:///workspace/sample.py",
        source=source,
        version=1,
    )

    assert state.report.axis_tokens[0].span.start == TextPosition(line=3, column=28)
    encoded = encode_semantic_tokens(
        state.report.axis_tokens,
        position_codec=LspPositionCodec(
            lines=state.source_lines,
            encoding=encoding,
        ),
    )
    assert encoded[:3] == [2, wire_start, 1]


@pytest.mark.parametrize(
    ("offered", "expected"),
    (
        (
            (lsp.PositionEncodingKind.Utf8, lsp.PositionEncodingKind.Utf16),
            lsp.PositionEncodingKind.Utf8,
        ),
        (("utf-8",), lsp.PositionEncodingKind.Utf8),
        ((lsp.PositionEncodingKind.Utf32,), lsp.PositionEncodingKind.Utf32),
        (("utf-7",), lsp.PositionEncodingKind.Utf16),
        (None, lsp.PositionEncodingKind.Utf16),
    ),
)
def test_server_uses_pygls_position_encoding_negotiation(
    offered: tuple[lsp.PositionEncodingKind | str, ...] | None,
    expected: lsp.PositionEncodingKind,
) -> None:
    server, result = _initialize_server(offered)

    assert server.workspace.position_encoding == expected
    assert server.einf_position_encoding == expected
    assert result.capabilities.position_encoding == expected


@pytest.mark.parametrize(
    ("encoding", "wire_start", "wire_end"),
    _ENCODING_CASES,
)
def test_negotiated_position_encoding_applies_to_incremental_edits(
    encoding: lsp.PositionEncodingKind,
    wire_start: int,
    wire_end: int,
) -> None:
    server, _ = _initialize_server((encoding,))
    uri = "file:///workspace/sample.py"
    server.workspace.put_text_document(
        lsp.TextDocumentItem(
            uri=uri,
            language_id="python",
            version=1,
            text=_SOURCE,
        )
    )

    server.workspace.update_text_document(
        lsp.VersionedTextDocumentIdentifier(uri=uri, version=2),
        lsp.TextDocumentContentChangePartial(
            range=lsp.Range(
                start=lsp.Position(line=0, character=wire_start),
                end=lsp.Position(line=0, character=wire_end),
            ),
            text="value",
        ),
    )

    assert server.workspace.get_text_document(uri).source == "한😋value\n"


@pytest.mark.parametrize(
    ("encoding", "wire_start", "wire_end"),
    _ENCODING_CASES,
)
def test_server_features_share_the_negotiated_position_codec(
    encoding: lsp.PositionEncodingKind,
    wire_start: int,
    wire_end: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server, _ = _initialize_server((encoding,))
    uri = "file:///workspace/sample.py"
    token = AxisToken(
        name="axis",
        span=_AXIS_SPAN,
        group=0,
        roles=("introduced",),
    )
    diagnostic = AnalysisDiagnostic(
        code="test-code",
        message="test diagnostic",
        severity="error",
        span=_AXIS_SPAN,
    )
    state = LspDocumentState(
        uri=uri,
        path=Path("/workspace/sample.py"),
        version=1,
        source=_SOURCE,
        semantic_report=ValidationFileReport(
            path="/workspace/sample.py",
            diagnostics=(diagnostic,),
            checker_diagnostics=(),
            axis_tokens=(token,),
            failures=(),
        ),
    )
    server.einf_service.commit_document_state(state)
    published: list[lsp.PublishDiagnosticsParams] = []
    monkeypatch.setattr(
        server,
        "text_document_publish_diagnostics",
        published.append,
    )

    semantic_handler = cast(
        Callable[[lsp.SemanticTokensParams], Awaitable[lsp.SemanticTokens]],
        server.protocol.fm.features[lsp.TEXT_DOCUMENT_SEMANTIC_TOKENS_FULL],
    )
    inlay_handler = cast(
        Callable[[lsp.InlayHintParams], Awaitable[list[lsp.InlayHint]]],
        server.protocol.fm.features[lsp.TEXT_DOCUMENT_INLAY_HINT],
    )
    hover_handler = cast(
        Callable[[lsp.HoverParams], Awaitable[lsp.Hover | None]],
        server.protocol.fm.features[lsp.TEXT_DOCUMENT_HOVER],
    )

    async def request_features():
        semantic_tokens = await semantic_handler(
            lsp.SemanticTokensParams(
                text_document=lsp.TextDocumentIdentifier(uri=uri),
            )
        )
        inlay_hints = await inlay_handler(
            lsp.InlayHintParams(
                text_document=lsp.TextDocumentIdentifier(uri=uri),
                range=lsp.Range(
                    start=lsp.Position(line=0, character=0),
                    end=lsp.Position(line=0, character=wire_end),
                ),
            )
        )
        hover = await hover_handler(
            lsp.HoverParams(
                text_document=lsp.TextDocumentIdentifier(uri=uri),
                position=lsp.Position(line=0, character=wire_start),
            )
        )
        return semantic_tokens, inlay_hints, hover

    semantic_tokens, inlay_hints, hover = asyncio.run(request_features())
    _publish_document_state(server, state)

    assert semantic_tokens.data == [0, wire_start, 4, 0, 1]
    assert len(inlay_hints) == 1
    assert inlay_hints[0].position == lsp.Position(line=0, character=wire_end)
    assert hover is not None
    assert published[0].diagnostics[0].range == lsp.Range(
        start=lsp.Position(line=0, character=wire_start),
        end=lsp.Position(line=0, character=wire_end),
    )
