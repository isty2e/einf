import asyncio
import json
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import cast

import pytest

pytest.importorskip("lsprotocol")
pytest.importorskip("pygls")
from lsprotocol import types as lsp

from einf.analysis.checkers import CheckerAdapter, CheckerExecutor, CheckerRequest
from einf.analysis.checkers.model import CheckerResult
from einf.analysis.checkers.pyright import PyrightAdapter
from einf.analysis.lsp.server import build_server


@pytest.mark.parametrize(
    ("encoding", "wire_start", "wire_end"),
    (
        (lsp.PositionEncodingKind.Utf8, 11, 15),
        (lsp.PositionEncodingKind.Utf16, 7, 11),
        (lsp.PositionEncodingKind.Utf32, 6, 10),
    ),
)
def test_pyright_diagnostics_publish_on_save_in_negotiated_encoding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    encoding: lsp.PositionEncodingKind,
    wire_start: int,
    wire_end: int,
) -> None:
    target = tmp_path / "sample.py"
    source = '"\ud55c\U0001f60b"; axis\n'
    target.write_text(source, encoding="utf-8")
    payload = json.dumps(
        {
            "generalDiagnostics": [
                {
                    "file": str(target),
                    "severity": "error",
                    "message": "undefined axis",
                    "range": {
                        "start": {"line": 0, "character": 7},
                        "end": {"line": 0, "character": 11},
                    },
                }
            ]
        }
    )

    async def run_all(
        executor: CheckerExecutor,
        adapters: tuple[CheckerAdapter, ...],
        request: CheckerRequest,
    ) -> CheckerResult:
        assert request.targets == (target.resolve(),)
        assert len(adapters) == 1
        return PyrightAdapter(
            name="basedpyright", executable="basedpyright"
        ).parse_output(stdout=payload, stderr="", request=request)

    monkeypatch.setattr(CheckerExecutor, "run_all", run_all)
    published: list[lsp.PublishDiagnosticsParams] = []

    async def save() -> None:
        server = build_server()
        initialize = server.protocol.lsp_initialize(
            lsp.InitializeParams(
                capabilities=lsp.ClientCapabilities(
                    general=lsp.GeneralClientCapabilities(
                        position_encodings=(encoding,)
                    )
                ),
                root_uri=tmp_path.as_uri(),
                initialization_options={"checkers": ["basedpyright"]},
            )
        )
        handler, args, kwargs = next(initialize)
        handler(*args, **(kwargs or {}))
        with pytest.raises(StopIteration):
            next(initialize)

        server.workspace.put_text_document(
            lsp.TextDocumentItem(
                uri=target.as_uri(), language_id="python", version=1, text=source
            )
        )
        monkeypatch.setattr(
            server, "text_document_publish_diagnostics", published.append
        )
        did_save = cast(
            Callable[[lsp.DidSaveTextDocumentParams], Awaitable[None]],
            server.protocol.fm.features[lsp.TEXT_DOCUMENT_DID_SAVE],
        )
        try:
            await did_save(
                lsp.DidSaveTextDocumentParams(
                    text_document=lsp.TextDocumentIdentifier(uri=target.as_uri())
                )
            )
        finally:
            await server.einf_analysis_queue.close()
            await server.einf_checker_coordinator.close()
            await server.einf_change_debouncer.close()

    asyncio.run(save())

    assert published[-1].diagnostics[0].range == lsp.Range(
        start=lsp.Position(line=0, character=wire_start),
        end=lsp.Position(line=0, character=wire_end),
    )
    assert published[-1].diagnostics[0].message == "undefined axis"
