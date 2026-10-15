from __future__ import annotations

import asyncio
import logging
import pickle
import sys
from typing import TYPE_CHECKING

from einf.analysis.parser.factory import SUPPORTED_PARSER_NAMES

from .service import LspDocumentState, LspService

if TYPE_CHECKING:
    from .analysis_queue import DocumentAnalysisRequest

_FRAME_MAGIC = b"EINF"
_HEADER_SIZE = 12
_CLEANUP_TIMEOUT_SECONDS = 1.0
_LOGGER = logging.getLogger(__name__)


class AnalysisProcessCleanupError(RuntimeError):
    """The worker slot cannot be safely reused after its cleanup deadline."""


class _AnalysisProtocol(asyncio.SubprocessProtocol):
    def __init__(self) -> None:
        self.transport: asyncio.SubprocessTransport | None = None
        self.closed: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        self.response: asyncio.Future[LspDocumentState] | None = None
        self.error: RuntimeError | None = None
        self._buffer = bytearray()

    def connection_made(self, transport: asyncio.BaseTransport) -> None:
        if not isinstance(transport, asyncio.SubprocessTransport):
            raise TypeError("semantic worker requires a subprocess transport")
        self.transport = transport

    def pipe_data_received(self, fd: int, data: bytes) -> None:
        if fd != 1:
            return
        self._buffer.extend(data)
        if len(self._buffer) < _HEADER_SIZE:
            return
        if self._buffer[:4] != _FRAME_MAGIC:
            self._fail("semantic worker sent an invalid response header")
            return

        size = int.from_bytes(self._buffer[4:_HEADER_SIZE], "big")
        if len(self._buffer) < _HEADER_SIZE + size:
            return
        payload = bytes(self._buffer[_HEADER_SIZE : _HEADER_SIZE + size])
        del self._buffer[: _HEADER_SIZE + size]
        response = self.response
        if response is None or response.done() or self._buffer:
            self._fail("semantic worker sent an unexpected response")
            return
        try:
            # These pipes connect only to our own same-interpreter child.
            result = pickle.loads(payload)
        except Exception as error:
            _LOGGER.exception("Semantic worker response could not be decoded")
            self._fail(f"semantic worker response could not be decoded: {error}")
            return

        if isinstance(result, LspDocumentState):
            response.set_result(result)
        elif isinstance(result, str):
            self._fail(result)
        else:
            self._fail("semantic worker sent an invalid result")

    def pipe_connection_lost(self, fd: int, exc: Exception | None) -> None:
        if fd == 1:
            self._fail(f"semantic worker output closed: {exc or 'end of stream'}")

    def connection_lost(self, exc: Exception | None) -> None:
        self._fail(f"semantic worker exited: {exc or 'process terminated'}")
        if not self.closed.done():
            self.closed.set_result(None)

    def _fail(self, message: str) -> None:
        self.error = RuntimeError(message)
        if self.response is not None and not self.response.done():
            self.response.set_exception(self.error)


class LspAnalysisProcess:
    """Own one reusable semantic-analysis child and its private pipes.

    Cancellation terminates the child. Cleanup waits at most one second before
    reporting failure; an unreaped child retains its slot rather than allowing
    another process to be started beside it.
    """

    def __init__(self) -> None:
        self._protocol: _AnalysisProtocol | None = None
        self._spawn: (
            asyncio.Task[tuple[asyncio.SubprocessTransport, _AnalysisProtocol]] | None
        ) = None
        self._retirement: asyncio.Task[None] | None = None
        self._cleanup_deadline: float | None = None
        self._parser: str | None = None
        self._closed = False

    async def analyze(
        self, request: DocumentAnalysisRequest, *, parser: str
    ) -> LspDocumentState:
        """Analyze one snapshot; callers must serialize use of this worker.

        Parameters
        ----------
        request : DocumentAnalysisRequest
            Immutable source snapshot whose result remains parent-owned.
        parser : str
            Built-in parser name, ``ast`` or ``libcst``.

        Returns
        -------
        LspDocumentState
            Complete semantic result for the submitted snapshot.

        Raises
        ------
        ValueError
            The parser name is unsupported.
        AnalysisProcessCleanupError
            Worker termination or pipe cleanup exceeded its deadline.
        RuntimeError
            The process is closed, analysis failed, or its response is invalid.
        asyncio.CancelledError
            The caller cancelled execution; worker retirement is requested.
        """
        if self._closed:
            raise RuntimeError("semantic analysis process is closed")
        if parser not in SUPPORTED_PARSER_NAMES:
            raise ValueError(f"unsupported parser backend: {parser}")

        try:
            if self._retirement is not None or (
                self._protocol is not None
                and (self._parser != parser or self._protocol.error is not None)
            ):
                await self._stop()
            if self._spawn is None:
                protocol = _AnalysisProtocol()
                self._protocol = protocol
                self._parser = parser
                self._spawn = asyncio.create_task(
                    asyncio.get_running_loop().subprocess_exec(
                        lambda: protocol,
                        sys.executable,
                        "-m",
                        "einf.analysis.lsp.analysis_process",
                        parser,
                        stdin=asyncio.subprocess.PIPE,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=None,
                    )
                )
            transport, protocol = await asyncio.shield(self._spawn)
            if protocol.error is not None:
                raise protocol.error
            stdin = transport.get_pipe_transport(0)
            if stdin is None:
                raise RuntimeError("semantic worker input pipe is unavailable")
            if not isinstance(stdin, asyncio.WriteTransport):
                raise TypeError("semantic worker input requires a write transport")

            response: asyncio.Future[LspDocumentState] = (
                asyncio.get_running_loop().create_future()
            )
            protocol.response = response
            payload = pickle.dumps((request.uri, request.source, request.version))
            stdin.write(_FRAME_MAGIC + len(payload).to_bytes(8, "big") + payload)
            state = await response
            protocol.response = None
            if (state.uri, state.source, state.version) != (
                request.uri,
                request.source,
                request.version,
            ):
                raise RuntimeError("semantic worker returned a different snapshot")
            return state
        except OSError as error:
            await self._stop()
            raise RuntimeError(f"semantic worker execution failed: {error}") from error
        except BaseException:
            await self._stop()
            raise

    async def close(self) -> None:
        """Stop the owned process and close its pipes.

        Raises
        ------
        AnalysisProcessCleanupError
            Worker cleanup exceeded its deadline; replacement remains prohibited.
        """
        self._closed = True
        await self._stop()

    async def _stop(self) -> None:
        if self._spawn is None:
            return
        if self._retirement is None:
            self._retirement = asyncio.create_task(self._reap())
            self._cleanup_deadline = (
                asyncio.get_running_loop().time() + _CLEANUP_TIMEOUT_SECONDS
            )
        try:
            if self._retirement.done():
                self._retirement.result()
            else:
                deadline = self._cleanup_deadline
                if deadline is None:
                    raise RuntimeError("semantic worker cleanup has no deadline")
                await asyncio.wait_for(
                    asyncio.shield(self._retirement),
                    timeout=max(0.0, deadline - asyncio.get_running_loop().time()),
                )
        except TimeoutError:
            raise AnalysisProcessCleanupError(
                "semantic worker cleanup did not complete within "
                f"{_CLEANUP_TIMEOUT_SECONDS:g} seconds"
            ) from None

        self._spawn = None
        self._protocol = None
        self._retirement = None
        self._cleanup_deadline = None
        self._parser = None

    async def _reap(self) -> None:
        spawn = self._spawn
        if spawn is None:
            return
        protocol = self._protocol
        if protocol is not None and protocol.transport is not None:
            protocol.transport.close()
        try:
            transport, protocol = await asyncio.shield(spawn)
        except (OSError, RuntimeError):
            if protocol is not None and protocol.transport is not None:
                await protocol.closed
            return
        transport.close()
        await protocol.closed


def main() -> None:
    """Serve private framed requests from the parent LSP process."""
    service = LspService(parser=sys.argv[1])
    stdin = sys.stdin.buffer
    stdout = sys.stdout.buffer
    while header := stdin.read(_HEADER_SIZE):
        if len(header) != _HEADER_SIZE or header[:4] != _FRAME_MAGIC:
            raise RuntimeError("invalid semantic worker request header")
        size = int.from_bytes(header[4:], "big")
        payload = stdin.read(size)
        if len(payload) != size:
            raise RuntimeError("incomplete semantic worker request")
        uri, source, version = pickle.loads(payload)
        try:
            result: LspDocumentState | str = service.analyze_document(
                uri=uri, source=source, version=version
            )
        except Exception as error:
            _LOGGER.exception("Semantic analysis failed")
            result = f"semantic analysis failed: {type(error).__name__}: {error}"
        response = pickle.dumps(result)
        stdout.write(_FRAME_MAGIC + len(response).to_bytes(8, "big") + response)
        stdout.flush()
        del uri, source, version, payload, result, response


if __name__ == "__main__":
    main()
