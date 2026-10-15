# LSP sidecar

`einf` ships a minimal external language server sidecar for editor
integration. Every installation exposes the `einf-lsp` console entry point;
the `lsp` extra installs the optional dependencies required to run it.

```bash
pip install "einf[lsp] @ git+https://github.com/isty2e/einf.git"
einf-lsp
```

A base-only installation keeps the command discoverable for editor
configuration, but invoking it reports the required `einf[lsp]` installation
instead of starting the server.

## Recommended editor model

1. run `einf-lsp` alongside your primary Python language server,
2. let the primary Python server handle Python typing and navigation,
3. use `einf-lsp` for the semantic features that the editor can route to it:
   diagnostics, semantic tokens, hover metadata, and inlay hints.

For editors that can comfortably run multiple language servers, keep
`einf-lsp` focused on `einf` semantics and leave external checker
execution to the primary Python toolchain.

Editor clients differ in which capabilities they support and how they select
between overlapping servers. Per-editor setup and limitations live under
[Editors](editors/index.md).

## Configuration

The LSP server uses `initialize` options as its configuration source of
truth. The default (and recommended) mode keeps `checkers` empty.

```json
{
  "parser": "ast",
  "checkers": [],
  "checkerTimeoutSeconds": 30,
  "checkerCleanupTimeoutSeconds": 1,
  "checkerMaxConcurrency": 1,
  "checkerMaxOutputBytes": 8388608,
  "checkerMaxDiagnostics": 10000,
  "checkerMaxFieldLength": 4096
}
```

### Optional analysis limits

No analysis resource limits are enabled by default. You can set just the limit
you need; for example, this caps each analyzed source snapshot at 8 MiB:

```json
{
  "analysisMaxDocumentBytes": 8388608
}
```

| Option | Resource |
| --- | --- |
| `analysisMaxDocuments` | Distinct document URIs with retained analysis snapshots, including work still being cancelled. |
| `analysisMaxDocumentBytes` | UTF-8 bytes in a single source snapshot. |
| `analysisMaxSourceBytes` | Sum of UTF-8 bytes across retained source snapshots. |

Each option is independent. Omit it or use `null` for no limit; otherwise supply
a positive integer. Invalid explicit values reject initialization rather than
silently disabling a limit. These options do not change `checkerMaxConcurrency`
or its default of 1.

Snapshots count while waiting for debounce or queue space, during analysis, and
while stored as the current result. Sharing one snapshot across these stages
counts it once. Overlapping versions count separately until obsolete work has
finished retiring, so a replacement may be refused until that work releases space.

When admission would exceed a limit, the sidecar discards that document's obsolete
analysis and publishes an `analysis_unavailable` warning for the refused version.
It does not run semantic analysis or saved-file checkers for that snapshot, and
hover, inlay hints and semantic tokens do not return the obsolete result. Document
synchronization continues. Reduce the source size, close other documents or
adjust the limits; restart the sidecar after changing initialize options. The
next edit, save or semantic request can retry. Other documents are not evicted
to make room.

These are logical source-snapshot limits, **not a process-memory cap**. They do
not bound pygls's document text, incoming messages, parser allocations, worker
copies or OS memory. An on-demand request for an unopened file reads it before
checking admission. UTF-8 byte limits also require valid UTF-8 source text.

Workspace folders are not access permissions: standalone files outside them and
untitled documents remain supported. The sidecar expects a trusted editor
client; these limits do not turn its stdio, TCP or WebSocket transport into a
sandbox or an authenticated remote service.

### Minimal scope

1. document sync,
2. `publishDiagnostics` from `einf` semantic analysis,
3. optional saved-file checker diagnostics from configured external
   checkers,
4. semantic tokens derived from `axis_tokens`.

### Richer editor affordances

1. hover metadata for axis-group relationships and role summaries,
2. inlay hints for selected operation roles (`contracted`, `reduced`,
   `introduced`) and axis-pack structure.

## Responsiveness model

`einf-lsp` keeps interactive editor requests on cached semantic state:

1. `didOpen` analyzes immediately,
2. `didChange` events are coalesced briefly so rapid edits analyze only the
   latest document version,
3. semantic analysis runs in a bounded queue backed by two reusable worker
   processes,
4. semantic tokens, hover, and inlay hints read the latest cached analysis,
5. a pending edit is flushed before a semantic-token, hover, or inlay request
   is answered.

A newer analysis request for the same document stops obsolete running analysis
before reusing its worker slot. Closing a document also stops its analysis.
Completed requests reuse the worker process; each edit does not start a new
interpreter. Only the parent sidecar stores results and publishes diagnostics.

Workers use Python's `-P` option so the editor's working directory is not
implicitly added to their module search path. Explicit interpreter configuration,
including `PYTHONPATH`, is preserved; this is not a sandbox for configured imports.

Shutdown waits for worker termination and pipe cleanup. Cleanup has a one-second
deadline per worker, separate from analysis itself: valid analysis has no default
time limit. If cleanup cannot be confirmed, the sidecar reports an error and stops
accepting semantic analysis rather than starting more workers. Restart the sidecar
in that case. Process isolation does not guarantee termination when the operating
system cannot complete it.

The sidecar also takes a conservative fast path for Python files that contain
no `einf` lexical marker. Those files produce no `einf` diagnostics or axis
metadata without paying the full semantic-analysis cost.

## Editor support summary

| Editor | Status | Notes |
| --- | --- | --- |
| Helix | Partial | Diagnostics and inlay hints are routable; hover has a server-ownership tradeoff and semantic tokens are unavailable. |
| Zed | Adapter needed | Zed's sidecar model fits `einf-lsp`, but its settings target recognized adapters and this repository does not ship one yet. |
| VS Code | Not first-class yet | A thin extension is the intended supported path. |

## Fallback single-server mode

If your editor setup cannot comfortably run `einf-lsp` alongside a
separate Python language server, you can ask `einf-lsp` to invoke
external checkers on save by passing `checkers` in `initialize` options.

```json
{
  "parser": "ast",
  "checkers": ["basedpyright", "pyrefly"],
  "checkerTimeoutSeconds": 30,
  "checkerCleanupTimeoutSeconds": 1,
  "checkerMaxConcurrency": 1,
  "checkerMaxOutputBytes": 8388608,
  "checkerMaxDiagnostics": 10000,
  "checkerMaxFieldLength": 4096
}
```

External checker diagnostics refresh on save boundaries. Unsaved document
changes continue to receive fresh `einf` semantic diagnostics and
semantic tokens, but stale checker diagnostics are not retained as if
they were current.

Checker execution uses a separate bounded async subprocess coordinator.
`checkerTimeoutSeconds` limits each checker run. After a timeout or
cancellation, `checkerCleanupTimeoutSeconds` limits how long the sidecar waits
for process and pipe cleanup. `checkerMaxConcurrency` limits checker processes
across documents and tools. On POSIX systems, the sidecar starts each checker
in its own process group and terminates that group on timeout or cancellation.
Saving or closing a newer document generation cancels obsolete checker work
before its results can be published.

Checker execution can be much slower than `einf` semantic analysis. Treat it
as a compatibility fallback for editors that cannot run a separate Python
language server, not as the recommended interactive path.
