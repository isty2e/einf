# LSP sidecar

`einf` ships a minimal external language server sidecar for editor
integration. It is packaged as the `einf-lsp` console entry point and
installed by the `lsp` extra.

```bash
pip install "einf[lsp] @ git+https://github.com/isty2e/einf.git"
einf-lsp
```

## Recommended editor model

1. run `einf-lsp` alongside your primary Python language server,
2. let the primary Python server handle Python typing and navigation,
3. use `einf-lsp` for `einf` semantics, semantic tokens, hover metadata,
   and inlay hints.

For editors that can comfortably run multiple language servers, keep
`einf-lsp` focused on `einf` semantics and leave external checker
execution to the primary Python toolchain.

Per-editor setup lives under [Editors](editors/index.md).

## Configuration

The LSP server uses `initialize` options as its configuration source of
truth. The default (and recommended) mode keeps `checkers` empty.

```json
{
  "parser": "ast",
  "checkers": []
}
```

### Minimal scope

1. document sync,
2. `publishDiagnostics` from `einf` semantic analysis,
3. optional saved-file checker diagnostics from configured external
   checkers,
4. semantic tokens derived from `axis_tokens`.

### Richer editor affordances

1. hover metadata for axis-group relationships and role summaries,
2. inlay hints for selected non-trivial axis roles (`contracted`,
   `reduced`, `introduced`, `pack`).

## Editor support summary

| Editor | Status | Notes |
| --- | --- | --- |
| Helix | Supported | Clean documented sidecar path. |
| Zed | Adapter needed | Zed's sidecar model fits `einf-lsp`, but its settings target recognized adapters and this repository does not ship one yet. |
| VS Code | Not first-class yet | A thin extension is the intended supported path. |

## Fallback single-server mode

If your editor setup cannot comfortably run `einf-lsp` alongside a
separate Python language server, you can ask `einf-lsp` to invoke
external checkers on save by passing `checkers` in `initialize` options.

```json
{
  "parser": "ast",
  "checkers": ["basedpyright", "pyrefly"]
}
```

External checker diagnostics refresh on save boundaries. Unsaved document
changes continue to receive fresh `einf` semantic diagnostics and
semantic tokens, but stale checker diagnostics are not retained as if
they were current.
