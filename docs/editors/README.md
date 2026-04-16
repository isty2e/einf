# Editor Integration

`einf` exposes three integration modes today.

## 1. Separate `einf-lsp` Sidecar

This is the recommended model when your editor can run more than one language
server for Python.

- Keep your primary Python language server for typing, completion, navigation,
  and general Python diagnostics.
- Run `einf-lsp` alongside it for `einf` semantic diagnostics, semantic tokens,
  hover metadata, and inlay hints.
- Keep `initializeOptions.checkers` empty in this mode.

Install:

```bash
pip install -e .
pip install -e ".[lsp]"
```

Launch command:

```bash
einf-lsp
```

Recommended initialize options:

```json
{
  "parser": "ast",
  "checkers": []
}
```

Current editor-specific status:

- Helix: clean documented path and the most validated editor target today
- Zed: documented manual sidecar path exists, but it is less polished and less validated than Helix
- VS Code: not yet a first-class path from this repository; a thin extension is still needed

See:

- `docs/editors/helix.md`
- `docs/editors/zed.md`
- `docs/editors/vscode.md`

## 2. Fallback Single-Server Mode

This mode keeps `einf-lsp` as the only documented server and asks it to invoke
external checkers on save.

Use this only when running a separate Python language server is impractical for
your editor setup.

Install:

```bash
pip install -e .
pip install -e ".[lsp]"
```

You must also install the checker executables you reference in `checkers`, for
example `basedpyright`, `pyright`, `ty`, `zuban`, or `pyrefly`.

Example initialize options:

```json
{
  "parser": "ast",
  "checkers": ["basedpyright", "pyrefly"]
}
```

Behavior:

- `einf` semantic diagnostics stay fresh on open/change/save
- external checker diagnostics refresh on save boundaries
- stale checker diagnostics are not presented as if they tracked unsaved edits

## 3. Static Analysis CLI

Use the validator when you want `einf` analysis in CI, scripts, or local batch
checks without editor integration.

Base install:

```bash
pip install -e .
```

Optional parser support:

```bash
pip install -e ".[analysis]"
```

Examples:

```bash
einf-validate path/to/module.py
einf-validate src/ --parser ast
einf-validate src/ --parser libcst
einf-validate src/ --checker basedpyright --checker pyrefly
```

The validator is checker-agnostic and writes stable JSON to stdout.
