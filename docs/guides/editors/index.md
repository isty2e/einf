# Editor Integration

## Per-editor status

| Editor | Status | Notes |
| --- | --- | --- |
| [Helix](helix.md) | **Supported** | Working copy-paste `languages.toml` recipe. |
| [Zed](zed.md) | **Adapter required** | Architectural fit is good, but no adapter ships from this repo yet. |
| [VS Code](vscode.md) | **Not shipped** | Needs a thin extension; fall back to the validator CLI or another editor. |

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
pip install "einf[lsp] @ git+https://github.com/isty2e/einf.git"
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

Per-editor recipes: [Helix](helix.md), [Zed](zed.md), [VS Code](vscode.md).
See the [status table](#per-editor-status) above for what each page
actually delivers today.

## 2. Fallback Single-Server Mode

This mode keeps `einf-lsp` as the only documented server and asks it to invoke
external checkers on save.

Use this only when running a separate Python language server is impractical for
your editor setup.

Install:

```bash
pip install "einf[lsp] @ git+https://github.com/isty2e/einf.git"
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
pip install git+https://github.com/isty2e/einf.git
```

Optional parser support:

```bash
pip install "einf[analysis] @ git+https://github.com/isty2e/einf.git"
```

Examples:

```bash
einf-validate path/to/module.py
einf-validate src/ --parser ast
einf-validate src/ --parser libcst
einf-validate src/ --checker basedpyright --checker pyrefly
```

The validator is checker-agnostic and writes stable JSON to stdout.
