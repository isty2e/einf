# Zed

Zed is a usable `einf-lsp` target today, but it is not as polished or as
well-validated as Helix.

The right mental model is:

1. keep your main Python language server for Python typing, navigation, and completion,
2. run `einf-lsp` as a second semantic sidecar,
3. keep `initializeOptions.checkers` empty unless you intentionally want fallback single-server behavior.

## Status

Current support tier:

- manual configuration path exists,
- no repository-owned Zed extension is required,
- less validated than the Helix path,
- not blocked on the same extension work as VS Code.

In other words, Zed is closer to Helix than to VS Code, but it is still a
manual path rather than a polished editor-specific product surface.

## Install

```bash
pip install -e .
pip install -e ".[lsp]"
```

Make sure `einf-lsp` is available on your `PATH`.

## Recommended Server Contract

Run `einf-lsp` with:

```json
{
  "parser": "ast",
  "checkers": []
}
```

Use it alongside your primary Python language server, not as a replacement for
general Python IDE features.

## What This Repository Promises

This repository currently documents the sidecar contract and the intended
server role:

- `einf-lsp` is the semantic sidecar,
- your primary Python server owns Python typing/navigation,
- checker execution through `einf-lsp` remains fallback-only behavior.

What this repository does **not** promise yet:

- a Zed-specific extension,
- a heavily tested Zed setup matrix,
- editor-specific affordances beyond the generic LSP sidecar contract.
