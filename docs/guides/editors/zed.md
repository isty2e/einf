# Zed

!!! warning "Adapter required — no working recipe yet"
    The architectural fit is good, but Zed wires language servers
    through recognized adapters and this repository does not ship one.
    The page below documents the intended server contract, not a
    copy-paste setup.

Zed is a plausible long-term home for `einf-lsp`, but today it still needs an
adapter layer rather than just repository-local settings.

The core model is still the same:

1. keep your main Python language server for Python typing, navigation, and completion,
2. run `einf-lsp` as a second semantic sidecar,
3. keep `initializeOptions.checkers` empty unless you intentionally want fallback single-server behavior.

## Why Zed Is Closer to Helix Than to VS Code

Zed already has the right kind of high-level product model for a sidecar server:

- multiple language servers per language,
- per-server initialization options,
- per-server binary configuration.

That means the *intended product shape* is much closer to Helix than to
VS Code, even though the last mile is different.

The relevant Zed documentation is:

- [Configuring Languages](https://zed.dev/docs/configuring-languages)

In particular, Zed documents language-server ordering plus per-server
initialization and binary configuration for recognized adapters.

## What Is Still Missing

The gap is not architectural fit. The gap is productization.

This repository does **not** currently ship:

- a Zed-specific adapter or extension for `einf-lsp`,
- a repository-owned copy-paste `settings.json` snippet that we can validate across versions,
- a heavily tested Zed setup matrix.

The key constraint from Zed's own docs is that the `lsp` section configures
language servers that Zed already recognizes through its built-in adapters or
installed extensions. This repository does not yet provide that adapter layer
for `einf-lsp`.

That is why this page stops short of promising a working Zed recipe today.

## Install

```bash
pip install "einf[lsp] @ git+https://github.com/isty2e/einf.git"
```

Make sure `einf-lsp` is available on your `PATH`.

## Intended Server Contract

When a Zed adapter exists, `einf-lsp` should be wired as a semantic sidecar with:

```json
{
  "parser": "ast",
  "checkers": []
}
```

It should live alongside the primary Python language server, not replace
general Python IDE features.

## What This Repository Promises

This repository currently documents the sidecar contract and intended server role:

- `einf-lsp` is the semantic sidecar,
- your primary Python server owns Python typing/navigation,
- checker execution through `einf-lsp` remains fallback-only behavior.

What this repository does **not** promise yet:

- a Zed-specific extension,
- a stable copy-paste Zed configuration owned by this repository,
- editor-specific affordances beyond the generic LSP sidecar contract.
