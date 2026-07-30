# Helix

!!! warning "Partial support"
    Helix can route `einf` diagnostics and inlay hints reliably. Hover
    ownership requires a tradeoff with the primary Python server, and Helix
    does not currently expose LSP semantic tokens.

Helix is the most validated editor path for `einf-lsp`, but it does not expose
every capability advertised by the server.

Helix supports language-server configuration directly in `languages.toml`, so
you can run your primary Python language server and `einf-lsp` side by side.

The relevant upstream reference is the Helix
[language configuration documentation](https://docs.helix-editor.com/languages.html),
which documents:

- `[language-server.<name>]` entries for command, args, and initialization config,
- `language-servers = [...]` per-language ordering,
- `only-features` and `except-features` routing across multiple language servers.

Helix's documented feature list does not include semantic tokens. The
`einf-lsp` semantic-token capability therefore cannot provide axis-role
coloring in Helix.

## Install

```bash
pip install "einf[lsp] @ git+https://github.com/isty2e/einf.git"
```

Make sure `einf-lsp` is on your `PATH`.

## Recommended Model

Run:

- one primary Python language server for typing/navigation
- `einf-lsp` for semantic diagnostics and inlay hints

Use a project-local `.helix/languages.toml` or your global Helix
`languages.toml`.

Inlay hints are disabled by default in Helix. Enable them in your Helix
`config.toml` as documented in the upstream
[`editor.lsp` settings](https://docs.helix-editor.com/editor.html#editorlsp-section):

```toml
# ~/.config/helix/config.toml
[editor.lsp]
display-inlay-hints = true
```

Then configure explicit feature ownership in `languages.toml`:

```toml
[language-server.einf-lsp]
command = "einf-lsp"

[language-server.einf-lsp.config]
parser = "ast"
checkers = []

[[language]]
name = "python"
language-servers = [
  { name = "pyright", except-features = ["inlay-hints"] },
  { name = "einf-lsp", only-features = ["diagnostics", "inlay-hints"] },
]
```

If your primary Python server is not named `pyright` in your Helix config,
replace it with the server you actually use.

This routing preserves the primary server's Python hover, completion, typing,
and navigation. Helix merges diagnostics from both servers and selects
`einf-lsp` as the first server still eligible to provide inlay hints.

## Hover Tradeoff

Helix routes features by server, not by the source expression under the
cursor. A configuration cannot reliably use the primary server for ordinary
Python hover while reserving hover on `einf` expressions for `einf-lsp`.

If `einf` axis metadata is more important than general Python hover, replace
the `language-servers` list above with:

```toml
[[language]]
name = "python"
language-servers = [
  { name = "pyright", except-features = ["hover", "inlay-hints"] },
  { name = "einf-lsp", only-features = ["diagnostics", "hover", "inlay-hints"] },
]
```

This explicitly gives hover and inlay hints to `einf-lsp`, but disables
hover from the primary Python server for the whole Python document.

## Feature Coverage

| Feature | Helix behavior |
| --- | --- |
| Semantic diagnostics | Available and merged with primary-server diagnostics. |
| Inlay hints | Available after `display-inlay-hints = true`; explicitly routed to `einf-lsp` above. |
| Hover metadata | Available only by accepting the document-wide ownership tradeoff above. |
| Semantic tokens | Unavailable because Helix does not currently support the LSP semantic-token feature. |

## Fallback Single-Server Mode

If you do not want a second Python language server in Helix, you can still run
only `einf-lsp` and let it invoke external checkers on save.

Example:

```toml
[language-server.einf-lsp]
command = "einf-lsp"

[language-server.einf-lsp.config]
parser = "ast"
checkers = ["basedpyright"]

[[language]]
name = "python"
language-servers = ["einf-lsp"]
```

This is less clean than the separate-sidecar model because `einf-lsp` now owns
both `einf` semantics and checker orchestration.
