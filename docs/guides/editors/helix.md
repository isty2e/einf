# Helix

Helix is the cleanest and most validated currently supported editor path for
`einf-lsp`.

Helix supports language-server configuration directly in `languages.toml`, so
you can run your primary Python language server and `einf-lsp` side by side.

## Install

```bash
pip install -e .
pip install -e ".[lsp]"
```

Make sure `einf-lsp` is on your `PATH`.

## Recommended Model

Run:

- one primary Python language server for typing/navigation
- `einf-lsp` for `einf` semantics

Use a project-local `.helix/languages.toml` or your global Helix
`languages.toml`.

Example:

```toml
[language-server.einf-lsp]
command = "einf-lsp"

[language-server.einf-lsp.config]
parser = "ast"
checkers = []

[[language]]
name = "python"
language-servers = ["pyright", "einf-lsp"]
```

If your primary Python server is not named `pyright` in your Helix config,
replace it with the server you actually use.

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
