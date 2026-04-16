# VS Code

VS Code is still not a first-class editor target for `einf-lsp` from this
repository.

## Current Status

Unlike Helix and unlike the documented manual Zed path, the practical VS Code
path is still a thin extension that launches and configures `einf-lsp`.

This repository does **not** currently ship that extension.

## Recommended Future Model

The target VS Code setup is:

1. keep your primary Python server or extension for Python typing, navigation,
   and completion,
2. run `einf-lsp` as a second semantic sidecar for `einf`,
3. keep `initializeOptions.checkers` empty in the default path,
4. treat checker execution through `einf-lsp` as fallback-only behavior.

## What You Can Use Today

Today, the supported paths from this repository are:

1. the static analysis CLI:

```bash
einf-validate path/to/module.py
einf-validate src/ --checker basedpyright
```

2. another editor with a documented sidecar path, such as Helix or Zed.

## Why There Is No VS Code Guide Yet

Without a thin VS Code extension, any setup instructions would be editor-local
workarounds rather than a stable supported path. That is not a good contract to
document as if it were productized support.
