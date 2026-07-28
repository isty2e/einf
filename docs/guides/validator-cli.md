# Validator CLI

`einf-validate` is a checker-agnostic CLI for static DSL analysis. It
parses Python source, runs `einf` semantic analysis over call sites, and
optionally routes external type-checker output through the same report.

## Install

The CLI ships with the base install. To use the richer LibCST parser
backend, install the `analysis` extra:

```bash
pip install "einf[analysis] @ git+https://github.com/isty2e/einf.git"
```

## Basic usage

```bash
einf-validate path/to/module.py
einf-validate src/
einf-validate src/ --parser ast
einf-validate src/ --parser libcst
einf-validate src/ --checker basedpyright --checker pyrefly
einf-validate src/ --checker basedpyright --checker-timeout-seconds 60
```

Positional arguments accept any mix of files and directories. Directories
are walked recursively for `*.py`.

## Flags

| Flag | Values | Default | Purpose |
| --- | --- | --- | --- |
| `--parser` | `ast`, `libcst` | `ast` | Parser backend. `libcst` needs the `analysis` extra. |
| `--checker` | `pyright`, `basedpyright`, `zuban`, `ty`, `pyrefly` | none | External type checker to invoke alongside semantic analysis. Repeatable. |
| `--checker-timeout-seconds` | finite positive number | `30` | Maximum runtime for each checker process. |
| `--checker-max-concurrency` | positive integer | `1` | Maximum number of checker processes running concurrently. |

## Exit codes

1. `0` — no diagnostics, no parse/read failures, no checker invocation
   failures.
2. `1` — any file contains semantic or checker diagnostics, parse failures,
   or read failures; or any configured checker failed to execute.

## Output contract

`einf-validate` writes stable JSON to stdout. Top-level fields:

- `schema_version` — output contract version,
- `parser_backend` — the parser that produced the report,
- `checker_failures[]` — external checker invocation failures
  (unavailable executable, spawn failure, timeout, malformed output, etc.),
- `files[]` — per-file report.

Each file entry:

- `path`
- `diagnostics[]` — `einf` semantic diagnostics,
- `checker_diagnostics[]` — normalized external checker diagnostics,
- `axis_tokens[]` — recognized axis identifier tokens,
- `failures[]` — validator ingress failures (unreadable files,
  parse errors).

## CI integration

The CLI is designed for CI gates:

```yaml
# example: pre-merge lint step
- run: einf-validate src/ --checker basedpyright
```

The stable JSON + exit-code contract means downstream tooling can pipe
stdout into a diff viewer or annotator without parsing unstable text.
