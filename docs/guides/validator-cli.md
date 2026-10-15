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
| `--max-discovery-entries` | positive integer | unrestricted | Maximum explicit target attempts plus directory entries encountered. |
| `--max-file-bytes` | positive integer | unrestricted | Maximum raw source bytes per file. |
| `--max-total-bytes` | positive integer | unrestricted | Maximum cumulative raw source bytes captured by the validator. |
| `--checker` | `pyright`, `basedpyright`, `zuban`, `ty`, `pyrefly` | none | External type checker to invoke alongside semantic analysis. Repeatable; repeated names are collapsed. |
| `--checker-timeout-seconds` | finite positive number | `30` | Maximum runtime for each checker process. |
| `--checker-cleanup-timeout-seconds` | finite positive number | `1` | Maximum time spent cleaning up a checker after timeout or cancellation. |
| `--checker-max-concurrency` | positive integer | `1` | Maximum number of checker processes running concurrently. |
| `--checker-max-output-bytes` | positive integer | `8388608` | Maximum combined buffered stdout+stderr bytes per checker process; larger output fails closed. |
| `--checker-max-diagnostics` | positive integer | `10000` | Maximum diagnostics per checker result and in the merged validator output. |
| `--checker-max-field-length` | positive integer | `4096` | Maximum characters in one externally reported diagnostic field (message, path, code). |

## Resource limits

Limits are opt-in. Leaving a flag unset does not impose a limit on that
resource. For example, this invocation allows up to 100,000 discovery
entries, 8 MiB per source file, and 64 MiB of source reads:

```bash
einf-validate src/ --max-discovery-entries 100000 \
  --max-file-bytes 8388608 --max-total-bytes 67108864
```

Discovery counts each explicit target attempt and every encountered directory
entry, including directories and non-Python files. This bounds traversal work,
not just the number of Python files found. Repeated target attempts consume
discovery work, but normalized duplicate source paths are read and analyzed once.
Directory symlinks are still followed, including links outside the requested
root; cycles are detected. These limits are not a workspace sandbox.

Byte limits count raw input bytes, before UTF-8 decoding or newline conversion.
Bytes read before a decoding or I/O failure still consume the total allowance.
File size is checked before reading, and bounded reads also detect growth after
that check. A read may consume one extra byte to detect an overflow before
stopping. When either byte limit is enabled, source inputs must be regular files.

Exceeding a limit stops the invocation and prevents external checkers from
running. The JSON report includes `limit_failures[]`, and the exit code is `1`.
Completed file reports may remain, but they are not evidence that the whole
request was analyzed. Discovery-limit failures identify the requested root;
byte-limit failures identify the source file.
The `observed` count can come from a file-size precheck, without reading the
file, or from actual bytes read.

The limits cover validator discovery and source capture, not process memory,
filesystem syscall duration, parser execution time, or work inside an external
checker, including its imported files. Use the separate `--checker-*` options
for checker process and output limits. Source byte admission still runs when
the semantic parser is unavailable.

Programmatic callers can pass the same policy:

```python
from pathlib import Path

from einf.analysis.parser import AstParserBackend
from einf.analysis.validator import ValidationLimits, run_validation

report = run_validation(
    targets=(Path("src"),),
    parser_backend=AstParserBackend(),
    limits=ValidationLimits(max_discovery_entries=100000, max_total_bytes=67108864),
)
```

`limits=None` and `ValidationLimits()` both leave every resource unrestricted.

## Exit codes

1. `0` — no diagnostics, no discovery/parse/read failures, no checker
   invocation failures.
2. `1` — any file contains semantic or checker diagnostics, parse failures,
   or read failures; any target directory cannot be traversed completely; or
   any configured checker failed to execute; or an opt-in resource limit was
   exceeded.

## Output contract

`einf-validate` writes stable JSON to stdout. Top-level fields:

- `schema_version` — output contract version,
- `parser_backend` — the parser that produced the report,
- `checker_failures[]` — external checker invocation failures
  (unavailable executable, spawn failure, timeout, malformed output, etc.),
- `discovery_failures[]` — directories that could not be traversed completely,
- `limit_failures[]` — quotas that stopped validation, each with `resource`,
  `limit`, `observed`, and `path` fields,
- `files[]` — per-file report.

Each file entry:

- `path`
- `diagnostics[]` — `einf` semantic diagnostics,
- `checker_diagnostics[]` — normalized external checker diagnostics,
- `axis_tokens[]` — recognized axis identifier tokens with separate `kind`,
  `side`, `relation`, and nullable operation `role` fields,
- `failures[]` — validator ingress failures (unreadable files,
  parse errors).

The current output schema version is `0.4`. Consumers of `0.3` must accept the
new `limit_failures` field, which is empty when no quota was exceeded. Axis
token `kind` distinguishes scalar axes from axis packs; `relation` records
whether the same structural symbol appears on both sides of the operation.

Diagnostic spans use one-based lines and zero-based Python character columns,
not byte offsets or UTF-16 units. Pyright and BasedPyright ranges are converted
using the saved source file. Keep the file unchanged while its checker runs;
external checking does not capture an atomic snapshot of concurrent edits.

If Pyright or BasedPyright omits its range, the diagnostic has `span: null`.
If its source cannot be read or decoded, the diagnostic text is retained with
`span: null` and a checker failure. Malformed or out-of-bounds ranges are
rejected as malformed checker output.

Position conversion accepts regular source files up to 8 MiB and shares an
8 MiB source-read budget across one checker result. It rejects FIFOs, devices
and over-budget sources, retaining their diagnostic text with `span: null`
and a checker failure. These are byte and memory bounds, not a deadline for
filesystem syscalls; the checker timeout still applies only to the subprocess.

## CI integration

The CLI is designed for CI gates:

```yaml
# example: pre-merge lint step
- run: einf-validate src/ --checker basedpyright
```

The stable JSON + exit-code contract means downstream tooling can pipe
stdout into a diff viewer or annotator without parsing unstable text.
