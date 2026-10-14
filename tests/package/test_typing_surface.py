import json
import os
import shutil
import sys
from pathlib import Path
from subprocess import CompletedProcess, run

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
ANNOTATION_CASE = REPO_ROOT / "tests/package/typing_cases/tensorop_annotations.py"
_DUMMY_TENSOR_SNIPPET = """
from collections.abc import Iterator
from types import EllipsisType
from typing import Self, TypeAlias
from typing_extensions import override

IndexKey: TypeAlias = int | slice | EllipsisType | None | tuple[int | slice | EllipsisType | None, ...]

class DummyDType:
    @override
    def __repr__(self) -> str:
        return "dummy"

    @override
    def __eq__(self, other: object) -> bool:
        return isinstance(other, DummyDType)

    @override
    def __hash__(self) -> int:
        return 0

class DummyTensor:
    shape: tuple[int, ...]
    ndim: int
    size: int
    dtype: DummyDType

    def __init__(self, shape: tuple[int, ...]) -> None:
        self.shape = shape
        self.ndim = len(shape)
        self.size = 1
        self.dtype = DummyDType()

    @property
    def T(self) -> Self:
        return self

    def __bool__(self) -> bool:
        return True

    def __complex__(self) -> complex:
        return complex(0.0)

    def __float__(self) -> float:
        return 0.0

    def __index__(self) -> int:
        return 0

    def __int__(self) -> int:
        return 0

    def __iter__(self) -> Iterator[Self]:
        return iter(())

    def __getitem__(self, key: IndexKey, /) -> Self:
        return self

    def __setitem__(self, key: IndexKey, value: Self | complex, /) -> None:
        return None

    def __xor__(self, other: Self | int, /) -> Self:
        return self
"""


@pytest.fixture
def basedpyright_bin() -> str:
    """Resolve the basedpyright binary for typing-surface checks."""
    binary = shutil.which("basedpyright")
    if binary is None:
        pytest.skip("basedpyright is not installed")
    return binary


@pytest.fixture
def pyrefly_bin() -> str:
    binary = shutil.which("pyrefly")
    if binary is None:
        pytest.skip("pyrefly is not installed")
    return binary


def _run_pyrefly(
    binary: str,
    source_path: Path,
    tmp_path: Path,
    *,
    config: Path | None = None,
) -> dict[str, object]:
    if config is None:
        config = tmp_path / "pyrefly.toml"
        config.write_text("", encoding="utf-8")
    result = run(
        [
            binary,
            "check",
            "--config",
            str(config),
            "--threads",
            "1",
            "--python-interpreter-path",
            sys.executable,
            "--output-format",
            "json",
            str(source_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode in (0, 1), result.stderr
    output = json.loads(result.stdout)
    assert isinstance(output, dict), result.stdout
    assert (result.returncode == 0) == (output["errors"] == []), output
    return output


def test_repository_metadata_supports_frozen_steps_and_mutable_adapters(
    pyrefly_bin: str, tmp_path: Path
) -> None:
    source_path = REPO_ROOT / "tests/package/typing_cases/repository_metadata.py"
    output = _run_pyrefly(
        pyrefly_bin, source_path, tmp_path, config=REPO_ROOT / "pyproject.toml"
    )
    assert output["errors"] == [], output

    executed = run(
        [sys.executable, "-I", str(source_path)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert executed.returncode == 0, executed.stderr


@pytest.mark.parametrize(
    ("import_path", "type_name", "fields"),
    [
        (
            "einf.steps.base",
            "RuntimeStep[StepProgram]",
            ("name", "input_arity", "output_arity", "program"),
        ),
        (
            "einf.steps.base",
            "SymbolicStep[StepProgram]",
            ("name", "input_arity", "output_arity", "program"),
        ),
        ("einf.analysis.checkers", "CheckerAdapter", ("name", "executable")),
        (
            "einf.steps.einsum",
            "EinsumSymbolicProgram",
            ("input_arity", "output_arity", "allow_native_matmul"),
        ),
    ],
)
def test_repository_metadata_is_read_only_to_base_consumers(
    pyrefly_bin: str,
    tmp_path: Path,
    import_path: str,
    type_name: str,
    fields: tuple[str, ...],
) -> None:
    source_path = tmp_path / "metadata_assignment.py"
    assignments = "\n".join(f"    value.{field} = value.{field}" for field in fields)
    source_path.write_text(
        "from einf.steps.base import StepProgram\n"
        f"from {import_path} import {type_name.split('[')[0]}\n"
        f"def mutate(value: {type_name}) -> None:\n{assignments}\n",
        encoding="utf-8",
    )
    output = _run_pyrefly(
        pyrefly_bin, source_path, tmp_path, config=REPO_ROOT / "pyproject.toml"
    )
    errors = output["errors"]
    assert isinstance(errors, list) and len(errors) == len(fields), output
    assert all(
        isinstance(error, dict) and error["name"] == "read-only" for error in errors
    ), output


def test_precise_annotations_preserve_call_configuration_and_callable_contracts(
    pyrefly_bin: str, tmp_path: Path
) -> None:
    output = _run_pyrefly(pyrefly_bin, ANNOTATION_CASE, tmp_path)
    assert output["errors"] == [], output


@pytest.mark.parametrize(
    ("source", "error_kind"),
    [
        (
            "bad: TensorOp[[Dummy, Dummy], Dummy] = rearrange(ax[b], ax[b])",
            "bad-assignment",
        ),
        (
            "bad: TensorOp[[Dummy], tuple[Dummy, Dummy]] = rearrange(ax[b], ax[b])",
            "bad-assignment",
        ),
        (
            "bad: ReducibleTensorOp[[Dummy], Dummy] = rearrange(ax[b], ax[b])",
            "bad-assignment",
        ),
        (
            "def bad(op: TensorOp[[Dummy], Dummy]) -> None:\n    op()",
            "bad-argument-count",
        ),
        (
            "def bad(op: TensorOp[[Dummy], Dummy]) -> None:\n    op(Dummy(), Dummy())",
            "bad-argument-count",
        ),
        (
            "def bad(op: TensorOp[[Dummy], Dummy]) -> None:\n    op(Dummy(), tensor=Dummy())",
            "unexpected-keyword",
        ),
        (
            'def bad(op: TensorOp[[Dummy], Dummy]) -> None:\n    op.name = "other"',
            "read-only",
        ),
        (
            'def bad(op: TensorOp[[Dummy], Dummy]) -> None:\n    op.with_sizes(b=2).reduce_by("sum")',
            "missing-attribute",
        ),
    ],
)
def test_precise_annotations_reject_invalid_contracts(
    pyrefly_bin: str, tmp_path: Path, source: str, error_kind: str
) -> None:
    snippet = tmp_path / "invalid_annotation.py"
    snippet.write_text(
        "from einf import TensorOp, ReducibleTensorOp, ax, axes, rearrange\n"
        'b, = axes("b")\n'
        "class Dummy:\n"
        "    shape: tuple[int, ...] = (2,)\n"
        "    def __getitem__(self, key: object, /) -> Dummy:\n"
        "        return self\n" + source + "\n",
        encoding="utf-8",
    )
    output = _run_pyrefly(pyrefly_bin, snippet, tmp_path)
    errors = output["errors"]
    assert isinstance(errors, list) and errors, output
    assert all(
        isinstance(error, dict) and error["name"] == error_kind for error in errors
    ), output


@pytest.fixture
def pyright_env() -> dict[str, str]:
    """Build deterministic environment for the external type-check process."""
    env = dict(os.environ)
    src_path = str(REPO_ROOT / "src")
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = src_path if not existing else f"{src_path}:{existing}"
    env["LANG"] = "C"
    env["LC_ALL"] = "C"
    return env


def _run_basedpyright(
    *,
    binary: str,
    source: str,
    tmp_path: Path,
    env: dict[str, str],
    expect_success: bool = True,
) -> dict[str, object]:
    snippet_path = tmp_path / "typing_case.py"
    snippet_path.write_text(source, encoding="utf-8")

    result: CompletedProcess[str] = run(
        [binary, str(snippet_path), "--outputjson"],
        check=False,
        text=True,
        capture_output=True,
        env=env,
    )
    output = json.loads(result.stdout)

    if expect_success and result.returncode != 0:
        raise AssertionError(
            f"basedpyright expected success but failed: code={result.returncode}, output={result.stdout}"
        )

    return output


def _collect_revealed_types(output: dict[str, object]) -> tuple[str, ...]:
    diagnostics = output.get("generalDiagnostics")
    if not isinstance(diagnostics, list):
        pytest.fail("basedpyright output is missing diagnostics list")

    revealed: list[str] = []
    for diagnostic in diagnostics:
        if not isinstance(diagnostic, dict):
            continue
        message = diagnostic.get("message")
        if isinstance(message, str) and message.startswith('Type of "'):
            revealed.append(message)
    return tuple(revealed)


def _collect_error_messages(output: dict[str, object]) -> tuple[str, ...]:
    diagnostics = output.get("generalDiagnostics")
    if not isinstance(diagnostics, list):
        pytest.fail("basedpyright output is missing diagnostics list")

    errors: list[str] = []
    for diagnostic in diagnostics:
        if not isinstance(diagnostic, dict):
            continue

        severity = diagnostic.get("severity")
        message = diagnostic.get("message")
        if severity == "error" and isinstance(message, str):
            errors.append(message)

    return tuple(errors)


def test_typing_surface_exact_arity_overloads(
    basedpyright_bin: str, pyright_env: dict[str, str], tmp_path: Path
) -> None:
    output = _run_basedpyright(
        binary=basedpyright_bin,
        env=pyright_env,
        tmp_path=tmp_path,
        source=f"""
from einf import ax, axes, contract, rearrange

b, n, m, d = axes("b", "n", "m", "d")

fn = rearrange(ax[b, n, d], ax[b, n, d])
fn_split = rearrange(ax[b, (n + m), d], (ax[b, n, d], ax[b, m, d]))
fn_concat = contract((ax[b, n, d], ax[b, m, d]), ax[b, n + m, d])

{_DUMMY_TENSOR_SNIPPET}

x = DummyTensor((3, 5, 2))
y = fn(x)
y_split = fn_split(x)

reveal_type(y)
reveal_type(y_split)
reveal_type(fn_concat(x, x))
""",
    )

    messages = _collect_revealed_types(output)
    assert 'Type of "y" is "DummyTensor"' in messages
    assert 'Type of "y_split" is "tuple[DummyTensor, DummyTensor]"' in messages
    assert 'Type of "fn_concat(x, x)" is "DummyTensor"' in messages


def test_typing_surface_fallback_quadrants(
    basedpyright_bin: str, pyright_env: dict[str, str], tmp_path: Path
) -> None:
    output = _run_basedpyright(
        binary=basedpyright_bin,
        env=pyright_env,
        tmp_path=tmp_path,
        source=f"""
from einf import ax, axes, rearrange

a, b, c, d, e, f, g = axes("a", "b", "c", "d", "e", "f", "g")

op_1n = rearrange(ax[a], (ax[a], ax[b], ax[c], ax[d], ax[e], ax[f], ax[g]))
op_n1 = rearrange((ax[a], ax[b], ax[c], ax[d], ax[e], ax[f], ax[g]), ax[a])
op_nn = rearrange(
    (ax[a], ax[b], ax[c], ax[d], ax[e], ax[f], ax[g]),
    (ax[a], ax[b], ax[c], ax[d], ax[e], ax[f], ax[g]),
)

{_DUMMY_TENSOR_SNIPPET}

x = DummyTensor((1,))
out_1n = op_1n(x)
out_n1 = op_n1(x, x, x, x, x, x, x)
out_nn = op_nn(x, x, x, x, x, x, x)

reveal_type(out_1n)
reveal_type(out_n1)
reveal_type(out_nn)
""",
    )

    messages = _collect_revealed_types(output)
    assert 'Type of "out_1n" is "tuple[DummyTensor, ...]"' in messages
    assert 'Type of "out_n1" is "DummyTensor"' in messages
    assert 'Type of "out_nn" is "tuple[DummyTensor, ...]"' in messages


def test_typing_surface_disallows_reduce_by_on_non_reduce_ops(
    basedpyright_bin: str, pyright_env: dict[str, str], tmp_path: Path
) -> None:
    output = _run_basedpyright(
        binary=basedpyright_bin,
        env=pyright_env,
        tmp_path=tmp_path,
        expect_success=False,
        source=f"""
from einf import ax, axes, reduce, view

b, c = axes("b", "c")

bad = view(ax[b], ax[b]).reduce_by("sum")
ok = reduce(ax[b, c], ax[b]).reduce_by("sum")

{_DUMMY_TENSOR_SNIPPET}

reveal_type(ok)
""",
    )

    error_messages = _collect_error_messages(output)
    assert any("reduce_by" in message for message in error_messages)


def test_exact_arity_result_is_a_public_tensor_op(
    basedpyright_bin: str, pyright_env: dict[str, str], tmp_path: Path
) -> None:
    output = _run_basedpyright(
        binary=basedpyright_bin,
        env=pyright_env,
        tmp_path=tmp_path,
        source="""
from collections.abc import Callable

from einf import TensorOp, ax, axes, rearrange

b, n = axes("b", "n")
exact = rearrange(ax[b, n], ax[n, b])
public: TensorOp = exact
callable_op: Callable[..., object] = public

reveal_type(public.name)
reveal_type(public.sizes)
reveal_type(public.sizes_items)
reveal_type(public.supports_reducer)
""",
    )

    messages = _collect_revealed_types(output)
    assert 'Type of "public.name" is "str"' in messages
    assert 'Type of "public.sizes" is "dict[str, int]"' in messages
    assert 'Type of "public.sizes_items" is "tuple[tuple[str, int], ...]"' in messages
    assert 'Type of "public.supports_reducer" is "bool"' in messages


def test_public_tensor_op_metadata_is_read_only(
    basedpyright_bin: str, pyright_env: dict[str, str], tmp_path: Path
) -> None:
    output = _run_basedpyright(
        binary=basedpyright_bin,
        env=pyright_env,
        tmp_path=tmp_path,
        expect_success=False,
        source="""
from einf import TensorOp

def mutate(op: TensorOp) -> None:
    op.name = "other"
    op.sizes = {}
    op.supports_reducer = False
""",
    )

    error_messages = _collect_error_messages(output)
    for property_name in ("name", "sizes", "supports_reducer"):
        assert any(property_name in message for message in error_messages)
