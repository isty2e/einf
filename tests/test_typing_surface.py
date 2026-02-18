import json
import os
import shutil
from pathlib import Path
from subprocess import CompletedProcess, run

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
_DUMMY_TENSOR_SNIPPET = """
from collections.abc import Iterator
from types import EllipsisType
from typing import TypeAlias
from typing_extensions import Self, override

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
def pyright_env() -> dict[str, str]:
    """Build environment with src path for external type-check process."""
    env = dict(os.environ)
    src_path = str(REPO_ROOT / "src")
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = src_path if not existing else f"{src_path}:{existing}"
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
        raise AssertionError("basedpyright output is missing diagnostics list")

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
        raise AssertionError("basedpyright output is missing diagnostics list")

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
reveal_type(fn_concat)
""",
    )

    messages = _collect_revealed_types(output)
    assert 'Type of "y" is "DummyTensor"' in messages
    assert 'Type of "y_split" is "tuple[DummyTensor, DummyTensor]"' in messages
    assert 'Type of "fn_concat" is "_TensorOp_2_1"' in messages


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

reveal_type(op_1n)
reveal_type(op_n1)
reveal_type(op_nn)
reveal_type(out_1n)
reveal_type(out_n1)
reveal_type(out_nn)
""",
    )

    messages = _collect_revealed_types(output)
    assert 'Type of "op_1n" is "_TensorOp_1_N"' in messages
    assert 'Type of "op_n1" is "_TensorOp_N_1"' in messages
    assert 'Type of "op_nn" is "_TensorOp_N_N"' in messages
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
