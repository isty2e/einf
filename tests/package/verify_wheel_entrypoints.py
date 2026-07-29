import argparse
import subprocess
import tempfile
import venv
from pathlib import Path

_LSP_INSTALL_ERROR = (
    "einf-lsp requires optional LSP dependencies; install them with "
    "'pip install \"einf[lsp]\"'"
)


def _resolve_wheel(path: Path) -> Path:
    if path.is_file() and path.suffix == ".whl":
        return path.resolve()
    if path.is_dir():
        wheels = tuple(path.glob("*.whl"))
        if len(wheels) == 1:
            return wheels[0].resolve()
    raise ValueError(f"expected one wheel file: {path}")


def _run(command: list[str], *, cwd: Path) -> None:
    subprocess.run(command, cwd=cwd, check=True)


def _run_entrypoint(
    command: list[str], *, cwd: Path
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
    )


def _raise_unexpected_result(
    *,
    stage: str,
    result: subprocess.CompletedProcess[str],
) -> None:
    raise RuntimeError(
        f"{stage} failed with exit code {result.returncode}\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )


def _verify_wheel_entrypoints(wheel: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="einf-wheel-entrypoints-") as temporary_dir:
        root = Path(temporary_dir)
        environment = root / "venv"
        venv.EnvBuilder(with_pip=True).create(environment)
        python = environment / "bin" / "python"
        einf_lsp = environment / "bin" / "einf-lsp"

        _run(
            [str(python), "-m", "pip", "install", str(wheel)],
            cwd=root,
        )
        base_result = _run_entrypoint([str(einf_lsp), "--help"], cwd=root)
        if (
            base_result.returncode != 1
            or base_result.stdout
            or base_result.stderr != f"{_LSP_INSTALL_ERROR}\n"
        ):
            _raise_unexpected_result(
                stage="base einf-lsp smoke test", result=base_result
            )

        _run(
            [
                str(python),
                "-m",
                "pip",
                "install",
                f"einf[lsp] @ {wheel.as_uri()}",
            ],
            cwd=root,
        )
        lsp_result = _run_entrypoint([str(einf_lsp), "--help"], cwd=root)
        if (
            lsp_result.returncode != 0
            or "usage:" not in lsp_result.stdout
            or "--tcp" not in lsp_result.stdout
        ):
            _raise_unexpected_result(
                stage="einf-lsp extra smoke test",
                result=lsp_result,
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check console entry points from an installed wheel."
    )
    parser.add_argument("wheel", type=Path)
    arguments = parser.parse_args()
    _verify_wheel_entrypoints(_resolve_wheel(arguments.wheel))


if __name__ == "__main__":
    main()
