import argparse
import shutil
import subprocess
import tempfile
import venv
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
TYPING_CASE = PACKAGE_DIR / "typing_cases" / "tensorop_contract.py"
STUBTEST_ALLOWLIST = PACKAGE_DIR / "stubtest_allowlist.txt"


def _resolve_wheel(path: Path) -> Path:
    if path.is_file() and path.suffix == ".whl":
        return path.resolve()
    if path.is_dir():
        wheels = tuple(path.glob("*.whl"))
        if len(wheels) == 1:
            return wheels[0].resolve()
    raise ValueError(f"expected one wheel file: {path}")


def _require_executable(name: str) -> str:
    executable = shutil.which(name)
    if executable is None:
        raise RuntimeError(f"required typing checker is not installed: {name}")
    return executable


def _run(command: list[str], *, cwd: Path) -> None:
    subprocess.run(command, cwd=cwd, check=True)


def _verify_wheel(wheel: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="einf-wheel-typing-") as temporary_dir:
        root = Path(temporary_dir)
        environment = root / "venv"
        venv.EnvBuilder(with_pip=True, system_site_packages=True).create(environment)
        python = environment / "bin" / "python"

        _run(
            [
                str(python),
                "-m",
                "pip",
                "install",
                "--ignore-installed",
                "--no-deps",
                str(wheel),
            ],
            cwd=root,
        )
        _run(
            [
                str(python),
                "-c",
                (
                    "from pathlib import Path; import einf; "
                    "assert Path(einf.__file__).resolve().is_relative_to("
                    f"{str(environment.resolve())!r})"
                ),
            ],
            cwd=root,
        )

        typing_case = root / TYPING_CASE.name
        shutil.copy2(TYPING_CASE, typing_case)
        _run(
            [
                _require_executable("basedpyright"),
                "--pythonpath",
                str(python),
                str(typing_case),
            ],
            cwd=root,
        )
        _run(
            [
                _require_executable("mypy"),
                "--python-executable",
                str(python),
                "--no-incremental",
                str(typing_case),
            ],
            cwd=root,
        )
        _run(
            [
                _require_executable("ty"),
                "check",
                "--python",
                str(python),
                str(typing_case),
            ],
            cwd=root,
        )
        _run(
            [
                str(python),
                "-m",
                "mypy.stubtest",
                "einf.operations.api",
                "--concise",
                "--allowlist",
                str(STUBTEST_ALLOWLIST),
            ],
            cwd=root,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check the TensorOp typing contract from an installed wheel."
    )
    parser.add_argument("wheel", type=Path)
    arguments = parser.parse_args()
    _verify_wheel(_resolve_wheel(arguments.wheel))


if __name__ == "__main__":
    main()
