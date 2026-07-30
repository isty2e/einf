import argparse
import subprocess
import tempfile
import venv
from pathlib import Path

_BASE_INSTALL_CASE = """
import numpy as np

from einf import ax, axes, rearrange

b, n, d = axes("b", "n", "d")
transpose = rearrange(ax[b, n, d], ax[b, d, n])
result = transpose(np.zeros((2, 3, 4), dtype=np.float32))
assert result.shape == (2, 4, 3)
"""


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


def _verify_wheel_base_install(wheel: Path) -> None:
    with tempfile.TemporaryDirectory(
        prefix="einf-wheel-base-install-"
    ) as temporary_dir:
        root = Path(temporary_dir)
        environment = root / "venv"
        venv.EnvBuilder(with_pip=True).create(environment)
        python = environment / "bin" / "python"

        _run(
            [
                str(python),
                "-m",
                "pip",
                "install",
                "numpy>=1.26",
                str(wheel),
            ],
            cwd=root,
        )
        _run([str(python), "-c", _BASE_INSTALL_CASE], cwd=root)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check base wheel execution with an explicit runtime backend."
    )
    parser.add_argument("wheel", type=Path)
    arguments = parser.parse_args()
    _verify_wheel_base_install(_resolve_wheel(arguments.wheel))


if __name__ == "__main__":
    main()
