"""Prepare fixed CI dependencies or refresh their reviewed snapshots."""

import argparse
import os
import re
import subprocess
import sys
import tempfile
import tomllib
import venv
from email.parser import Parser
from pathlib import Path
from types import MappingProxyType
from zipfile import ZipFile

PROFILE_EXTRAS = MappingProxyType(
    {
        "base": (),
        # The core job's isolated artifact checks also install LSP extras.
        "core": ("dev", "analysis", "docs", "lsp"),
        "checkers": ("dev",),
        "windows": (),
        "lsp": ("dev", "analysis", "lsp"),
        "torch": ("dev", "analysis", "lsp"),
    }
)
PROFILE_TOOLS = MappingProxyType(
    {
        "base": ("pytest",),
        "checkers": ("pyright", "zuban"),
        "windows": ("build", "pytest", "pyrefly"),
        "torch": ("torch",),
    }
)
CPU_INDEX = "https://download.pytorch.org/whl/cpu"
BUILD_REQUIREMENTS = ("pip>=25.3", "setuptools", "wheel", "Cython")


def profile_requirements(root: Path, profile: str) -> tuple[str, ...]:
    """Read package/build requirements and add the selected CI role's tools.

    Parameters
    ----------
    root : Path
        Checkout containing the canonical package metadata.
    profile : str
        Existing CI role from ``PROFILE_EXTRAS``.

    Returns
    -------
    tuple[str, ...]
        Package, build-hook, and role-specific requirement specifiers.
    """
    with (root / "pyproject.toml").open("rb") as source:
        metadata = tomllib.load(source)

    requirements = [
        *metadata["project"]["dependencies"],
        *metadata["build-system"]["requires"],
        "pip>=25.3",
        # Hatchling declares this through its editable-build hook.
        "editables~=0.3",
    ]
    for extra in PROFILE_EXTRAS[profile]:
        requirements.extend(metadata["project"]["optional-dependencies"][extra])

    requirements.extend(PROFILE_TOOLS.get(profile, ()))
    return tuple(requirements)


def wheel_constraints(wheelhouse: Path) -> dict[str, str]:
    """Read exact distribution versions from downloaded wheel metadata.

    Parameters
    ----------
    wheelhouse : Path
        Prepared wheels, downloaded or built from reviewed inputs.

    Returns
    -------
    dict[str, str]
        Normalized distribution names mapped to exact versions.

    Raises
    ------
    ValueError
        A wheel has incomplete metadata, distributions repeat, or pip is absent.
    zipfile.BadZipFile
        A downloaded wheel is not a readable ZIP archive.
    """
    versions: dict[str, str] = {}
    for wheel in sorted(wheelhouse.glob("*.whl")):
        with ZipFile(wheel) as archive:
            metadata_files = [
                name
                for name in archive.namelist()
                if name.count("/") == 1 and name.endswith(".dist-info/METADATA")
            ]
            if len(metadata_files) != 1:
                raise ValueError(f"wheel must contain one METADATA file: {wheel.name}")

            metadata = Parser().parsestr(
                archive.read(metadata_files[0]).decode("utf-8")
            )

        name = metadata.get("Name")
        version = metadata.get("Version")
        if not name or not version:
            raise ValueError(f"wheel has no distribution name/version: {wheel.name}")

        name = re.sub(r"[-_.]+", "-", name).lower()
        if name in versions:
            raise ValueError(f"duplicate wheel distribution: {name}")

        versions[name] = version

    if "pip" not in versions:
        raise ValueError("dependency snapshot must include pip")

    return versions


def prepare_dependencies(
    root: Path, profile: str, wheelhouse: Path, github_env: Path
) -> None:
    """Assemble the role's wheels, then activate offline constrained installs.

    Parameters
    ----------
    root : Path
        Checkout containing the reviewed requirements snapshots.
    profile : str
        CI role selecting one requirements snapshot.
    wheelhouse : Path
        New directory retained for the lifetime of the CI job.
    github_env : Path
        GitHub Actions environment file, appended only after preparation succeeds.

    Raises
    ------
    FileExistsError
        The wheelhouse already exists and could contain unreviewed wheels.
    ValueError
        Wheel metadata is incomplete or the Torch role is not CPU-only.
    subprocess.CalledProcessError
        Hash verification, downloading, building, or bootstrapping fails.
    subprocess.TimeoutExpired
        Downloading, building, or installer bootstrapping exceeds its deadline.
    """
    snapshot = root / ".github" / "requirements" / f"{profile}.txt"
    build_snapshot = root / ".github" / "requirements" / "build.txt"
    wheelhouse.mkdir(parents=True, exist_ok=False)
    build_wheels = wheelhouse / "build-wheels"
    build_wheels.mkdir()
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "download",
            "--disable-pip-version-check",
            "--require-hashes",
            "--only-binary=:all:",
            "--no-deps",
            "--dest",
            str(build_wheels),
            "--requirement",
            str(build_snapshot),
        ],
        check=True,
        timeout=600,
    )
    wheel_constraints(build_wheels)
    build_directory = wheelhouse / "build-environment"
    venv.EnvBuilder(with_pip=True).create(build_directory)
    build_python = build_directory / (
        "Scripts/python.exe" if os.name == "nt" else "bin/python"
    )
    build_environment = {
        **os.environ,
        "PIP_NO_INDEX": "1",
        "PIP_FIND_LINKS": build_wheels.resolve().as_uri(),
        "PIP_DISABLE_PIP_VERSION_CHECK": "1",
        "PIP_NO_CACHE_DIR": "1",
    }
    subprocess.run(
        [
            str(build_python),
            "-m",
            "pip",
            "install",
            "--ignore-installed",
            "--require-hashes",
            "-r",
            str(build_snapshot),
        ],
        env=build_environment,
        check=True,
        timeout=120,
    )
    subprocess.run(
        [
            str(build_python),
            "-m",
            "pip",
            "wheel",
            "--require-hashes",
            "--no-deps",
            "--no-build-isolation",
            "--check-build-dependencies",
            "--wheel-dir",
            str(wheelhouse),
            "--requirement",
            str(snapshot),
        ],
        env={
            **os.environ,
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
            "PIP_NO_CACHE_DIR": "1",
        },
        check=True,
        timeout=600,
    )
    versions = wheel_constraints(wheelhouse)
    if profile == "torch" and not versions.get("torch", "").endswith("+cpu"):
        raise ValueError("torch CI requires a CPU-only torch distribution")

    constraints = wheelhouse / "constraints.txt"
    constraints.write_text(
        "".join(f"{name}=={version}\n" for name, version in sorted(versions.items())),
        encoding="utf-8",
    )
    environment = {
        "PIP_NO_INDEX": "1",
        "PIP_FIND_LINKS": wheelhouse.resolve().as_uri(),
        "PIP_CONSTRAINT": constraints.resolve().as_uri(),
        "PIP_BUILD_CONSTRAINT": constraints.resolve().as_uri(),
        "PIP_DISABLE_PIP_VERSION_CHECK": "1",
        "PIP_NO_CACHE_DIR": "1",
    }
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--ignore-installed",
            "--upgrade",
            "pip",
        ],
        env={**os.environ, **environment},
        check=True,
        timeout=120,
    )

    with github_env.open("a", encoding="utf-8", newline="\n") as output:
        output.writelines(f"{name}={value}\n" for name, value in environment.items())


def refresh_snapshots(root: Path) -> None:
    """Resolve new CI snapshots without changing the package's allowed ranges.

    Parameters
    ----------
    root : Path
        Checkout whose package metadata supplies requirement ranges.

    Raises
    ------
    subprocess.CalledProcessError
        uv cannot resolve a role's requirements.
    subprocess.TimeoutExpired
        Resolving a role exceeds its deadline.
    """
    destination = root / ".github" / "requirements"
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        for profile in ("build", *PROFILE_EXTRAS):
            requirements = directory / f"{profile}.in"
            specifiers = (
                BUILD_REQUIREMENTS
                if profile == "build"
                else profile_requirements(root, profile)
            )
            requirements.write_text(
                "\n".join(specifiers) + "\n",
                encoding="utf-8",
            )
            snapshot = directory / f"{profile}.txt"
            command = [
                "uv",
                "pip",
                "compile",
                str(requirements),
                "--python-version",
                "3.11",
                "--generate-hashes",
                "--no-header",
                "--no-annotate",
                "--quiet",
                "--output-file",
                str(snapshot),
            ]
            if profile == "torch":
                command.extend(["--python-platform", "linux", "--torch-backend", "cpu"])
            else:
                command.append("--universal")

            subprocess.run(command, check=True, timeout=600)
            content = snapshot.read_text(encoding="utf-8")
            if profile == "torch":
                content = f"--extra-index-url {CPU_INDEX}\n\n{content}"

            (destination / f"{profile}.txt").write_text(content, encoding="utf-8")


def main() -> None:
    """Run the selected CI dependency command."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("profile", choices=tuple(PROFILE_EXTRAS))
    commands.add_parser("refresh")
    arguments = parser.parse_args()
    root = Path(__file__).resolve().parents[2]

    if arguments.command == "refresh":
        refresh_snapshots(root)
    else:
        prepare_dependencies(
            root,
            arguments.profile,
            Path(os.environ["RUNNER_TEMP"]) / "einf-ci-wheels",
            Path(os.environ["GITHUB_ENV"]),
        )


if __name__ == "__main__":
    main()
