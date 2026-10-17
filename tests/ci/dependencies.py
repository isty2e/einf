"""Resolve, verify, and prepare CI dependency snapshots."""

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
import tomllib
import venv
from datetime import UTC, datetime
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
SNAPSHOT_PROFILES = ("build", *PROFILE_EXTRAS)
RESOLUTION_INPUTS = ("pyproject.toml", "tests/ci/dependencies.py")


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
    root: Path,
    profile: str,
    wheelhouse: Path,
    github_env: Path,
    *,
    bundle: Path | None = None,
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
    bundle : Path or None, optional
        Fresh-resolution bundle for this checkout. None uses reviewed snapshots.

    Raises
    ------
    FileExistsError
        The wheelhouse already exists and could contain unreviewed wheels.
    ValueError
        Bundle evidence does not match, wheel metadata is incomplete, or the
        Torch role is not CPU-only.
    OSError
        A required bundle file cannot be read.
    subprocess.CalledProcessError
        Git discovery, hash verification, downloading, building, or bootstrapping fails.
    subprocess.TimeoutExpired
        Git discovery, downloading, building, or bootstrapping exceeds its deadline.
    """
    if bundle is not None:
        validate_bundle(root, bundle)

    snapshots = bundle if bundle is not None else root / ".github" / "requirements"
    snapshot = snapshots / f"{profile}.txt"
    build_snapshot = snapshots / "build.txt"
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


def refresh_snapshots(root: Path, destination: Path | None = None) -> None:
    """Resolve new CI snapshots without changing the package's allowed ranges.

    Parameters
    ----------
    root : Path
        Checkout whose package metadata supplies requirement ranges.
    destination : Path or None, optional
        Output directory. None refreshes the checkout's reviewed snapshots.

    Raises
    ------
    subprocess.CalledProcessError
        uv cannot resolve a role's requirements.
    subprocess.TimeoutExpired
        Resolving a role exceeds its deadline.
    """
    if destination is None:
        destination = root / ".github" / "requirements"

    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        for profile in SNAPSHOT_PROFILES:
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


def checkout_revision(root: Path) -> str:
    """Read the commit checked out by the workflow.

    Parameters
    ----------
    root : Path
        Git checkout being tested.

    Returns
    -------
    str
        Full commit object identifier.

    Raises
    ------
    subprocess.CalledProcessError
        Git cannot identify the checkout.
    subprocess.TimeoutExpired
        Git does not respond within ten seconds.
    """
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True, timeout=10
    ).strip()


def bundle_hashes(root: Path, directory: Path) -> dict[str, dict[str, str]]:
    """Fingerprint the resolution inputs and complete role snapshot set.

    Parameters
    ----------
    root : Path
        Checkout supplying package and resolver inputs.
    directory : Path
        Directory containing all seven requirements snapshots.

    Returns
    -------
    dict[str, dict[str, str]]
        Input and snapshot names mapped to SHA-256 digests. Text inputs use
        normalized newlines so Windows and POSIX checkouts agree.

    Raises
    ------
    OSError
        A required input or snapshot cannot be read.
    """
    return {
        "inputs": {
            name: hashlib.sha256(
                (root / name).read_text(encoding="utf-8").encode("utf-8")
            ).hexdigest()
            for name in RESOLUTION_INPUTS
        },
        "snapshots": {
            f"{profile}.txt": hashlib.sha256(
                (directory / f"{profile}.txt").read_bytes()
            ).hexdigest()
            for profile in SNAPSHOT_PROFILES
        },
    }


def resolve_bundle(root: Path, destination: Path) -> None:
    """Resolve a fresh, source-bound dependency bundle.

    Parameters
    ----------
    root : Path
        Checkout supplying package and CI role requirements.
    destination : Path
        New directory for snapshots and resolution evidence.

    Raises
    ------
    FileExistsError
        The destination already exists.
    OSError
        An input or output file cannot be accessed.
    subprocess.CalledProcessError
        Git, uv version discovery, or dependency resolution fails.
    subprocess.TimeoutExpired
        A discovery or resolution command exceeds its deadline.
    """
    destination.mkdir(parents=True, exist_ok=False)
    source_sha = checkout_revision(root)
    resolver = subprocess.check_output(
        ["uv", "--version"], text=True, timeout=10
    ).strip()
    refresh_snapshots(root, destination)
    receipt = {
        "schema_version": "v1",
        "source_sha": source_sha,
        "resolved_at": datetime.now(UTC).isoformat(),
        "resolver": resolver,
        "python": platform.python_version(),
        "platform": sys.platform,
        "machine": platform.machine(),
        "hashes": bundle_hashes(root, destination),
    }
    (destination / "resolution.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def validate_bundle(root: Path, directory: Path) -> None:
    """Reject a bundle not bound to this checkout and its resolution inputs.

    Parameters
    ----------
    root : Path
        Checkout whose code will consume the dependency bundle.
    directory : Path
        Downloaded complete bundle and resolution evidence.

    Raises
    ------
    ValueError
        The receipt is invalid or its source, inputs, or snapshots do not match.
    OSError
        Required resolution evidence cannot be read.
    subprocess.CalledProcessError
        Git cannot identify the checkout.
    subprocess.TimeoutExpired
        Git does not respond within ten seconds.
    """
    receipt = json.loads((directory / "resolution.json").read_text(encoding="utf-8"))
    if not isinstance(receipt, dict) or receipt.get("schema_version") != "v1":
        raise ValueError("unsupported dependency resolution receipt")

    if receipt.get("source_sha") != checkout_revision(root):
        raise ValueError("dependency bundle source does not match the checkout")

    if receipt.get("hashes") != bundle_hashes(root, directory):
        raise ValueError("dependency bundle input or snapshot hashes do not match")


def main() -> None:
    """Run the selected CI dependency command."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("profile", choices=tuple(PROFILE_EXTRAS))
    prepare.add_argument("--bundle", type=Path)
    commands.add_parser("refresh")
    resolve = commands.add_parser("resolve")
    resolve.add_argument("destination", type=Path)
    arguments = parser.parse_args()
    root = Path(__file__).resolve().parents[2]

    if arguments.command == "refresh":
        refresh_snapshots(root)
    elif arguments.command == "resolve":
        resolve_bundle(root, arguments.destination)
    else:
        prepare_dependencies(
            root,
            arguments.profile,
            Path(os.environ["RUNNER_TEMP"]) / "einf-ci-wheels",
            Path(os.environ["GITHUB_ENV"]),
            bundle=arguments.bundle,
        )


if __name__ == "__main__":
    main()
