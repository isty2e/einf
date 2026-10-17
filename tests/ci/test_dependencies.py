import subprocess
import venv
from pathlib import Path
from zipfile import BadZipFile, ZipFile

import pytest

from tests.ci import dependencies


def _write_project(root: Path) -> None:
    (root / "pyproject.toml").write_text(
        """[build-system]
requires = ["build-backend>=2"]
[project]
dependencies = ["runtime>=1"]
[project.optional-dependencies]
dev = ["dev-tool>=3"]
analysis = ["parser>=4"]
docs = ["docs-tool>=5"]
lsp = ["transport>=6"]
""",
        encoding="utf-8",
    )


def _write_wheel(directory: Path, name: str, version: str) -> None:
    with ZipFile(directory / f"{name}-{version}-py3-none-any.whl", "w") as archive:
        archive.writestr(
            f"{name}-{version}.dist-info/METADATA",
            f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n",
        )


@pytest.mark.parametrize(
    ("profile", "extra_requirements"),
    [
        ("base", {"pytest"}),
        (
            "core",
            {"dev-tool>=3", "parser>=4", "docs-tool>=5", "transport>=6"},
        ),
        ("checkers", {"dev-tool>=3", "pyright", "zuban"}),
        ("windows", {"build", "pytest", "pyrefly"}),
        ("lsp", {"dev-tool>=3", "parser>=4", "transport>=6"}),
        ("torch", {"dev-tool>=3", "parser>=4", "transport>=6", "torch"}),
    ],
)
def test_profile_uses_package_metadata_and_only_required_role_tools(
    tmp_path: Path, profile: str, extra_requirements: set[str]
) -> None:
    _write_project(tmp_path)

    assert set(dependencies.profile_requirements(tmp_path, profile)) == {
        "runtime>=1",
        "build-backend>=2",
        "pip>=25.3",
        "editables~=0.3",
        *extra_requirements,
    }


def test_wheel_constraints_use_distribution_metadata(tmp_path: Path) -> None:
    _write_wheel(tmp_path, "pip", "26.2.1")
    _write_wheel(tmp_path, "Some_Package.NS", "1.2.3+cpu")

    assert dependencies.wheel_constraints(tmp_path) == {
        "pip": "26.2.1",
        "some-package-ns": "1.2.3+cpu",
    }


def test_wheel_constraints_ignore_vendored_distribution_metadata(
    tmp_path: Path,
) -> None:
    _write_wheel(tmp_path, "pip", "26.2.1")
    _write_wheel(tmp_path, "setuptools", "84.0.0")
    with ZipFile(tmp_path / "setuptools-84.0.0-py3-none-any.whl", "a") as archive:
        archive.writestr(
            "setuptools/_vendor/other.dist-info/METADATA",
            "Name: other\nVersion: 1.0\n",
        )

    assert dependencies.wheel_constraints(tmp_path) == {
        "pip": "26.2.1",
        "setuptools": "84.0.0",
    }


def test_wheel_constraints_reject_duplicate_distribution(tmp_path: Path) -> None:
    _write_wheel(tmp_path, "pip", "26.2.1")
    _write_wheel(tmp_path, "some_package", "1.0")
    _write_wheel(tmp_path, "Some-Package", "2.0")

    with pytest.raises(ValueError, match="duplicate wheel distribution: some-package"):
        dependencies.wheel_constraints(tmp_path)


@pytest.mark.parametrize(
    "contents",
    [
        {},
        {"pkg.dist-info/METADATA": "Name: pkg\n"},
        {"pkg.dist-info/METADATA": "Version: 1.0\n"},
        {"a.dist-info/METADATA": "", "b.dist-info/METADATA": ""},
    ],
)
def test_wheel_constraints_reject_malformed_metadata(
    tmp_path: Path, contents: dict[str, str]
) -> None:
    _write_wheel(tmp_path, "pip", "26.2.1")
    with ZipFile(tmp_path / "broken.whl", "w") as archive:
        for name, value in contents.items():
            archive.writestr(name, value)

    with pytest.raises(ValueError, match="METADATA|name/version"):
        dependencies.wheel_constraints(tmp_path)


def test_wheel_constraints_require_pinned_installer(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="snapshot must include pip"):
        dependencies.wheel_constraints(tmp_path)


@pytest.mark.parametrize(
    "failure",
    [
        None,
        "download",
        "build_bootstrap",
        "assembly",
        "bootstrap",
        "metadata",
        "non_cpu",
    ],
)
def test_preparation_activates_only_verified_complete_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str | None
) -> None:
    wheelhouse = tmp_path / "path with spaces" / "wheels"
    github_env = tmp_path / "environment"
    github_env.write_text("EXISTING=kept\n", encoding="utf-8")
    commands: list[list[str]] = []
    bootstrap_environments: list[dict[str, str]] = []

    def run(
        command: list[str],
        *,
        check: bool,
        timeout: int,
        env: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        assert check and timeout > 0
        commands.append(command)
        if "download" in command:
            assert "--require-hashes" in command
            assert "--only-binary=:all:" in command
            assert "--no-deps" in command
            assert command[-1].endswith("build.txt")
            if failure == "download":
                raise subprocess.CalledProcessError(1, command)

            build_wheels = Path(command[command.index("--dest") + 1])
            _write_wheel(build_wheels, "pip", "26.2.1")
        elif "wheel" in command:
            assert Path(command[0]).is_relative_to(wheelhouse / "build-environment")
            assert "--require-hashes" in command
            assert "--no-build-isolation" in command
            assert "--check-build-dependencies" in command
            assert command[-1].endswith("torch.txt")
            if failure == "assembly":
                raise subprocess.CalledProcessError(1, command)

            _write_wheel(wheelhouse, "pip", "26.2.1")
            if failure != "metadata":
                _write_wheel(
                    wheelhouse,
                    "torch",
                    "2.14.1" if failure == "non_cpu" else "2.14.1+cpu",
                )
            else:
                (wheelhouse / "broken.whl").write_bytes(b"invalid zip")
        else:
            assert env is not None
            assert github_env.read_text() == "EXISTING=kept\n"
            assert env["PIP_NO_INDEX"] == "1"
            if "--require-hashes" in command:
                assert Path(command[0]).is_relative_to(wheelhouse / "build-environment")
                if failure == "build_bootstrap":
                    raise subprocess.CalledProcessError(1, command)
            else:
                assert command[0] == dependencies.sys.executable
                bootstrap_environments.append(env)
                if failure == "bootstrap":
                    raise subprocess.CalledProcessError(1, command)

        return subprocess.CompletedProcess(command, 0)

    def create(builder: venv.EnvBuilder, directory: Path) -> None:
        assert builder.with_pip
        assert not builder.system_site_packages
        assert directory == wheelhouse / "build-environment"

    monkeypatch.setattr(dependencies.subprocess, "run", run)
    monkeypatch.setattr(dependencies.venv.EnvBuilder, "create", create)
    if failure is not None:
        expected, message = {
            "download": (subprocess.CalledProcessError, "non-zero exit status"),
            "build_bootstrap": (subprocess.CalledProcessError, "non-zero exit status"),
            "assembly": (subprocess.CalledProcessError, "non-zero exit status"),
            "bootstrap": (subprocess.CalledProcessError, "non-zero exit status"),
            "metadata": (BadZipFile, "not a zip file"),
            "non_cpu": (ValueError, "CPU-only torch"),
        }[failure]
        with pytest.raises(expected, match=message):
            dependencies.prepare_dependencies(tmp_path, "torch", wheelhouse, github_env)

        assert github_env.read_text() == "EXISTING=kept\n"
        return

    dependencies.prepare_dependencies(tmp_path, "torch", wheelhouse, github_env)

    assert len(commands) == 4
    constraints = wheelhouse / "constraints.txt"
    assert constraints.read_text() == "pip==26.2.1\ntorch==2.14.1+cpu\n"
    assert len(bootstrap_environments) == 1
    environment = bootstrap_environments[0]
    for name in ("PIP_CONSTRAINT", "PIP_BUILD_CONSTRAINT"):
        assert environment[name] == constraints.resolve().as_uri()

    assert environment["PIP_NO_INDEX"] == "1"
    assert environment["PIP_FIND_LINKS"] == wheelhouse.resolve().as_uri()
    assert github_env.read_text().startswith("EXISTING=kept\n")
    assert (
        f"PIP_CONSTRAINT={constraints.resolve().as_uri()}\n" in github_env.read_text()
    )
    assert (
        f"PIP_BUILD_CONSTRAINT={constraints.resolve().as_uri()}\n"
        in github_env.read_text()
    )


def test_preparation_refuses_preexisting_wheelhouse(tmp_path: Path) -> None:
    with pytest.raises(FileExistsError):
        dependencies.prepare_dependencies(tmp_path, "base", tmp_path, tmp_path / "env")


def test_refresh_uses_universal_snapshots_except_the_existing_cpu_torch_role(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_project(tmp_path)
    commands: list[list[str]] = []

    def run(
        command: list[str], *, check: bool, timeout: int
    ) -> subprocess.CompletedProcess[str]:
        assert check and timeout > 0
        commands.append(command)
        if Path(command[3]).stem == "build":
            assert set(Path(command[3]).read_text().splitlines()) == {
                "pip>=25.3",
                "setuptools",
                "wheel",
                "Cython",
            }
        Path(command[command.index("--output-file") + 1]).write_text("pip==26.2.1\n")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(dependencies.subprocess, "run", run)
    dependencies.refresh_snapshots(tmp_path)

    assert len(commands) == 7
    for command in commands:
        profile = Path(command[3]).stem
        content = (tmp_path / ".github" / "requirements" / f"{profile}.txt").read_text()
        assert "--generate-hashes" in command
        if profile == "torch":
            assert "--universal" not in command
            assert command[-4:] == [
                "--python-platform",
                "linux",
                "--torch-backend",
                "cpu",
            ]
            assert content.startswith(f"--extra-index-url {dependencies.CPU_INDEX}\n")
        else:
            assert "--universal" in command
