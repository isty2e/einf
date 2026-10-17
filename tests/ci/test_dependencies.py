import hashlib
import json
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


def _write_bundle(root: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    _write_project(root)
    helper = root / "tests" / "ci" / "dependencies.py"
    helper.parent.mkdir(parents=True)
    helper.write_text("# resolver input\n", encoding="utf-8")
    directory = root / "release bundle"
    directory.mkdir()
    for profile in dependencies.SNAPSHOT_PROFILES:
        (directory / f"{profile}.txt").write_text("pip==26.2.1\n", encoding="utf-8")

    monkeypatch.setattr(dependencies, "checkout_revision", lambda root: "a" * 40)
    (directory / "resolution.json").write_text(
        json.dumps(
            {
                "schema_version": "v1",
                "source_sha": "a" * 40,
                "hashes": dependencies.bundle_hashes(root, directory),
            }
        ),
        encoding="utf-8",
    )
    return directory


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
@pytest.mark.parametrize("use_bundle", [False, True])
def test_preparation_activates_only_verified_complete_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str | None,
    use_bundle: bool,
) -> None:
    bundle = _write_bundle(tmp_path, monkeypatch) if use_bundle else None
    snapshots = bundle if bundle is not None else tmp_path / ".github" / "requirements"
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
            assert Path(command[-1]) == snapshots / "build.txt"
            if failure == "download":
                raise subprocess.CalledProcessError(1, command)

            build_wheels = Path(command[command.index("--dest") + 1])
            _write_wheel(build_wheels, "pip", "26.2.1")
        elif "wheel" in command:
            assert Path(command[0]).is_relative_to(wheelhouse / "build-environment")
            assert "--require-hashes" in command
            assert "--no-build-isolation" in command
            assert "--check-build-dependencies" in command
            assert Path(command[-1]) == snapshots / "torch.txt"
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
            dependencies.prepare_dependencies(
                tmp_path, "torch", wheelhouse, github_env, bundle=bundle
            )

        assert github_env.read_text() == "EXISTING=kept\n"
        return

    dependencies.prepare_dependencies(
        tmp_path, "torch", wheelhouse, github_env, bundle=bundle
    )

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


def test_resolve_bundle_records_complete_evidence_without_refreshing_checkout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_project(tmp_path)
    helper = tmp_path / "tests" / "ci" / "dependencies.py"
    helper.parent.mkdir(parents=True)
    helper.write_text("# resolver input\n", encoding="utf-8")
    destination = tmp_path / "new bundle"
    commands: list[list[str]] = []

    def run(
        command: list[str], *, check: bool, timeout: int
    ) -> subprocess.CompletedProcess[str]:
        assert check and timeout > 0
        commands.append(command)
        Path(command[command.index("--output-file") + 1]).write_text("pip==26.2.1\n")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(dependencies.subprocess, "run", run)
    monkeypatch.setattr(dependencies, "checkout_revision", lambda root: "a" * 40)
    monkeypatch.setattr(
        dependencies.subprocess, "check_output", lambda *args, **kwargs: "uv 0.12.23\n"
    )
    dependencies.resolve_bundle(tmp_path, destination)

    assert len(commands) == 7
    assert not (tmp_path / ".github").exists()
    receipt = json.loads((destination / "resolution.json").read_text())
    assert receipt["source_sha"] == "a" * 40
    assert receipt["resolver"] == "uv 0.12.23"
    assert receipt["resolved_at"].endswith("+00:00")
    assert receipt["python"] and receipt["platform"]
    assert receipt["hashes"]["snapshots"] == {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in destination.glob("*.txt")
    }
    assert len(receipt["hashes"]["snapshots"]) == 7
    assert receipt["hashes"]["inputs"] == {
        name: hashlib.sha256(
            (tmp_path / name).read_text(encoding="utf-8").encode("utf-8")
        ).hexdigest()
        for name in ("pyproject.toml", "tests/ci/dependencies.py")
    }
    dependencies.validate_bundle(tmp_path, destination)


def test_bundle_resolution_failure_cannot_leave_a_completed_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_project(tmp_path)
    destination = tmp_path / "failed bundle"
    monkeypatch.setattr(dependencies, "checkout_revision", lambda root: "a" * 40)
    monkeypatch.setattr(
        dependencies.subprocess, "check_output", lambda *args, **kwargs: "uv 0.12.23\n"
    )

    def fail(root: Path, destination: Path) -> None:
        (destination / "build.txt").write_text("partial result\n")
        raise subprocess.CalledProcessError(1, ["uv", "pip", "compile"])

    monkeypatch.setattr(dependencies, "refresh_snapshots", fail)
    with pytest.raises(subprocess.CalledProcessError, match="non-zero exit status"):
        dependencies.resolve_bundle(tmp_path, destination)

    assert not (destination / "resolution.json").exists()


def test_bundle_resolution_refuses_preexisting_destination(tmp_path: Path) -> None:
    with pytest.raises(FileExistsError):
        dependencies.resolve_bundle(tmp_path, tmp_path)


@pytest.mark.parametrize(
    "damage",
    [
        "schema",
        "scalar",
        "source",
        "hashes",
        "receipt",
        "pyproject.toml",
        "tests/ci/dependencies.py",
        *(f"{profile}.txt" for profile in dependencies.SNAPSHOT_PROFILES),
    ],
)
def test_external_bundle_is_rejected_before_preparation_without_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, damage: str
) -> None:
    bundle = _write_bundle(tmp_path, monkeypatch)
    monkeypatch.setattr(
        dependencies.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("invalid bundle reached the installer"),
    )
    receipt_path = bundle / "resolution.json"
    receipt = json.loads(receipt_path.read_text())
    if damage == "schema":
        receipt["schema_version"] = "v2"
    elif damage == "scalar":
        receipt = None
    elif damage == "source":
        receipt["source_sha"] = "b" * 40
    elif damage == "hashes":
        del receipt["hashes"]
    elif damage == "receipt":
        receipt_path.unlink()
    else:
        path = (
            tmp_path / damage
            if damage in ("pyproject.toml", "tests/ci/dependencies.py")
            else bundle / damage
        )
        path.write_text("changed\n", encoding="utf-8")

    if damage != "receipt":
        receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    wheelhouse = tmp_path / "uncreated wheels"
    environment = tmp_path / "environment"
    environment.write_text("EXISTING=kept\n")
    expected = {
        "schema": (ValueError, "unsupported dependency resolution receipt"),
        "scalar": (ValueError, "unsupported dependency resolution receipt"),
        "source": (ValueError, "source does not match"),
        "receipt": (FileNotFoundError, "resolution.json"),
    }.get(damage, (ValueError, "hashes do not match"))
    with pytest.raises(expected[0], match=expected[1]):
        dependencies.prepare_dependencies(
            tmp_path, "base", wheelhouse, environment, bundle=bundle
        )

    assert not wheelhouse.exists()
    assert environment.read_text() == "EXISTING=kept\n"


def test_bundle_accepts_native_checkout_line_endings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = _write_bundle(tmp_path, monkeypatch)
    for name in dependencies.RESOLUTION_INPUTS:
        path = tmp_path / name
        path.write_bytes(path.read_text().replace("\n", "\r\n").encode("utf-8"))

    dependencies.validate_bundle(tmp_path, bundle)


@pytest.mark.parametrize("profile", dependencies.SNAPSHOT_PROFILES)
def test_bundle_requires_every_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, profile: str
) -> None:
    bundle = _write_bundle(tmp_path, monkeypatch)
    (bundle / f"{profile}.txt").unlink()

    with pytest.raises(FileNotFoundError, match=f"{profile}.txt"):
        dependencies.validate_bundle(tmp_path, bundle)


@pytest.mark.parametrize("use_bundle", [False, True])
def test_prepare_cli_preserves_dependency_source_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, use_bundle: bool
) -> None:
    bundle = tmp_path / "release bundle" if use_bundle else None
    argv = ["dependencies.py", "prepare", "base"]
    if bundle is not None:
        argv.extend(["--bundle", str(bundle)])

    calls: list[Path | None] = []

    def prepare(
        root: Path,
        profile: str,
        wheelhouse: Path,
        github_env: Path,
        *,
        bundle: Path | None = None,
    ) -> None:
        assert root == Path(dependencies.__file__).resolve().parents[2]
        assert profile == "base"
        assert wheelhouse == tmp_path / "einf-ci-wheels"
        assert github_env == tmp_path / "environment"
        calls.append(bundle)

    monkeypatch.setattr(dependencies.sys, "argv", argv)
    monkeypatch.setenv("RUNNER_TEMP", str(tmp_path))
    monkeypatch.setenv("GITHUB_ENV", str(tmp_path / "environment"))
    monkeypatch.setattr(dependencies, "prepare_dependencies", prepare)
    dependencies.main()

    assert calls == [bundle]


def test_resolve_cli_routes_to_bundle_production(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[Path, Path]] = []
    monkeypatch.setattr(
        dependencies.sys, "argv", ["dependencies.py", "resolve", str(tmp_path)]
    )
    monkeypatch.setattr(
        dependencies,
        "resolve_bundle",
        lambda root, destination: calls.append((root, destination)),
    )
    dependencies.main()

    assert calls == [(Path(dependencies.__file__).resolve().parents[2], tmp_path)]
