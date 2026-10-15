import json
import os
import subprocess
import sys
import venv
from pathlib import Path

import pytest

import tests.package.verify_wheel_entrypoints as verifier


def _validator_result(
    target: Path,
    *,
    parser: str = "ast",
    diagnostic_codes: tuple[str, ...] = (),
    failure_kinds: tuple[str, ...] = (),
) -> subprocess.CompletedProcess[str]:
    report = {
        "schema_version": "0.4",
        "parser_backend": parser,
        "checker_failures": [],
        "discovery_failures": [],
        "limit_failures": [],
        "files": [
            {
                "path": str(target.resolve()),
                "checker_diagnostics": [],
                "diagnostics": [{"code": code} for code in diagnostic_codes],
                "failures": [{"kind": kind} for kind in failure_kinds],
            }
        ],
    }
    return subprocess.CompletedProcess(
        args=["einf-validate"],
        returncode=int(bool(diagnostic_codes or failure_kinds)),
        stdout=json.dumps(report),
        stderr="",
    )


@pytest.mark.parametrize("name", ["einf.whl", "einf.tar.gz"])
def test_artifact_verifier_accepts_individual_artifacts(
    tmp_path: Path, name: str
) -> None:
    artifact = tmp_path / name
    artifact.touch()

    assert verifier._resolve_artifacts(artifact) == (artifact.resolve(),)


def test_artifact_verifier_requires_both_unique_artifacts(tmp_path: Path) -> None:
    wheel = tmp_path / "einf.whl"
    sdist = tmp_path / "einf.tar.gz"
    with pytest.raises(ValueError, match="one wheel and sdist"):
        verifier._resolve_artifacts(tmp_path)

    wheel.touch()
    with pytest.raises(ValueError, match="one wheel and sdist"):
        verifier._resolve_artifacts(tmp_path)

    sdist.touch()
    assert verifier._resolve_artifacts(tmp_path) == (wheel.resolve(), sdist.resolve())

    second_wheel = tmp_path / "stale.whl"
    second_wheel.touch()
    with pytest.raises(ValueError, match="one wheel and sdist"):
        verifier._resolve_artifacts(tmp_path)

    second_wheel.unlink()
    (tmp_path / "stale.tar.gz").touch()
    with pytest.raises(ValueError, match="one wheel and sdist"):
        verifier._resolve_artifacts(tmp_path)


def test_artifact_verifier_main_exercises_wheel_and_sdist(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifacts = (tmp_path / "einf.whl", tmp_path / "einf.tar.gz")
    for artifact in artifacts:
        artifact.touch()

    verified: list[Path] = []
    monkeypatch.setattr(verifier, "_verify_entrypoints", verified.append)
    monkeypatch.setattr(sys, "argv", ["verify-entrypoints", str(tmp_path)])

    verifier.main()

    assert verified == list(artifacts)


def test_artifact_subprocess_environment_excludes_host_python_and_tools(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PYTHONPATH", "/checkout/src")
    monkeypatch.setenv("PYTHONHOME", "/host/python")
    monkeypatch.setenv("PATH", "/host/checkers")

    environment = verifier._subprocess_environment(tmp_path)

    assert "PYTHONPATH" not in environment
    assert "PYTHONHOME" not in environment
    assert environment["PYTHONNOUSERSITE"] == "1"
    assert environment["PATH"] == f"{tmp_path}{os.pathsep}{os.defpath}"
    assert os.environ["PYTHONPATH"] == "/checkout/src"


def test_artifact_entrypoint_subprocess_uses_isolated_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PYTHONPATH", "/checkout/src")
    monkeypatch.setenv("PYTHONHOME", "/host/python")
    monkeypatch.setenv("PATH", "/host/checkers")
    command = [
        sys.executable,
        "-c",
        "import json, os; print(json.dumps(dict(os.environ)))",
    ]

    result = verifier._run_entrypoint(command, cwd=tmp_path)

    assert result.returncode == 0, result.stderr
    environment = json.loads(result.stdout)
    assert "PYTHONPATH" not in environment
    assert "PYTHONHOME" not in environment
    assert environment["PATH"] == (
        f"{Path(sys.executable).parent}{os.pathsep}{os.defpath}"
    )


@pytest.mark.parametrize("name", ["einf.whl", "einf.tar.gz"])
def test_artifact_entrypoint_verifier_exercises_base_then_extras(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    artifact = tmp_path / name
    roots: list[Path] = []
    commands: list[list[str]] = []
    validator_calls: list[tuple[str, bool]] = []
    lsp_calls: list[list[str]] = []

    def create_environment(builder: venv.EnvBuilder, environment: Path) -> None:
        assert builder.with_pip
        assert not builder.system_site_packages
        environment.mkdir()
        roots.append(environment.parent)

    def run(command: list[str], *, cwd: Path) -> None:
        assert cwd == roots[0]
        commands.append(command)

    def verify_validator(
        einf_validate: Path, *, cwd: Path, parser: str, available: bool = True
    ) -> None:
        assert cwd == roots[0]
        assert einf_validate == cwd / "venv" / "bin" / "einf-validate"
        validator_calls.append((parser, available))

    def run_entrypoint(
        command: list[str], *, cwd: Path
    ) -> subprocess.CompletedProcess[str]:
        assert cwd == roots[0]
        lsp_calls.append(command)
        if len(lsp_calls) == 1:
            return subprocess.CompletedProcess(
                command, 1, "", f"{verifier._LSP_INSTALL_ERROR}\n"
            )
        return subprocess.CompletedProcess(command, 0, "usage: einf-lsp --tcp", "")

    monkeypatch.setattr(venv.EnvBuilder, "create", create_environment)
    monkeypatch.setattr(verifier, "_run", run)
    monkeypatch.setattr(verifier, "_verify_validator", verify_validator)
    monkeypatch.setattr(verifier, "_run_entrypoint", run_entrypoint)

    verifier._verify_entrypoints(artifact)

    assert validator_calls == [
        ("ast", True),
        ("libcst", False),
        ("ast", True),
        ("libcst", True),
    ]
    assert len(commands) == 4
    assert commands[0][1:] == [
        "-I",
        "-m",
        "pip",
        "install",
        "--no-cache-dir",
        str(artifact),
    ]
    assert commands[1][1:3] == ["-I", "-c"]
    assert commands[1][-1] == "base"
    assert commands[2][1:] == [
        "-I",
        "-m",
        "pip",
        "install",
        "--no-cache-dir",
        f"einf[analysis,lsp] @ {artifact.as_uri()}",
    ]
    assert commands[3][1:3] == ["-I", "-c"]
    assert commands[3][-1] == "extras"
    assert len(lsp_calls) == 2
    assert not roots[0].exists()


def test_artifact_entrypoint_verifier_cleans_up_failed_install(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    roots: list[Path] = []

    def create_environment(builder: venv.EnvBuilder, environment: Path) -> None:
        environment.mkdir()
        roots.append(environment.parent)

    def fail_install(command: list[str], *, cwd: Path) -> None:
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(venv.EnvBuilder, "create", create_environment)
    monkeypatch.setattr(verifier, "_run", fail_install)

    with pytest.raises(subprocess.CalledProcessError):
        verifier._verify_entrypoints(tmp_path / "einf.tar.gz")

    assert not roots[0].exists()


@pytest.mark.parametrize("parser", ["ast", "libcst"])
@pytest.mark.parametrize(
    ("diagnostic_codes", "failure_kinds"),
    [((), ()), (("ANALYSIS_AXIS_NOT_IN_INPUT",), ()), ((), ("parse_error",))],
)
def test_artifact_validator_accepts_expected_behavior(
    tmp_path: Path,
    parser: str,
    diagnostic_codes: tuple[str, ...],
    failure_kinds: tuple[str, ...],
) -> None:
    target = tmp_path / "sample.py"
    result = _validator_result(
        target,
        parser=parser,
        diagnostic_codes=diagnostic_codes,
        failure_kinds=failure_kinds,
    )

    verifier._check_validator_result(
        result,
        target=target,
        parser=parser,
        diagnostic_codes=diagnostic_codes,
        failure_kinds=failure_kinds,
    )


def test_artifact_validator_accepts_explicit_parser_unavailability(
    tmp_path: Path,
) -> None:
    target = tmp_path / "sample.py"
    result = _validator_result(
        target, parser="libcst", failure_kinds=("parser_unavailable",)
    )

    verifier._check_validator_result(
        result,
        target=target,
        parser="libcst",
        diagnostic_codes=(),
        failure_kinds=("parser_unavailable",),
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "exit",
        "stderr",
        "schema",
        "limits",
        "parser",
        "path",
        "coverage",
        "diagnostics",
        "failure",
        "json",
    ],
)
def test_artifact_validator_rejects_false_green_results(
    tmp_path: Path, mutation: str
) -> None:
    target = tmp_path / "sample.py"
    diagnostic_codes = ("ANALYSIS_AXIS_NOT_IN_INPUT",)
    result = _validator_result(target, diagnostic_codes=diagnostic_codes)
    report = json.loads(result.stdout)
    match mutation:
        case "exit":
            result.returncode = 0
        case "stderr":
            result.stderr = "Traceback: broken installed package"
        case "schema":
            report["schema_version"] = "0.3"
        case "limits":
            report["limit_failures"] = [
                {
                    "resource": "file_bytes",
                    "limit": 1,
                    "observed": 2,
                    "path": str(target),
                }
            ]
        case "parser":
            report["parser_backend"] = "libcst"
        case "path":
            report["files"][0]["path"] = "other.py"
        case "coverage":
            report["files"] = []
        case "diagnostics":
            report["files"][0]["diagnostics"] = []
        case "failure":
            report["files"][0]["failures"] = [{"kind": "parser_unavailable"}]

    result.stdout = "not JSON" if mutation == "json" else json.dumps(report)

    with pytest.raises(RuntimeError, match="ast validator on sample.py failed"):
        verifier._check_validator_result(
            result,
            target=target,
            parser="ast",
            diagnostic_codes=diagnostic_codes,
            failure_kinds=(),
        )


def test_artifact_validator_rejects_missing_parser_false_success(
    tmp_path: Path,
) -> None:
    target = tmp_path / "sample.py"
    result = _validator_result(target, parser="libcst")

    with pytest.raises(RuntimeError, match="libcst validator on sample.py failed"):
        verifier._check_validator_result(
            result,
            target=target,
            parser="libcst",
            diagnostic_codes=(),
            failure_kinds=("parser_unavailable",),
        )
