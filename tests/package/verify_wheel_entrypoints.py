import argparse
import json
import os
import subprocess
import tempfile
import venv
from pathlib import Path
from typing import NoReturn

_LSP_INSTALL_ERROR = (
    "einf-lsp requires optional LSP dependencies; install them with "
    "'pip install \"einf[lsp]\"'"
)

_INSTALL_ORIGIN_CASE = """
import importlib
import sys
from importlib.util import find_spec
from pathlib import Path

prefix = Path(sys.prefix).resolve()
for name in (
    "einf", "einf.analysis.validator.cli", "einf.analysis.parser.ast_backend",
    "einf.analysis.parser.libcst_backend", "einf.analysis.lsp.cli",
):
    module = importlib.import_module(name)
    assert Path(module.__file__).resolve().is_relative_to(prefix), module.__file__

extras = sys.argv[1] == "extras"
for name in ("libcst", "pygls", "lsprotocol"):
    assert (find_spec(name) is not None) == extras, name
for name in ("numpy", "torch", "array_api_strict", "pytest"):
    assert find_spec(name) is None, name
"""

_VALID_SOURCE = """from einf import ax, axes, rearrange
b = axes("b")[0]
rearrange(ax[b], ax[b])
"""

_DIAGNOSTIC_SOURCE = """from einf import ax, axes, reduce
b, n, z = axes("b", "n", "z")
reduce(ax[b, n], ax[b, z])
"""

_SYNTAX_ERROR_SOURCE = "from einf import rearrange\nrearrange(\n"


def _resolve_artifacts(path: Path) -> tuple[Path, ...]:
    if path.is_file() and (path.suffix == ".whl" or path.name.endswith(".tar.gz")):
        return (path.resolve(),)
    if path.is_dir():
        wheels = tuple(path.glob("*.whl"))
        sdists = tuple(path.glob("*.tar.gz"))
        if len(wheels) == 1 and len(sdists) == 1:
            return (wheels[0].resolve(), sdists[0].resolve())
    raise ValueError(
        f"expected an artifact or a directory with one wheel and sdist: {path}"
    )


def _subprocess_environment(bin_directory: Path) -> dict[str, str]:
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTHONHOME", None)
    environment["PYTHONNOUSERSITE"] = "1"
    environment["PATH"] = f"{bin_directory}{os.pathsep}{os.defpath}"
    return environment


def _run(command: list[str], *, cwd: Path) -> None:
    subprocess.run(
        command,
        cwd=cwd,
        env=_subprocess_environment(Path(command[0]).parent),
        check=True,
        timeout=300,
    )


def _run_entrypoint(
    command: list[str], *, cwd: Path
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        env=_subprocess_environment(Path(command[0]).parent),
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )


def _raise_unexpected_result(
    *,
    stage: str,
    result: subprocess.CompletedProcess[str],
) -> NoReturn:
    raise RuntimeError(
        f"{stage} failed with exit code {result.returncode}\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )


def _check_validator_result(
    result: subprocess.CompletedProcess[str],
    *,
    target: Path,
    parser: str,
    diagnostic_codes: tuple[str, ...],
    failure_kinds: tuple[str, ...],
) -> None:
    stage = f"{parser} validator on {target.name}"
    expected_exit = int(bool(diagnostic_codes or failure_kinds))
    if result.returncode != expected_exit or result.stderr:
        _raise_unexpected_result(stage=stage, result=result)

    try:
        report = json.loads(result.stdout)
    except json.JSONDecodeError:
        _raise_unexpected_result(stage=stage, result=result)

    if (
        report["schema_version"] != "0.4"
        or report["parser_backend"] != parser
        or report["checker_failures"] != []
        or report["discovery_failures"] != []
        or report["limit_failures"] != []
        or len(report["files"]) != 1
    ):
        _raise_unexpected_result(stage=stage, result=result)

    file_report = report["files"][0]
    if (
        file_report["path"] != str(target.resolve())
        or file_report["checker_diagnostics"] != []
        or tuple(entry["code"] for entry in file_report["diagnostics"])
        != diagnostic_codes
        or tuple(entry["kind"] for entry in file_report["failures"]) != failure_kinds
    ):
        _raise_unexpected_result(stage=stage, result=result)


def _verify_validator(
    einf_validate: Path, *, cwd: Path, parser: str, available: bool = True
) -> None:
    for name, source, diagnostic_codes, failure_kinds in (
        ("valid.py", _VALID_SOURCE, (), ()),
        (
            "diagnostic.py",
            _DIAGNOSTIC_SOURCE,
            ("ANALYSIS_AXIS_NOT_IN_INPUT",),
            (),
        ),
        ("syntax_error.py", _SYNTAX_ERROR_SOURCE, (), ("parse_error",)),
    ):
        target = cwd / name
        target.write_text(source, encoding="utf-8")
        result = _run_entrypoint(
            [str(einf_validate), "--parser", parser, str(target)], cwd=cwd
        )
        _check_validator_result(
            result,
            target=target,
            parser=parser,
            diagnostic_codes=diagnostic_codes if available else (),
            failure_kinds=failure_kinds if available else ("parser_unavailable",),
        )


def _verify_entrypoints(artifact: Path) -> None:
    with tempfile.TemporaryDirectory(
        prefix="einf-artifact-entrypoints-"
    ) as temporary_dir:
        root = Path(temporary_dir)
        environment = root / "venv"
        venv.EnvBuilder(with_pip=True, system_site_packages=False).create(environment)
        python = environment / "bin" / "python"
        einf_lsp = environment / "bin" / "einf-lsp"
        einf_validate = environment / "bin" / "einf-validate"

        _run(
            [
                str(python),
                "-I",
                "-m",
                "pip",
                "install",
                "--no-cache-dir",
                str(artifact),
            ],
            cwd=root,
        )
        _run([str(python), "-I", "-c", _INSTALL_ORIGIN_CASE, "base"], cwd=root)
        _verify_validator(einf_validate, cwd=root, parser="ast")
        _verify_validator(einf_validate, cwd=root, parser="libcst", available=False)

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
                "-I",
                "-m",
                "pip",
                "install",
                "--no-cache-dir",
                f"einf[analysis,lsp] @ {artifact.as_uri()}",
            ],
            cwd=root,
        )
        _run([str(python), "-I", "-c", _INSTALL_ORIGIN_CASE, "extras"], cwd=root)
        _verify_validator(einf_validate, cwd=root, parser="ast")
        _verify_validator(einf_validate, cwd=root, parser="libcst")

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
        description="Check installed wheel and sdist origins, validators and LSP entry points."
    )
    parser.add_argument("artifact", type=Path)
    arguments = parser.parse_args()
    for artifact in _resolve_artifacts(arguments.artifact):
        _verify_entrypoints(artifact)
        print(f"Installed entry points verified: {artifact.name}")


if __name__ == "__main__":
    main()
