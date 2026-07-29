import subprocess
import sys

_LSP_INSTALL_ERROR = (
    "einf-lsp requires optional LSP dependencies; install them with "
    "'pip install \"einf[lsp]\"'"
)


def test_lsp_cli_reports_missing_optional_dependencies_without_traceback() -> None:
    script = """
import builtins

real_import = builtins.__import__

def import_without_lsp(name, globals=None, locals=None, fromlist=(), level=0):
    root = name.partition(".")[0]
    if root in {"lsprotocol", "pygls"}:
        raise ModuleNotFoundError(f"No module named {root!r}", name=root)
    return real_import(name, globals, locals, fromlist, level)

builtins.__import__ = import_without_lsp

from einf.analysis.lsp.cli import main

main()
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert result.stdout == ""
    assert result.stderr == f"{_LSP_INSTALL_ERROR}\n"
    assert "Traceback" not in result.stderr
