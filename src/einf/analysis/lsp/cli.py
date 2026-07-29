_LSP_INSTALL_ERROR = (
    "einf-lsp requires optional LSP dependencies; install them with "
    "'pip install \"einf[lsp]\"'"
)
_LSP_DEPENDENCY_MODULES = frozenset({"lsprotocol", "pygls"})


def main() -> None:
    """Run the einf language server over the selected transport."""
    try:
        from pygls.cli import start_server

        from .server import build_server
    except ModuleNotFoundError as error:
        if error.name not in _LSP_DEPENDENCY_MODULES:
            raise
        raise SystemExit(_LSP_INSTALL_ERROR) from None

    start_server(build_server())


__all__ = ["main"]
