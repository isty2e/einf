from pygls.cli import start_server

from .server import build_server


def main() -> None:
    """Run the einf language server over the selected transport."""
    start_server(build_server())


__all__ = ["main"]
