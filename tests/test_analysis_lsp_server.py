import pytest

pygls = pytest.importorskip("pygls")
_ = pygls


def test_build_server_returns_language_server() -> None:
    from einf.analysis.lsp.server import EinfLanguageServer, build_server

    server = build_server()

    assert isinstance(server, EinfLanguageServer)
    assert server.einf_service.config.parser == "ast"
    assert server.einf_service.config.checkers == ()
