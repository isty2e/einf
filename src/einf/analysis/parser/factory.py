from einf.analysis.parser.ast_backend import AstParserBackend
from einf.analysis.parser.base import ParserBackend
from einf.analysis.parser.libcst_backend import LibCstParserBackend

SUPPORTED_PARSER_NAMES = ("ast", "libcst")


def build_parser_backend(parser_name: str) -> ParserBackend:
    """Build one parser backend from a supported parser name."""
    match parser_name:
        case "ast":
            return AstParserBackend()
        case "libcst":
            return LibCstParserBackend()
        case _:
            raise ValueError(f"unsupported parser backend: {parser_name}")


__all__ = ["SUPPORTED_PARSER_NAMES", "build_parser_backend"]
