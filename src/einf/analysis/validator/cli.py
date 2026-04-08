import argparse
import json
from dataclasses import asdict
from pathlib import Path

from einf.analysis.checkers import SUPPORTED_CHECKER_NAMES, build_checker_adapters
from einf.analysis.validator.run import (
    SUPPORTED_PARSER_NAMES,
    build_parser_backend,
    run_validation,
)


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the validator CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="einf-validate",
        description="Validate einf DSL usage in Python source files.",
    )
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="Python files or directories to analyze.",
    )
    parser.add_argument(
        "--parser",
        choices=SUPPORTED_PARSER_NAMES,
        default="ast",
        help="Parser backend used for static analysis.",
    )
    parser.add_argument(
        "--checker",
        action="append",
        choices=SUPPORTED_CHECKER_NAMES,
        default=[],
        help="External type checker to run alongside einf semantic analysis.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the einf validator CLI."""
    arguments = build_argument_parser().parse_args(argv)
    parser_backend = build_parser_backend(arguments.parser)
    checker_adapters = build_checker_adapters(tuple(arguments.checker))
    report = run_validation(
        targets=tuple(arguments.paths),
        parser_backend=parser_backend,
        checker_adapters=checker_adapters,
    )
    print(json.dumps(asdict(report), indent=2, sort_keys=True))
    return report.exit_code()


__all__ = ["build_argument_parser", "main"]
