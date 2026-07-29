import argparse
import json
import math
from pathlib import Path

from einf.analysis.checkers import (
    SUPPORTED_CHECKER_NAMES,
    CheckerExecutionPolicy,
    build_checker_adapters,
)
from einf.analysis.validator.json_projection import project_validation_report
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
    parser.add_argument(
        "--checker-timeout-seconds",
        type=_positive_float,
        default=30.0,
        help="Maximum runtime for each external checker process.",
    )
    parser.add_argument(
        "--checker-max-concurrency",
        type=_positive_int,
        default=1,
        help="Maximum number of external checker processes run concurrently.",
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
        checker_execution_policy=CheckerExecutionPolicy(
            timeout_seconds=arguments.checker_timeout_seconds,
            max_concurrency=arguments.checker_max_concurrency,
        ),
    )
    print(json.dumps(project_validation_report(report), indent=2, sort_keys=True))
    return report.exit_code()


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a finite positive number")
    return parsed


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


__all__ = ["build_argument_parser", "main"]
