import argparse
import json
import math
from pathlib import Path

from einf.analysis.checkers import (
    SUPPORTED_CHECKER_NAMES,
    CheckerExecutionPolicy,
    build_checker_adapters,
)
from einf.analysis.parser import SUPPORTED_PARSER_NAMES, build_parser_backend
from einf.analysis.validator.json_projection import project_validation_report
from einf.analysis.validator.limits import ValidationLimits
from einf.analysis.validator.run import run_validation


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the validator CLI argument parser."""
    checker_policy_defaults = CheckerExecutionPolicy()
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
        "--max-discovery-entries",
        type=_positive_int,
        default=None,
        help="Maximum explicit target attempts plus directory entries; unrestricted by default.",
    )
    parser.add_argument(
        "--max-file-bytes",
        type=_positive_int,
        default=None,
        help="Maximum raw source bytes per file; unrestricted by default.",
    )
    parser.add_argument(
        "--max-total-bytes",
        type=_positive_int,
        default=None,
        help="Maximum cumulative validator source bytes; unrestricted by default.",
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
        default=checker_policy_defaults.timeout_seconds,
        help="Maximum runtime for each external checker process.",
    )
    parser.add_argument(
        "--checker-cleanup-timeout-seconds",
        type=_positive_float,
        default=checker_policy_defaults.cleanup_timeout_seconds,
        help="Maximum cleanup time after checker timeout or cancellation.",
    )
    parser.add_argument(
        "--checker-max-concurrency",
        type=_positive_int,
        default=checker_policy_defaults.max_concurrency,
        help="Maximum number of external checker processes run concurrently.",
    )
    parser.add_argument(
        "--checker-max-output-bytes",
        type=_positive_int,
        default=checker_policy_defaults.max_output_bytes,
        help="Maximum combined buffered stdout+stderr bytes per checker process.",
    )
    parser.add_argument(
        "--checker-max-diagnostics",
        type=_positive_int,
        default=checker_policy_defaults.max_diagnostics,
        help=(
            "Maximum diagnostics per checker result and in the merged validator output."
        ),
    )
    parser.add_argument(
        "--checker-max-field-length",
        type=_positive_int,
        default=checker_policy_defaults.max_field_length,
        help="Maximum characters in one externally reported diagnostic field.",
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
        limits=ValidationLimits(
            max_discovery_entries=arguments.max_discovery_entries,
            max_file_bytes=arguments.max_file_bytes,
            max_total_bytes=arguments.max_total_bytes,
        ),
        checker_adapters=checker_adapters,
        checker_execution_policy=CheckerExecutionPolicy(
            timeout_seconds=arguments.checker_timeout_seconds,
            cleanup_timeout_seconds=arguments.checker_cleanup_timeout_seconds,
            max_concurrency=arguments.checker_max_concurrency,
            max_output_bytes=arguments.checker_max_output_bytes,
            max_diagnostics=arguments.checker_max_diagnostics,
            max_field_length=arguments.checker_max_field_length,
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
