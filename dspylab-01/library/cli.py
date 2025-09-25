"""Command-line interface for DSPyLab."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .config import load_experiment_config, ExperimentConfigError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dspylab",
        description="Run DSPy experiments using declarative configs.",
    )
    parser.add_argument(
        "run",
        help="Execute an experiment using a YAML config.",
        nargs="?",
    )
    parser.add_argument(
        "config",
        help="Path to the experiment YAML file.",
        nargs="?",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging output.",
    )
    return parser


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s:%(name)s:%(message)s")


def run_cli(args: list[str] | None = None) -> int:
    parser = build_parser()
    parsed = parser.parse_args(args=args)

    if parsed.run != "run" or not parsed.config:
        parser.print_help()
        return 1

    setup_logging(parsed.verbose)

    config_path = Path(parsed.config)

    try:
        load_experiment_config(config_path)
    except ExperimentConfigError as exc:
        logging.error("Config validation failed:\n%s", exc)
        return 2

    logging.info("Configuration %s validated successfully", config_path)
    logging.info("Execution pipeline not yet implemented.")
    return 0


def main() -> None:
    raise SystemExit(run_cli())


if __name__ == "__main__":
    main()

