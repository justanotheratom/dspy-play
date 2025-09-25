"""Command-line interface for DSPyLab."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from .config import load_experiment_config, ExperimentConfigError
from .config.models import ExperimentConfig, expand_matrix
from .program.loader import load_program_components
from .runtime import ExperimentExecutor


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
        doc = load_experiment_config(config_path)
    except ExperimentConfigError as exc:
        logging.error("Config validation failed:\n%s", exc)
        return 2

    dataset = doc.data.get("dataset")
    if dataset and "path" in dataset:
        dataset_path = Path(dataset["path"])
        if not dataset_path.is_absolute():
            dataset["path"] = str((config_path.parent / dataset_path).resolve())

    config = ExperimentConfig(**doc.data)

    logging.info("Configuration %s validated successfully", config_path)

    try:
        bundle = load_program_components(
            module_path=config.program.module,
            factory_name=config.program.factory,
            dataset_loader_name=config.program.dataset_loader,
            metrics=config.program.metrics,
        )
    except Exception as exc:
        logging.error("Failed to load program components: %s", exc)
        return 3

    runs = expand_matrix(config)

    executor = ExperimentExecutor(config=config, bundle=bundle, runs=runs, config_path=config_path)
    try:
        outcomes = executor.execute()
    except Exception as exc:
        logging.exception("Experiment execution failed")
        return 4

    _write_outputs(config, outcomes)
    logging.info("Experiment completed successfully")
    return 0


def _write_outputs(config: ExperimentConfig, outcomes):
    outputs = config.outputs
    root = Path(outputs.root_dir)
    root.mkdir(parents=True, exist_ok=True)

    raw_path = root / outputs.raw_filename
    summary_path = root / outputs.summary_filename

    raw_records = []
    summary_entries = []

    for outcome in outcomes:
        for record in outcome.records:
            raw_records.append({
                "run": outcome.name,
                "example": record.get("example"),
                "output": record.get("output"),
            })

        summary_entries.append({
            "run": outcome.name,
            "metrics": outcome.metrics,
            "optimizer": outcome.optimizer_id,
            "strategy": outcome.strategy_id,
        })

    raw_path.write_text("\n".join(json.dumps(r) for r in raw_records), encoding="utf-8")
    summary_path.write_text(json.dumps(summary_entries, indent=2), encoding="utf-8")


def main() -> None:
    raise SystemExit(run_cli())


if __name__ == "__main__":
    main()

