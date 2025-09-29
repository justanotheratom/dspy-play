"""Command-line interface for DSPyLab."""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Dict, Any
import traceback
import uuid

from .config import load_experiment_config, ExperimentConfigError
from .config.models import ExperimentConfig, RunSpec, expand_matrix
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
    parser.add_argument(
        "--hypothesis",
        help="Hypothesis or goal for this experiment run.",
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

    hypothesis = parsed.hypothesis
    if hypothesis is None:
        try:
            hypothesis = input("Hypothesis for this run: ").strip()
        except EOFError:
            hypothesis = ""

    if not hypothesis:
        logging.error("A hypothesis is required to start a run.")
        return 5

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

    run_id = _generate_run_id()
    logging.info("Assigned run id %s", run_id)
    started_at = datetime.now(timezone.utc)

    outcomes = []
    status = "succeeded"
    error_text = None
    exit_code = 0

    try:
        outcomes = executor.execute()
        logging.info("Experiment completed successfully")
    except Exception:
        logging.exception("Experiment execution failed")
        status = "failed"
        error_text = traceback.format_exc()
        exit_code = 4

    finished_at = datetime.now(timezone.utc)

    try:
        _finalize_run(
            config=config,
            runs=runs,
            outcomes=outcomes,
            run_id=run_id,
            hypothesis=hypothesis,
            started_at=started_at,
            finished_at=finished_at,
            status=status,
            error_text=error_text,
            config_path=config_path,
        )
    except Exception:
        logging.exception("Failed to finalize run outputs")
        return max(exit_code, 6)

    return exit_code


def _generate_run_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suffix = uuid.uuid4().hex[:6]
    return f"{timestamp}-{suffix}"


def _slugify(value: str) -> str:
    lowered = value.lower()
    cleaned = [ch if ch.isalnum() else "-" for ch in lowered]
    slug = "".join(cleaned).strip("-")
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug or "run"


def _finalize_run(
    *,
    config: ExperimentConfig,
    runs,
    outcomes,
    run_id: str,
    hypothesis: str,
    started_at: datetime,
    finished_at: datetime,
    status: str,
    error_text: str | None,
    config_path: Path,
) -> None:
    outputs = config.outputs
    root = Path(outputs.root_dir)
    root.mkdir(parents=True, exist_ok=True)

    run_path = root / run_id
    subruns_root = run_path / "subruns"
    subruns_root.mkdir(parents=True, exist_ok=True)

    outcome_map = {outcome.name: outcome for outcome in outcomes}

    aggregated_raw: List[dict] = []
    aggregated_summary: List[dict] = []
    sub_run_entries: List[dict] = []
    aggregated_cost_events: List[dict] = []

    overall_costs = {
        "actual_usd": 0.0,
        "economic_usd": 0.0,
        "calls": 0,
        "cache_hits": 0,
        "events": 0,
    }
    overall_phases: Dict[str, Dict[str, float]] = {}
    overall_program_calls = 0
    aggregated_warning_messages: List[str] = []
    aggregated_warning_details: List[Dict[str, Any]] = []

    for index, run_spec in enumerate(runs):
        slug = _slugify(run_spec.name)
        sub_run_id = f"{index + 1:02d}-{slug}"
        sub_run_path = subruns_root / sub_run_id
        entry = {
            "sub_run_id": sub_run_id,
            "name": run_spec.name,
            "model": run_spec.model.id,
            "optimizer": run_spec.optimizer.id if run_spec.optimizer else None,
            "status": "pending",
            "paths": {},
        }

        outcome = outcome_map.get(run_spec.name)
        if outcome is not None:
            sub_run_path.mkdir(parents=True, exist_ok=True)

            raw_records = []
            for record in outcome.records:
                raw_entry = {
                    "run": outcome.name,
                    "example": record.get("example"),
                    "output": record.get("output"),
                }
                raw_records.append(raw_entry)
                aggregated_raw.append(raw_entry)

            raw_file = sub_run_path / "raw.jsonl"
            raw_file.write_text("\n".join(json.dumps(r) for r in raw_records), encoding="utf-8")

            metrics_file = sub_run_path / "metrics.json"
            metrics_file.write_text(json.dumps(outcome.metrics, indent=2), encoding="utf-8")

            cost_events = outcome.cost_events or []
            costs_file = sub_run_path / "costs.jsonl"
            if cost_events:
                costs_file.write_text("\n".join(json.dumps(event) for event in cost_events), encoding="utf-8")
            else:
                costs_file.write_text("", encoding="utf-8")

            entry.update(
                {
                    "status": "completed",
                    "metrics": outcome.metrics,
                    "record_count": len(raw_records),
                    "paths": {
                        "raw": str(raw_file.relative_to(run_path)),
                        "metrics": str(metrics_file.relative_to(run_path)),
                        "costs": str(costs_file.relative_to(run_path)),
                    },
                }
            )

            cost_summary = outcome.cost_summary or {}
            entry["cost_summary"] = cost_summary
            entry["warnings"] = outcome.warnings

            aggregated_summary.append(
                {
                    "run": outcome.name,
                    "metrics": outcome.metrics,
                    "optimizer": outcome.optimizer_id,
                    "model": run_spec.model.id,
                    "cost_summary": cost_summary,
                    "warnings": outcome.warnings,
                }
            )

            for event in cost_events:
                enriched_event = dict(event)
                enriched_event["run"] = outcome.name
                aggregated_cost_events.append(enriched_event)

            totals = cost_summary.get("totals") or {}
            overall_costs["actual_usd"] += float(totals.get("actual_usd", cost_summary.get("actual_usd", 0.0)))
            overall_costs["economic_usd"] += float(totals.get("economic_usd", cost_summary.get("economic_usd", 0.0)))
            overall_costs["calls"] += int(totals.get("calls", 0))
            overall_costs["cache_hits"] += int(totals.get("cache_hits", 0))
            overall_costs["events"] += int(totals.get("events", cost_summary.get("events", 0)))

            phases = cost_summary.get("phases", {})
            for phase_name, phase_values in phases.items():
                accumulator = overall_phases.setdefault(
                    phase_name,
                    {
                        "actual_usd": 0.0,
                        "economic_usd": 0.0,
                        "prompt_tokens": 0,
                        "cached_prompt_tokens": 0,
                        "completion_tokens": 0,
                        "calls": 0,
                        "cache_hits": 0,
                    },
                )
                accumulator["actual_usd"] += float(phase_values.get("actual_usd", 0.0))
                accumulator["economic_usd"] += float(phase_values.get("economic_usd", 0.0))
                accumulator["prompt_tokens"] += int(phase_values.get("prompt_tokens", 0))
                accumulator["cached_prompt_tokens"] += int(phase_values.get("cached_prompt_tokens", 0))
                accumulator["completion_tokens"] += int(phase_values.get("completion_tokens", 0))
                accumulator["calls"] += int(phase_values.get("calls", 0))
                accumulator["cache_hits"] += int(phase_values.get("cache_hits", 0))

            aggregated_warning_messages.extend(outcome.warnings)
            aggregated_warning_details.extend(cost_summary.get("warnings", []))
            overall_program_calls += int(
                cost_summary.get("program_call_count")
                or cost_summary.get("program_calls")
                or 0
            )
        else:
            entry["status"] = "skipped"

        sub_run_entries.append(entry)

    aggregated_raw_path = run_path / outputs.raw_filename
    aggregated_summary_path = run_path / outputs.summary_filename
    aggregated_costs_path = run_path / "costs.jsonl"

    aggregated_raw_path.write_text("\n".join(json.dumps(r) for r in aggregated_raw), encoding="utf-8")
    aggregated_summary_path.write_text(json.dumps(aggregated_summary, indent=2), encoding="utf-8")
    if aggregated_cost_events:
        aggregated_costs_path.write_text("\n".join(json.dumps(event) for event in aggregated_cost_events), encoding="utf-8")
    else:
        aggregated_costs_path.write_text("", encoding="utf-8")

    for phase_name, phase_values in overall_phases.items():
        calls = phase_values.get("calls", 0)
        cache_hits = phase_values.get("cache_hits", 0)
        phase_values["cache_hit_ratio"] = (cache_hits / calls) if calls else 0.0

    program_calls_denominator = overall_program_calls or overall_costs["events"]
    program_cost_per_million = None
    program_economic_cost_per_million = None
    if program_calls_denominator:
        multiplier = 1_000_000 / program_calls_denominator
        program_cost_per_million = overall_costs["actual_usd"] * multiplier
        program_economic_cost_per_million = overall_costs["economic_usd"] * multiplier

    manifest = {
        "run_id": run_id,
        "experiment_name": config.experiment_name,
        "hypothesis": hypothesis,
        "status": status,
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "config_path": str(config_path),
        "outputs": {
            "root": str(run_path),
            "aggregated_raw": str(aggregated_raw_path.relative_to(run_path)),
            "aggregated_summary": str(aggregated_summary_path.relative_to(run_path)),
            "aggregated_costs": str(aggregated_costs_path.relative_to(run_path)),
        },
        "sub_runs": sub_run_entries,
        "costs": {
            "totals": overall_costs,
            "phases": overall_phases,
            "program_calls": program_calls_denominator,
            "program_cost_per_million_runs": program_cost_per_million,
            "program_economic_cost_per_million_runs": program_economic_cost_per_million,
            "warning_messages": aggregated_warning_messages,
            "warning_details": aggregated_warning_details,
            "paths": {
                "costs_jsonl": str(aggregated_costs_path.relative_to(run_path)),
            },
        },
    }

    if error_text:
        manifest["error"] = error_text

    run_path.mkdir(parents=True, exist_ok=True)
    manifest_path = run_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    ledger_entry = {
        "run_id": run_id,
        "experiment_name": config.experiment_name,
        "hypothesis": hypothesis,
        "status": status,
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "manifest": str(manifest_path.relative_to(root)),
        "cost_actual_usd": overall_costs["actual_usd"],
        "cost_economic_usd": overall_costs["economic_usd"],
    }

    ledger_path = root / "ledger.jsonl"
    with ledger_path.open("a", encoding="utf-8") as ledger_file:
        ledger_file.write(json.dumps(ledger_entry) + "\n")

    logging.info("Run %s artifacts stored in %s", run_id, run_path)


def main() -> None:
    raise SystemExit(run_cli())


if __name__ == "__main__":
    main()

