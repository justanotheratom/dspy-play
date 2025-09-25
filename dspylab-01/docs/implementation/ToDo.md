# Immediate Tasks

- Schema: Define experiment YAML JSON Schema, publish in repo, generate example config.
- CLI: Scaffold `dspylab run` command (Typer/argparse), wire to config loader, add help output.
- Config Engine: Implement parser + matrix expansion using Pydantic models.
- Program Loader: Build import utilities with unit tests for missing modules/functions.

# Near-Term

- DSPy Integration: Wrap `dspy.configure` with provider registry, ensure LiteLLM settings map correctly.
- Metrics: Implement built-in latency/TTFT/tokens-per-second metrics; design adapter for custom metrics.
- Output System: Create raw JSONL writer and aggregate summary manager with schema.
- Example Program: Provide sample `program.py` showcasing program factory, dataset loader, metric functions.

# Testing & Docs

- Tests: Add unit tests for config parsing, loader, metrics; integration test for CLI run using mock program/dataset.
- Documentation: Write README sections (quickstart, config reference) and link to schema.
- Dev Tooling: Set up CI pipeline (lint, tests) and pre-commit hooks if desired.


