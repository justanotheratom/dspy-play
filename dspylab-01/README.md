# DSPyLab

`dspylab` is a CLI-first toolkit that makes it easy to configure and run reproducible experiments with [DSPy](https://github.com/stanfordnlp/dspy). Experiments are described via declarative YAML files that reference Python program modules and metrics.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'

dspylab run docs/examples/preference_validator.yaml
```

## Features

- YAML schema covering models, strategies, optimizers, datasets, and outputs
- Program loader that imports user-defined modules and metrics
- Execution engine coordinating DSPy optimizers and evaluation loop
- Built-in latency metrics (latency, TTFT, tokens-per-second)
- Results written per-run with aggregate summaries (planned)

## Project Structure

```text
library/
  config/      # schema, loader, pydantic models
  metrics/     # built-in metrics utilities
  program/     # dynamic import helpers
  runtime/     # DSPy integration + execution engine
programs/
  preferencevalidator/01/  # sample program module
tests/          # unit tests covering components and CLI
docs/implementation # requirements, implementation plans, todos
```

## Development

Run the test suite:

```bash
pytest
```

## Roadmap

- Rich results management with JSONL exports
- Parallel run execution across matrices
- CLI-driven dry runs and experiment summaries
- Additional built-in metrics and visual dashboards


