# Architecture Overview

- **CLI Layer** `dspylab/__main__.py` exposes `dspylab run <config>` using `argparse` or `typer`. It validates CLI args, loads YAML, and hands control to the orchestration engine.
- **Config Engine** Module responsible for schema validation (via `jsonschema`), default resolution, and expansion of experiment combinations into discrete run specs. Outputs normalized `ExperimentPlan` objects.
- **Program Loader** Utility that imports user-specified Python module (e.g. `programs.preferencevalidator.01.program`) and fetches required callables: `build_program`, metric functions, dataset hooks. Uses `importlib` with informative error mapping.
- **Runtime Orchestrator** Core service that iterates over execution matrix, initializing DSPy strategies/optimizers per run, managing dataset splits, executing training/evaluation, capturing metrics, and writing artifacts.
- **Results Manager** Handles structured logging, raw result streaming (JSONL), aggregate accumulation, and summary file updates. Produces metadata (config hash, timestamps) for traceability.
- **Integrations** Thin shims around DSPy and LiteLLM to configure model providers using YAML settings. Provide provider registry with cost metadata and token accounting support.

# Execution Flow

1. **CLI Invocation** User runs `dspylab run path/to/experiment.yaml`.
2. **Config Validation** Load YAML, validate against JSON Schema, apply defaults. Produce normalized `ExperimentConfig`.
3. **Program Import** Resolve module path (from config), import, obtain program factory (`build_program()`), dataset loader, and metric functions. Validate presence + signature.
4. **Matrix Expansion** Enumerate run combinations (e.g., for each model × optimizer × strategy). For each run derive unique run id + output paths.
5. **Dataset Prep** Invoke dataset loader to get dev/test splits. Apply seeds and caching.
6. **Run Execution** For each run:
   - Configure DSPy settings (model provider via LiteLLM, strategies, optimizers).
   - Instantiate program via factory.
   - Run optimizer training on dev (if specified).
   - Evaluate on test set, gathering per-example outputs and metrics.
   - Stream raw results to JSONL file.
   - Compute aggregates (built-in + custom metrics) and append to shared summary file.
7. **Reporting** Emit CLI summary, durations, and exit status. Optionally expose a machine-readable exit artifact (e.g., `run_manifest.json`).

# Key Components

- `dspylab/config/schema.py` JSON Schema loader and validation helpers.
- `dspylab/config/models.py` Pydantic models mirroring schema for internal use.
- `dspylab/program/loader.py` Import utilities, caching, and signature checks.
- `dspylab/runtime/executor.py` Main orchestration engine.
- `dspylab/runtime/metrics.py` Built-in metric implementations + adapter for custom callables.
- `dspylab/runtime/output.py` Raw/aggregate writers with atomic file updates.
- `dspylab/cli.py` CLI entrypoint binding everything together.

# Configuration Layout

Example YAML structure:

```
experiment_name: preference_validator_baseline
program:
  module: programs.preferencevalidator.01.program
  factory: build_program
  dataset_loader: load_dataset
  metrics:
    accuracy: compute_accuracy
    custom_cost: compute_cost
models:
  - id: gpt4o
    provider: openai
    base_url: https://api.openai.com/v1
    api_key_env: OPENAI_API_KEY
    temperature: 0.3
    max_output_tokens: 512
    pricing:
      input_per_1k: 0.01
      output_per_1k: 0.03
strategies:
  - id: default_cot
    type: chain_of_thought
    params:
      enabled: true
optimizers:
  - id: bootstrap
    type: bootstrap_few_shot
    params:
      max_rounds: 3
matrix:
  - model: gpt4o
    optimizer: bootstrap
    strategy: default_cot
dataset:
  source: local
  path: data/dev_test.jsonl
  split:
    dev_ratio: 0.2
    seed: 42
outputs:
  root_dir: programs/preferencevalidator/01/results
logging:
  level: info
  verbose: true
```

# Results & Storage

- **Raw results** Each run writes `results/<experiment_name>/<run_id>/raw.jsonl` with one record per test example: input, model output, metrics, latency data, config snapshot.
- **Aggregate summary** Append-per-run entry in `results/<experiment_name>/summary.json` containing run id, configuration, aggregate metrics, costs, timestamps.
- **Optional extras** Provide run manifest, logs, and optional CSV export of aggregates.

# DSPy & LiteLLM Integration

- Use DSPy’s `dspy.configure()` to set model (via LiteLLM alias) and strategies before program execution.
- Implement provider registry to translate YAML provider blocks into LiteLLM configs (model name, base URL, headers, rate limits).
- Capture LiteLLM token usage callbacks for aggregate reporting.
- Ensure compatibility with DSPy optimizers like `dspy.optimizers.BootstrapFewShot` by mapping YAML params to constructor arguments.

# Testing Strategy

- **Unit Tests**
  - Schema validation (valid/invalid configs).
  - Program loader error cases (missing module, attribute).
  - Metric adapters (built-in metrics + custom stub).
  - Output manager ensures correct file writes.
- **Integration Tests**
  - Mock DSPy program with synthetic dataset to exercise full run pipeline.
  - CLI invocation test using `pytest` + `CliRunner` (if Typer) or `subprocess`.
  - Verify summary aggregation across multiple matrix combinations.
- **Smoke Tests**
  - Optional real API test gated by env var (requires API keys) to ensure provider compatibility.

# Roadmap Ideas

- Parallel execution across matrix combinations.
- Rich experiment dashboard (web UI) reading summary file.
- Support for additional metrics (cost/latency percentiles), caching datasets, and resuming partial runs.
- Plugin hooks for custom result sinks (e.g., Langfuse, Weights & Biases).


