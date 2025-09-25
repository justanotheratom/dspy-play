# Overview

- **Goal** Develop `dspylab`, a CLI-first library that makes configuring and running DSPy experiments reproducible via declarative YAML configs and reusable Python programs.
- **Scope** Provide config schema, execution runtime, metric tracking, and experiment output management for single-program experiments. Users supply DSPy program modules and metric functions; `dspylab` orchestrates experiment execution across model providers and strategies.
- **Out of scope** Building new DSPy optimizers, hosting LiteLLM proxies, dataset curation tooling, or general-purpose orchestration beyond DSPy experiments.

# Functional Requirements

- **CLI entrypoint** Provide `dspylab run <config.yaml>` (accessible after installing the library) to execute an experiment configuration.
- **Config loading** Parse experiment YAML describing model provider, DSPy runtime options, strategies, optimizers, datasets, metrics, and execution matrix (combinations). Validate against a published JSON Schema.
- **Program integration** Allow YAML to reference a Python module path and callables (program function, metrics, auxiliary hooks). Load user-supplied module dynamically, with helpful error reporting if entry points are missing.
- **Experiment matrix** Support running multiple parameter combinations defined in YAML (e.g., different models × optimizers × strategies). Each combination is treated as a distinct run with shared dataset splits.
- **Model providers** Support at least OpenAI-compatible APIs via DSPy’s built-in LiteLLM integration. Allow specifying provider name, base URL, API key env var, temperature, max tokens, and cost metadata per model.
- **Strategies & optimizers** Allow configuring DSPy strategies (chain-of-thought, retry, best-of-n, etc.) and optimizers (Labeled Few-Shot, Bootstrap Few-Shot, GEPA, …) with their parameters. Ensure multiple metrics can be associated with optimizers.
- **Dataset selection** Allow YAML to reference dataset loaders or file paths. Support dev/test split configuration and seeding for reproducibility.
- **Metrics** Provide built-in latency tracking and support custom metric callables exported from the program module.
- **Execution lifecycle** For each run: initialize DSPy settings, load program, execute optimizer training (if any) on dev split, evaluate on test split, record per-example and aggregate results.
- **Output files** Emit raw per-instance results to a run-specific file (JSONL/CSV) and append aggregate metrics for each run to a shared summary file. Include metadata (config hash, timestamps, dataset info).
- **Logging & progress** Provide structured logging (to stdout and optional file) for config resolution, run progress, and metrics. Support verbose flag.
- **Error handling** Fail gracefully with actionable messages for schema validation errors, import failures, missing API keys, or run-time DSPy exceptions. Return non-zero exit code on failure.

# Non-Functional Requirements

- **Extensibility** Design config schema and execution pipeline to accept new strategies, optimizers, metrics, and providers without breaking existing configs.
- **Reproducibility** Ensure deterministic behavior when seeds are provided (dataset shuffling, optimizer randomization). Capture git commit (if available) and config fingerprints.
- **Performance** Efficiently batch runs; avoid unnecessary reinitialization when reusing datasets or models. Support parallel execution where practical (future work but avoid blocking design).
- **Usability** Provide schema docs, annotated example configs, and clear CLI help. Error messages should explain remediation steps.
- **Compatibility** Support macOS and Linux environments with Python ≥3.10. Respect environment variables for API keys; no hard-coded secrets.
- **Testing** Include unit tests for config parsing/validation, CLI behavior, and dataset/metric integration plus integration smoke tests with mocked DSPy modules.
- **Documentation** Publish README and docs with architecture overview, configuration reference, and troubleshooting guide.

# Open Questions / Risks

- Need confirmation on preferred raw vs aggregate output formats (JSONL vs CSV). Current assumption: raw results per run in JSONL; shared aggregates in JSON.
- Parallel or distributed execution requirements unclear; may need future enhancement.
- Dataset ingestion variability (local files vs HF datasets) may require plugin system.


