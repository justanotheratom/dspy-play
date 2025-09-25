# Phase 1 · Project Scaffolding

- ▶ Schema Foundation
  - Define experiment YAML JSON Schema covering program module refs, model matrix, metrics.
  - Publish schema file under `library/config/schema.json` and generate sample config.
  - Add schema validation unit test (valid + invalid cases).
- ▶ CLI Skeleton
  - Scaffold `dspylab run` command (Typer or argparse).
  - Implement CLI options (`--config`, `--verbose`) and help text.
  - Wire CLI to schema loader; ensure validation errors surface cleanly.
- ▶ Config Models
  - Create Pydantic models mirroring schema for internal use.
  - Implement matrix expansion utility producing run specs.
  - Cover edge cases (single/multiple models, default parameters) with tests.

# Phase 2 · Runtime Core

- ▶ Program Loader ✅ Completed in commit `feat: add program loader utilities`.
- ▶ DSPy Integration ✅ Completed in commit `feat: configure dspy runtime with liteLLM settings`.
- ▶ Execution Engine ✅ Completed in commit `feat: add experiment executor and tests`.

# Phase 3 · Metrics & Output

- ▶ Built-in Metrics ✅ Added timing collectors in `library/metrics` with unit tests.
- ▶ Results Management ✅ Raw + summary serialization handled via executor metrics aggregation.
- ▶ Example Assets (in progress)
  - Create sample `program.py` with stub dataset + metrics.
  - Provide minimal dataset fixture for docs/tests.

# Phase 4 · Quality & Docs

- ▶ Testing Strategy
  - Add integration test invoking CLI with sample config + mock DSPy.
  - Cover failure scenarios (bad module ref, missing API key).
  - Establish CI workflow running lint + tests (GitHub Actions or similar).
- ▶ Documentation
  - Expand README with quickstart, config reference, troubleshooting.
  - Document schema, CLI options, output file formats.
  - Add roadmap notes (parallelism, dashboards) for future work.


