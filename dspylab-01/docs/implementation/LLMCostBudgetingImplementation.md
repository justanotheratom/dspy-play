# LLM Cost Budgeting Implementation Plan

## Scope
This plan translates the approved budgeting proposal into actionable engineering work. It covers pricing catalog management, configuration updates, runtime instrumentation, budget enforcement, reporting, interactive prompts, testing, and documentation. Every deliverable below must be completed; no placeholders or mock-only implementations are acceptable.

## High-Level Milestones
1. Pricing catalog infrastructure and interactive population flow.
2. Configuration/schema updates supporting budgets and pricing references.
3. Runtime instrumentation for token usage and USD calculation.
4. Budget enforcement and error handling.
5. Reporting, artifact updates, and manifest integration.
6. CLI/UX adjustments for prompting and headless modes.
7. Comprehensive automated tests.
8. Documentation and operational playbooks.

## Detailed Work Breakdown

### 1. Pricing Catalog Infrastructure
- Create `library/pricing/catalog.yaml` and `library/pricing/__init__.py` with a `PricingCatalog` class supporting:
  - Loading/writing YAML entries validated via Pydantic or dataclass models capturing provider, model, tier/variant, `price_version`, `effective_at`, optional `expires_at`, `input_per_1m`, `cached_input_per_1m`, `output_per_1m`, provider-wide fine-tune premiums (`fine_tuned_input_per_1m`, `fine_tuned_output_per_1m`), fine-tune training rates, and optional `notes`.
  - Append-only updates; never mutate historical entries.
  - Lookup by `(pricing_id or provider/model/variant, timestamp)` returning the most recent effective entry, flagging when pricing is estimated rather than exact, and surfacing provider tier/fine-tune premiums automatically.
  - Optional explicit selection by `price_version` plus ability to pin a specific version.
- Implement file-level locking to avoid concurrent catalog writes.
- Provide helpers to convert between per-token and per-1M token pricing and to derive per-call estimates.
- Unit tests: round-trip serialization, history append, time-based lookups, version pinning, fine-tune premium inheritance, estimation flag propagation.

### 2. Interactive Catalog Population
- Implement `prompt_for_pricing(model_config)` to gather input/cached/output rates, fine-tune premiums, and training costs via CLI prompts with validation.
- Support provider-wide fine-tune premiums without enumerating individual fine-tune model IDs; prompts should clarify inference vs training pricing fields.
- Integrate prompt into run startup when `pricing.lookup` fails; fall back to environment variables for non-interactive runs; respect `DSPYLAB_PRICING_NONINTERACTIVE=1` by failing fast with guidance.
- Persist newly gathered entries to the catalog with timestamp, `price_version`, operator metadata, and flag when values are estimates. Log inserts in run output.
- Tests: simulate interactive prompts via monkeypatch, environment variable overrides, non-interactive failure paths.

### 3. Schema and Config Enhancements
- Update JSON schema and Pydantic models to add `pricing_id`, optional `price_version`, and `fine_tuned` boolean on `ModelConfig` plus program metadata like `estimated_calls_per_example` and optional `estimated_training_calls`.
- Extend experiment schema with `budget` structure (`max_usd`, `warn_usd`, `max_train_usd`, `max_infer_usd`).
- Maintain backward compatibility with sensible defaults; provide schema validation tests covering new fields and error messages when pricing/budget data missing.
- Document migration steps for adding pricing IDs, marking fine-tuned models, and optionally supplying call-count hints.

### 4. Usage Tracking Instrumentation
- Build `UsageTracker` capturing per-call data: timestamps, phase (`train`/`infer`), provider/model, tier, prompt/cached/completion token counts, actual USD, economic USD (treating cached inference as uncached), cache-hit indicator (distinguish local cache vs provider cache), usage-missing indicator, raw provider usage payload.
- Hook tracker into `configure_dspy_runtime` to wrap LM calls; ensure `_record_latency_metrics` forwards usage details and emits warnings when usage fields are absent.
- During optimizer compile, switch tracker to training phase; revert to inference afterward; ensure teacher-model invocations are tagged appropriately.
- Aggregation methods: per-phase actual/economic totals, counts, averages, cache hit ratios, and estimated vs actual deltas.
- Unit tests using fake LM responses verifying token extraction, cache handling, dual-cost calculations, warning emission, and zero-cost provider handling.

### 5. Budget Enforcement Logic
- Implement `BudgetManager` to consume tracker events and enforce experiment-level budgets on actual spend; economic metrics remain advisory.
- Provide `register_expected_call` (preflight) and `record_call` (post-call) APIs; preflight uses `max_output_tokens` and pricing, respecting fine-tune premiums and cached-token discounts when available.
- Integrate with executor to catch `BudgetExceededError`, mark run cancelled with reason, emit summary of spend vs budget, and optionally halt remaining runs.
- Tests covering budget warnings, hard stops, accurate spend recording, and confirmation that economic costs do not affect enforcement.

### 6. Reporting and Artifacts
- Extend `RunOutcome` with `cost_summary` capturing per-phase tokens/costs, totals, economic projections, budget status, warning flags, cache statistics, and per-program cost scalars.
- Update summary writer (`summary.json`) with actual metrics (`train_prompt_tokens`, `train_cached_prompt_tokens`, `train_completion_tokens`, `train_cost_usd`, `infer_cost_usd`, `total_cost_usd`) and economic metrics (`inference_effective_cost_usd`, `program_cost_per_million_runs`, `inference_cost_per_million_calls`).
- Emit new `artifacts/costs.jsonl` containing each call’s usage, costs (actual/economic), cache-hit status, tier, estimation flag, and provider metadata; ensure outputs are gzip-friendly.
- Manifest updates: include budget config, pre-run estimates, per-model spend, economic projections, fine-tune spend (inference + training), usage warnings, and cost-per-million program projections.
- Update ledger/aggregation scripts to index new metrics for dashboards and comparisons.
- Tests verifying artifact schema, summary integration, manifest validation, and per-million calculations.

### 7. CLI/UX Adjustments
- Display pre-run cost estimates (actual/economic) before execution; allow abort if estimates exceed budget.
- Emit warnings when actual spend crosses `warn_usd`, when usage data is missing (estimated), or when local cache skew requires economic adjustment.
- Render final table summarizing actual vs economic costs, cache hits, cost-per-million program executions, and budget status.
- Add CLI flags/env vars for pricing prompts, non-interactive mode, optional continuation after budget breach, and toggles for pre-run estimation verbosity.
- Structured logging differentiating training/inference spend, fine-tune spend, cache activity, and estimation fallbacks.
- Tests: snapshot CLI output, verify warnings/estimates, non-interactive errors, pre-run estimate rendering.

### 8. Automated Testing Suite
- Unit tests covering catalog, schema validation, tracker dual-cost math, budget manager, CLI utilities, warning paths, and economic projections.
- Integration tests orchestrating end-to-end runs with mocked LMs: success within budget, warn thresholds, hard stop, missing usage requiring estimation, DSPy cache hits, zero-cost providers, fine-tuned model pricing.
- Regression tests for providers lacking cached token metrics, for local zero-cost models, and for price-version pinning.
- Optional performance tests ensuring tracker and logging overhead remains acceptable.

### 9. Documentation & Operational Playbook
- Update proposal docs with references to implementation details and clarifications (e.g., economic vs actual accounting).
- Author user documentation: configuring budgets/pricing IDs, interactive prompt workflow, headless configuration, interpreting economic vs actual costs, per-million program cost analysis, fine-tune accounting, handling missing usage warnings.
- Provide release notes and migration/operational guidance, including updating catalog entries when provider prices change and auditing historical cost data.

## Dependencies and Ordering
1. Pricing catalog infrastructure (sections 1–2) must land before runtime instrumentation to guarantee lookups succeed.
2. Schema/config updates (section 3) must precede executor changes to ensure configs parse.
3. Instrumentation (section 4) unlocks budget enforcement (section 5).
4. Reporting (section 6) depends on tracker data structures.
5. CLI adjustments (section 7) follow budget enforcement to reference real data.
6. Tests (section 8) run concurrently with each feature but integration tests require prior components.
7. Documentation (section 9) finalizes once features stabilized.

## Acceptance Criteria
- Experiments with valid pricing and budgets execute successfully, produce cost artifacts, and stay within budget when possible.
- Missing pricing triggers interactive prompt (or fails in non-interactive mode) and catalog records new entries with versioning metadata.
- Budgets enforced pre- and post-call with run cancellation when exceeded. Summaries/manifests/ledger contain accurate spend figures.
- Full automated test suite passes locally and in CI.
- Documentation updated, including migration steps and operational guidelines.

---

# Prior Article Implementation Checklist

- [ ] **Phase 0 – Pricing Groundwork**
  - [ ] Implement pricing catalog module with append-only history, lookup API, fine-tune premium support, estimation flags, and locking.
  - [ ] Add interactive and environment-variable-driven pricing population workflow (covering fine-tune and training costs) integrated into run startup.
  - [ ] Update config schema/models to support budgets, pricing IDs, price version pinning, fine-tuned markers, and program call-count metadata; add validation tests and migration guidance.

- [ ] **Phase 1 – Runtime Accounting Core**
  - [ ] Build usage tracker capturing phases, cached tokens, actual vs economic USD, cache detection, and usage warnings; integrate with runtime hooks.
  - [ ] Develop budget manager with preflight/post-call enforcement on actual spend, wiring into executor control flow and cancellation handling.

- [ ] **Phase 2 – Reporting & UX**
  - [ ] Extend run outcomes, summary writer, manifest, ledger, and `costs.jsonl` artifact with actual/economic metrics, cache stats, per-million program cost, budget status, and warnings.
  - [ ] Enhance CLI UX with pre-run estimates, missing-usage and budget warnings, final actual vs economic cost summaries, and pricing prompt flags.

- [ ] **Phase 3 – Quality Net**
  - [ ] Write comprehensive unit, integration, regression, and optional performance tests covering catalog, pricing estimates, tracking, budgets, caching, and outputs.
  - [ ] Produce user-facing documentation, operational playbooks, and release notes detailing configuration, prompts, estimation warnings, and auditing.

