# LLM Cost Budgeting Proposal

## Goals
- Add guardrails so experiments halt once a configurable USD budget is exhausted.
- Track prompt vs completion token usage for both training and inference phases.
- Emit per-call costs plus rollups so runs can be compared and audited later.
- Keep pricing data centralized and versioned inside the library for repeatability.

## Constraints & Assumptions
- LiteLLM (and DSPy’s `LM`) exposes token usage fields on responses; some providers may omit them and require estimates.
- `ModelConfig.pricing` already exists but is optional and not automatically populated today.
- Experiments may run multiple models per matrix; budget enforcement must consider cumulative spend across runs.
- We prioritise synchronous Python execution; distributed workers can adopt the same hooks later.

## Pricing Source of Truth
- Introduce `library/pricing/catalog.yaml` storing published per-1k prompt and completion prices keyed by provider/model.
- Provide a helper `pricing.lookup(model_config)` that prefers inline `model_config.pricing` but falls back to the catalog and raises if neither is present.
- Add documentation on updating the catalog and scripts/tests that fail when required entries are missing.
- CLI gains `dspylab pricing sync` (optional future work) to refresh the catalog from provider APIs when available.

## Config Additions
- Extend experiment YAML schema with a top-level `budget` block (`max_usd`, optional `warn_usd`, optional `max_train_usd`, `max_infer_usd`).
- Allow each matrix entry to override the budget (useful for A/B runs) while the experiment-level cap remains authoritative.
- Support `pricing_id` on `ModelConfig` to reference catalog entries when per-project overrides are unnecessary.
- Emit validation errors when budgets exist without corresponding pricing metadata, preventing runs that would immediately violate guardrails.

## Runtime Instrumentation
- Wrap LiteLLM calls inside `library/runtime/dspy_runtime.configure_dspy_runtime` with a new `UsageTracker` that records phase (`train`/`infer`), prompt tokens, completion tokens, derived USD, model id, and timestamp.
- During optimizer compilation (`ExperimentExecutor._maybe_optimize`) switch the tracker to `train` mode; evaluation (`_evaluate_program`) switches to `infer` mode. Reset to inference for any teacher model calls.
- Calculate USD via `pricing.lookup` and store both per-call facts (for logs) and aggregated totals in the tracker.
- Bubble aggregated cost stats back into `RunOutcome` so downstream reporting can add them to summaries and manifests.

## Budget Enforcement Flow
- Introduce `BudgetManager` within `ExperimentExecutor` that consumes tracker deltas and checks against configured caps.
- Before issuing a provider call, predict the worst-case incremental spend (e.g., cap by `max_output_tokens`) to early-abort when a single request would overflow the remaining budget.
- After each completion, recompute totals; if any limit is breached, raise `BudgetExceededError` captured by the CLI to mark the run `cancelled` with reason `budget_exceeded`.
- Emit structured log warnings when spending crosses `warn_usd` to help humans intervene before hard stops.

## Output & Reporting Updates
- Append cost rollups to each run’s metrics output (`summary.json` gains `train_prompt_tokens`, `train_completion_tokens`, `train_cost_usd`, `infer_cost_usd`, `total_cost_usd`).
- Write a new artifact `artifacts/costs.jsonl` listing every model call with phase, tokens, USD, latency, and provider metadata for fine-grained analysis.
- Update manifest schema to capture experiment-level totals, per-model breakdowns, and budget status (`within_budget`, `exceeded_threshold`, `cancelled_reason`).
- Add CLI table columns (and docs) showing cost metrics alongside accuracy so runs can be compared by value vs spend.

## Testing & Verification
- Unit tests cover pricing lookup fallbacks, budget validation, and tracker arithmetic with mocked LiteLLM responses.
- Integration tests simulate training + inference passes, ensuring budget overrun triggers cancellation and manifests reflect the status.
- Regression tests guard against providers that omit usage data by forcing estimation paths and verifying graceful degradation.
- Smoke test ensures `summary.json` remains backwards-compatible when cost fields are absent (legacy runs).

## Implementation Steps
- Land pricing catalog + helper utilities and backfill existing configs with `pricing_id` references.
- Introduce `BudgetManager` and `UsageTracker`, wiring them through `ExperimentExecutor` and `RunOutcome`.
- Update schema/validation plus CLI to accept budgets, with docs guiding users on setting thresholds.
- Enhance output writers (summary, manifest, new costs log) and extend tests.
- Document operational playbook: updating prices, configuring budgets, interpreting cost reports.

## Open Questions
- Should budgets be enforced per experiment invocation, per run, or across rolling windows (e.g., daily quotas)?
- How should we estimate spend when providers omit or delay usage data (especially for streaming responses)?
- Do we need separate accounting for optimizer teacher-model calls versus student calls?
- What retention policy should apply to per-call cost logs to avoid bloating run artifacts?


