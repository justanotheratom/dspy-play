# LLM Cost Budgeting Proposal

## Goals
- Add guardrails so experiments halt once a configurable USD budget is exhausted.
- Track prompt vs completion token usage for both training and inference phases, including cached-token discounts and alternative billing tiers.
- Emit per-call costs plus rollups so runs can be compared and audited later.
- Keep pricing data centralized, versioned, and easy to refresh inside the library for repeatability.

## Constraints & Assumptions
- LiteLLM (and DSPy’s `LM`) exposes token usage fields on responses; some providers may omit them and require estimates. When usage is missing we must emit a visible warning and note that costs were estimated.
- Providers increasingly quote prices per 1M tokens; inputs, cached inputs, and outputs may have distinct rates, plus optional batch/flex tiers.
- `ModelConfig.pricing` already exists but is optional and not automatically populated today.
- Experiments may run multiple models per matrix; budget enforcement must consider cumulative spend across runs.
- Pricing can change over time, differ for fine-tuned variants, and vary by billing tier; the catalog must retain version history.
- Local LLM inference may carry zero marginal cost; accounting must represent actual spend as zero while still reporting economic estimates (uncached equivalents).
- We prioritise synchronous Python execution; distributed workers can adopt the same hooks later.

## Pricing Source of Truth
- Introduce `library/pricing/catalog.json` storing per-1M token rates keyed by provider/model/variant with fields for `input`, `cached_input`, `output`, optional tier identifiers (`batch`, `realtime`, `flex`, etc.), and optional provider-wide fine-tune premiums (`fine_tuned_input_per_1m`, `fine_tuned_output_per_1m`).
- Catalog entries include effective dates and optional expirations so we can audit historical spend after providers change rates; new entries append to history rather than mutating in place.
- Fine-tuned models reference base-model entries via an inheritance field; they reuse base pricing unless an override is provided or a provider-level premium is specified. Capture fine-tuning *training* prices separately (e.g., per 1K training tokens or per compute-hour) so manifests can report fine-tune job spend alongside inference.
- Provide a helper `pricing.lookup(model_config, when, *, fine_tuned: bool = False, tier: str | None = None)` that prefers inline `model_config.pricing` but falls back to the catalog (selecting the newest entry whose effective date ≤ run start timestamp). Raise clear errors if nothing matches and record when estimated pricing is used.
- When a run references a model without catalog coverage, prompt the user at runtime (CLI/interactive or via environment variables) to supply pricing data, then persist it to the catalog file with the current timestamp. This “just-in-time” update avoids maintaining a separate `dspylab pricing sync` command.
- Document how to pre-populate or auto-approve updates in CI (e.g., flag to fail instead of prompting) to keep headless workflows deterministic.

### Handling Price Changes Over Time
- Each catalog entry stores a `price_version` (monotonic integer) alongside `effective_at`/`expires_at` timestamps so multiple historical prices can coexist for the same provider/model/tier combo.
- Pricing lookups resolve to the newest entry whose `effective_at` ≤ the run’s start timestamp and (`expires_at` is null or ≥ start). This keeps historical runs auditable when providers change rates mid-project.
- When new provider rates arrive, append a fresh entry with an incremented `price_version`; existing entries stay immutable, preserving past run accounting.
- Allow configs to pin a specific `price_version` explicitly when reproducibility outweighs auto-selection (e.g., rerunning an experiment with legacy pricing).

### Pre-Run Cost Estimation
- Before execution, compute best-effort estimates leveraging catalog pricing, dataset sizes, and optional program metadata describing expected model calls per example (including multi-step chains).
- Produce dual projections per run: actual spend (respecting DSPy cache hits and zero-cost providers) and economic spend (assuming uncached calls). Compare both to budget thresholds in CLI output.
- Permit manual overrides when dataset sizes or per-example call counts are unknown; otherwise use conservative upper bounds derived from split lengths and configured `max_output_tokens`.
- Persist estimation data to manifests and ledger entries so realized costs can be checked against projections.

## Config Additions
- Extend experiment YAML schema with a top-level `budget` block (`max_usd`, optional `warn_usd`, optional `max_train_usd`, `max_infer_usd`).
- Allow each matrix entry to override the budget (useful for A/B runs) while the experiment-level cap remains authoritative.
- Support `pricing_id` on `ModelConfig` to reference catalog entries when per-project overrides are unnecessary.
- Emit validation errors when budgets exist without corresponding pricing metadata, preventing runs that would immediately violate guardrails.

## Runtime Instrumentation
- Wrap LiteLLM calls inside `library/runtime/dspy_runtime.configure_dspy_runtime` with a new `UsageTracker` that records phase (`train`/`infer`), prompt tokens, cached prompt tokens (when providers surface them), completion tokens, derived USD, model id, billing tier, timestamp, cache-hit status (local cache vs provider), and whether pricing was estimated due to missing usage (triggering warnings).
- During optimizer compilation (`ExperimentExecutor._maybe_optimize`) switch the tracker to `train` mode; evaluation (`_evaluate_program`) switches to `infer` mode. Reset to inference for any teacher model calls.
- Calculate two cost streams per call: actual spend (ignores locally cached inference calls and zero-cost providers) and economic cost (counts cached inference calls as if uncached). Both use `pricing.lookup`, defaulting cached-token rates when surfaced by providers.
- Bubble aggregated cost stats (actual vs economic) back into `RunOutcome` so downstream reporting can add them to summaries and manifests.

## Budget Enforcement Flow
- Introduce `BudgetManager` within `ExperimentExecutor` that consumes tracker deltas and checks against configured caps (actual spend only; economic projections remain advisory).
- Before issuing a provider call, predict the worst-case incremental spend (e.g., cap by `max_output_tokens`) to early-abort when a single request would overflow the remaining budget.
- After each completion, recompute totals; if any limit is breached, raise `BudgetExceededError` captured by the CLI to mark the run `cancelled` with reason `budget_exceeded`.
- Emit structured log warnings when spending crosses `warn_usd` to help humans intervene before hard stops.

## Output & Reporting Updates
- Append cost rollups to each run’s metrics output (`summary.json` gains actual spend metrics such as `train_prompt_tokens`, `train_cached_prompt_tokens`, `train_completion_tokens`, `train_cost_usd`, `infer_cost_usd`, `total_cost_usd`, plus economic estimates like `inference_economic_cost_usd` and aggregate program economics including `program_cost_per_million_runs`).
- Write a new artifact `artifacts/costs.jsonl` listing every model call with phase, tokens, actual vs economic USD, cache-hit flags, latency, provider metadata, and whether usage was estimated.
- Update manifest schema to capture experiment-level totals, per-model breakdowns, budget status (`within_budget`, `exceeded_threshold`, `cancelled_reason`), economic cost projections, pre-run cost estimates, and fine-tuning spend when applicable.
- Add CLI table columns (and docs) showing cost metrics alongside accuracy so runs can be compared by value vs spend, including projected cost per million program executions.

## Testing & Verification
- Unit tests cover pricing lookup fallbacks, fine-tune premium application, budget validation, tracker arithmetic (actual vs economic), cache-hit handling, and warning emission when usage is missing.
- Integration tests simulate training + inference passes, ensuring budget overrun triggers cancellation, manifests reflect the status, and pre-run estimates plus program-per-million projections align with recorded usage.
- Regression tests guard against providers that omit usage data by forcing estimation paths and verifying graceful degradation.
- Smoke test ensures `summary.json` remains backwards-compatible when cost fields are absent (legacy runs).

## Implementation Steps
- Land pricing catalog + helper utilities (including interactive prompts) and backfill existing configs with `pricing_id` references.
- Introduce `BudgetManager` and `UsageTracker`, wiring them through `ExperimentExecutor` and `RunOutcome`.
- Update schema/validation plus CLI to accept budgets, with docs guiding users on setting thresholds.
- Enhance output writers (summary, manifest, new costs log) and extend tests.
- Document operational playbook: updating prices, configuring budgets, interpreting cost reports.

## Open Questions
- Should budgets be enforced per experiment invocation, per run, or across rolling windows (e.g., daily quotas)?
- How should we estimate spend when providers omit or delay usage data (especially for streaming responses)?
- Do we need separate accounting for optimizer teacher-model calls versus student calls?
- What retention policy should apply to per-call cost logs to avoid bloating run artifacts?
- How should headless environments provide pricing data when interactive prompting is disabled (e.g., flags to fail fast vs pull from secrets managers)?


