# Experiment Tracking & Lineage Proposal

## Objectives

- Ensure every experiment run is reproducible, comparable, and explainable.
- Capture the context around each run: hypothesis, configuration, code, dataset, environment, outputs, and ancestry.
- Provide tooling to visualize the lineage (parent/child runs) and highlight metric deltas for theory-driven iteration.

## Core Concepts

- **Run ID**: Timestamped slug (e.g. `2025-09-26T12-30-05Z-abc123`) that keys every artifact and manifest.
- **Manifest**: A JSON file stored with each run that records metadata, inputs, outputs, and status.
- **Ledger**: Aggregated index of all manifests (CSV, SQLite, or external tracker) used for comparisons, reporting, and lineage graphs.
- **Parent Run**: Optional identifier; every run except the root declares the run it extends, enabling an explicit experiment tree.
- **Program Version**: High-level task identifier (e.g. `preference_validator/01`). All runs are scoped to a specific program version. Only runs within the same program version are directly comparable because the conceptual input/output contract is guaranteed consistent. If that contract changes, bump the version (e.g. `01 → 02`) and start a new lineage tree under the new version.

## Run Wrapper Responsibilities

- **Pre-run capture**
  - Record `git rev-parse HEAD`, plus a dirty flag and the textual diff (`git diff`, `git diff --cached`).
  - Snapshot working-tree copies of critical files (e.g. `programs/preferencevalidator/01/program.py`, `programs/preferencevalidator/preference_validator.yaml`, helper modules). Save under `artifacts/config/`.
  - Compute SHA-256 hashes for those files and include them in the manifest.
  - Obtain dataset version info (see "Dataset Versioning").
  - Require CLI parameters for `--hypothesis` and `--parent-run` (the latter optional for root runs). Refuse to continue if hypothesis is missing.

- **Execution**
  - Invoke `python -m library run programs/preferencevalidator/preference_validator.yaml` (or target path).
  - Stream logs to `artifacts/logs/run.log` while tee-ing to stdout.

- **Post-run capture**
  - Copy DSPy outputs (`raw.jsonl`, `summary.json`, compiled programs, checkpoints) into `artifacts/results/`.
  - Record final metrics, duration, and environment details (Python version, DSPy version, platform) in the manifest.
  - Mark `status` as `succeeded`, `failed`, or `cancelled`. For failures, include stack trace and partial artifacts.
  - Write `manifest.json` containing all metadata described above.

## Manifest Schema (draft)

```json
{
  "run_id": "2025-09-26T12-30-05Z-abc123",
  "parent_run_id": "2025-09-25T18-14-30Z-xyz789",
  "hypothesis": "Raise metric_threshold to improve precision",
  "status": "succeeded",
  "timestamps": {
    "started": "2025-09-26T12:30:05Z",
    "ended": "2025-09-26T13:04:41Z"
  },
  "git": {
    "commit": "d34db33f",
    "is_dirty": true,
    "unstaged_diff_file": "artifacts/config/git-diff.txt",
    "staged_diff_file": "artifacts/config/git-diff-staged.txt"
  },
  "config_files": [
    {
      "path": "programs/preferencevalidator/preference_validator.yaml",
      "copy": "artifacts/config/preference_validator.yaml",
      "sha256": "..."
    },
    {
      "path": "programs/preferencevalidator/01/program.py",
      "copy": "artifacts/config/program.py",
      "sha256": "..."
    }
  ],
  "dataset": {
    "id": "pref-validator@1.2.0",
    "hash": "...",
    "dvc_pointer": "artifacts/config/preferencevalidatordataset.jsonl.dvc"
  },
  "environment": {
    "python": "3.11.8",
    "dspy": "0.4.0",
    "platform": "darwin-24.6.0"
  },
  "parameters": {
    "num_threads": 2,
    "metric_threshold": 0.8,
    "teacher_model": "groq-meta-llama-maverick"
  },
  "metrics": {
    "accuracy": 0.71,
    "cosine_similarity": 0.78
  },
  "artifacts": [
    { "name": "summary", "type": "json", "path": "artifacts/results/summary.json", "sha256": "..." },
    { "name": "raw_predictions", "type": "jsonl", "path": "artifacts/results/raw.jsonl", "sha256": "..." }
  ],
  "rerun_of": null,
  "notes": "Observed slight recall drop; precision improved."
}
```

## Ledger & Visualization

- Parse manifests into a central ledger (CSV or SQLite). Keep columns for run_id, parent_run_id, hypothesis, key metrics, dataset id/hash, config hash, status.
- Provide helper scripts to:
  - List runs sorted by time or lineage branch.
  - Produce diff reports between two run_ids (config, metrics, dataset).
  - Render a lineage tree (e.g. convert ledger into Graphviz DOT, output SVG) showing hypotheses along each branch.
  - Drive hierarchical views by grouping on program, version, config_id, and optional branch labels (e.g. `bootstrap/noteacher`). This enables a dashboard structure such as:

```
program_foo
  /01
    /bootstrap
      /noteacher
    /gepa
      /metricfunction1
      /metricfunction2
program_bar
  /01
    /...
  /02
    /...
```

  Each leaf node represents the root run for a lineage tied to a specific config file; descendants retain their parent relationships within that subtree.

- Track deployment promotions by attaching environment badges to ledger entries. When a run’s artifacts are promoted to `dev`, `test`, `prod`, etc., append a record in a `promotions` table (run_id, environment, timestamp, promoting user, notes). Expose promotion status alongside lineage so it’s obvious which branch is powering each environment. Optionally enforce that only `succeeded` runs with attached artifacts can be promoted.

## Handling Failures & Reruns

- Always emit a manifest, even when the run fails (status `failed`). Include the exception text in `notes` and preserve partial outputs.
- Allow reruns to specify `--rerun-of <run_id>`; include both the parent/ancestor (for lineage) and rerun pointer (for variance analysis).
- Provide tooling to aggregate metrics across reruns of the same configuration (mean, stddev) to understand stochastic variation.

## Dataset Versioning

- Adopt DVC for dataset control:
  - `dvc init` in the repo root.
  - `dvc add programs/preferencevalidator/01/data/preferencevalidatordataset.jsonl` to create a `.dvc` pointer file tracked by git.
  - Store the dataset either in-repo (for small files) or push to a remote (`dvc remote add` + `dvc push`).
- Record in the manifest:
  - The DVC hash/version.
  - Location of the pointer file copy.
- Updating the dataset (adding/removing examples or applying human-reviewed fixes) produces a new DVC version; runs referencing that version inherit its ID, letting us correlate metric shifts with data changes.

## Artifact Management

- Mirror all DSPy outputs and produced models into the run directory under `artifacts/`.
- Generate an `artifacts/index.json` capturing file names, types, hashes, sizes, and loading instructions.
- For compiled DSPy programs or finetuned checkpoints, include metadata about the source run and intended use (e.g. "candidate best-of" vs "production candidate").

## Optional Integrations

- **MLflow or Weights & Biases**: plug manifests into external platforms for dashboards, metric charts, and collaborative annotations. Treat external IDs as additional fields in the manifest.
- **Automation Hooks**: after each run, trigger scripts that post summaries to Slack/Notion or update a shared dashboard.

## Hosted Deployment Outline

- **Control plane & storage**: Run manifests, lineage metadata, promotions, and user identities live in a managed relational database (e.g. Postgres on RDS/Cloud SQL). A containerized API service (FastAPI/GraphQL) provides authenticated access for the dashboard and CLI.
- **Experiment execution**: Jobs are enqueued via the API and consumed by container workers (ECS/Fargate, Cloud Run, or Kubernetes) that bundle the CLI library, pull the pinned config/dataset/artifacts, and invoke the DSPy run.
- **LLM logging**: Raw provider requests/responses stream to object storage (S3/GCS) partitioned by run_id. Metadata such as token counts or error codes is indexed in the database for search/debugging.
- **Dataset versions**: Store large datasets in a DVC-compatible remote (object storage or LakeFS). Workers `dvc pull` the requested hash before execution; dataset IDs/hashes are mirrored in the run manifest.
- **Run outputs & artifacts**: Persist raw/summary results, compiled programs, finetune checkpoints, and logs under `s3://.../runs/<program>/<version>/<run_id>/`. Maintain an artifact index table mapping run_id → artifact URIs, checksum, size, and purpose.
- **Dashboard UI**: Host a React/Next.js front-end (Vercel, Netlify, or static hosting behind CloudFront) that consumes the API to render lineage trees, diff reports, promotions, and environment mappings.
- **CLI distribution**: Package the run wrapper as a pip-installable tool (publish to PyPI or an internal registry). The CLI authenticates to the hosted API, submits runs, fetches manifests, and can pull artifacts/logs.

## Next Steps

1. Implement the run wrapper CLI (Python script) and manifest schema.
2. Backfill existing runs by retroactively creating manifests where possible.
3. Introduce the ledger (start with CSV, upgrade to SQLite if needed).
4. Initialize DVC, version the dataset, and educate contributors on the workflow (`dvc pull`, `dvc push`).
5. Prototype lineage visualization and diff reporting scripts.
6. Iterate on automation (hypothesis enforcement, rerun grouping, artifact indexing) as the workflow stabilizes.


