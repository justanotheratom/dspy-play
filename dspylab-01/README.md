# DSPyLab

`dspylab` is a CLI-first toolkit that turns DSPy experiments into reproducible, declarative workflows. Experiments are described with YAML configs that reference Python program modules, metrics, and optimizer settings.

## For Users

### Prerequisites
- Python 3.10+
- Access to a supported LLM provider (Groq, OpenAI, etc.) with API key

### Installation
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
```

### Configure Credentials
Add your provider key to `.env` (or export it directly):
```bash
echo "GROQ_API_KEY=sk-your-key" >> .env
source .venv/bin/activate
export GROQ_API_KEY=sk-your-key  # if you prefer environment variables
```

### Run an Experiment
```bash
source .venv/bin/activate
python -m library run programs/preferencevalidator/preference_validator.yaml
```
This will:
- Load the program at `programs/preferencevalidator/01/program.py`
- Configure the Groq `openai/gpt-oss-20b` model
- Compile the labeled-few-shot optimizer using the dataset in `programs/preferencevalidator/01/data/preferencevalidatordataset.jsonl`
- Write outputs to `programs/preferencevalidator/01/results/raw.jsonl` and `summary.json`

### Customize
1. Duplicate `programs/preferencevalidator` to a new directory.
2. Update the program code (metrics, strategies, optimizer settings).
3. Point a new YAML config at your program module and dataset.
4. Run `python -m library run <your-config>.yaml`.

Refer to `library/config/schema.json` for the full config schema (validated automatically).

## For Contributors

### Project Layout
```text
library/
  cli.py              # CLI entrypoint
  config/             # JSON schema + pydantic models
  program/            # Program loader utilities
  runtime/            # DSPy runtime integration & executor
programs/             # Example programs + datasets + results
  preferencevalidator/
    01/
      program.py
      data/
      results/
    preference_validator.yaml
 tests/               # Unit tests
 docs/implementation/ # Requirements, design, todos
```

### Common Tasks
- Install dev deps: `pip install -e '.[dev]'`
- Run tests: `pytest`
- Lint (optional): `ruff check .` (if installed)
- Update schema: edit `library/config/schema.json` and regenerate docs/examples as needed

### Adding Features
1. Create/modify code under `library/`.
2. Add or update tests in `tests/`.
3. Run `pytest` to confirm passing.
4. Update `README.md` or docs if behavior changes.
5. Commit with a descriptive message and open a PR.

### Experiment Assets
- Keep datasets in `programs/<name>/<version>/data/` (JSONL or other formats handled by the executor).
- Results should be written under `programs/<name>/<version>/results/` (automatically created by the CLI).
- Ensure configs checked into version control point to relative paths inside the repository.

### Releasing Changes
- After merging, run `git pull` locally and tag if appropriate.
- Publish to PyPI (future work) once build pipeline is in place.

## Roadmap
- Multi-run orchestration and parallel execution
- Richer result summaries and visualization hooks
- Additional built-in metrics and evaluator interfaces
- CI pipeline for lint/tests and packaged releases


