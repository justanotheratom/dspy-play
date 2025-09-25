"""Sample DSPy program for testing the execution engine."""

from __future__ import annotations

from typing import Dict, Iterable, List

from library.metrics import TimedCompletion, compute_latency_metrics


def build_program():  # pragma: no cover - exercised via executor tests
    completions: List[TimedCompletion] = []

    def program(example: Dict[str, object]) -> Dict[str, object]:
        completion = TimedCompletion(
            started_at=0.0,
            first_token_at=0.05,
            finished_at=0.2,
            prompt_tokens=10,
            completion_tokens=5,
        )
        completions.append(completion)
        return {
            "prediction": example.get("label", 0),
            "timing": {
                "started_at": completion.started_at,
                "first_token_at": completion.first_token_at,
                "finished_at": completion.finished_at,
                "prompt_tokens": completion.prompt_tokens,
                "completion_tokens": completion.completion_tokens,
            },
        }

    program.completions = completions  # type: ignore[attr-defined]

    return program


def load_dataset(dataset_config) -> Dict[str, Iterable[Dict[str, object]]]:  # pragma: no cover - exercised
    # Provide minimal dataset for tests and examples.
    return {
        "train": [{"text": "train example", "label": 1}],
        "dev": [
            {"text": "example1", "label": 1},
            {"text": "example2", "label": 0},
        ],
    }


def compute_accuracy(records: List[Dict[str, object]], examples: List[Dict[str, object]]) -> float:
    correct = 0
    for record, example in zip(records, examples):
        if record["output"].get("prediction") == example.get("label"):
            correct += 1
    return correct / len(examples)


