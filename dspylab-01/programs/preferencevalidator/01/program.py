"""Sample DSPy program for testing the execution engine."""

from __future__ import annotations

from typing import Dict, Iterable, List


def build_program():  # pragma: no cover - exercised via executor tests
    def program(example: Dict[str, object]) -> Dict[str, object]:
        # Simplistic model: echo the label as prediction if available.
        return {"prediction": example.get("label", 0)}

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


