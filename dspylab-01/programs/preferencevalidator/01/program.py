"""DSPy program for the preference validator experiment."""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import dspy


class PreferenceClassifier(dspy.Signature):
    """Classify a dietary preference request into the canonical action string."""

    preference_request: str = dspy.InputField(
        desc="User natural-language description of a dietary preference"
    )
    normalized_action: str = dspy.OutputField(
        desc="Call to report_success or report_failure with explicit justification"
    )


class PreferenceValidatorModule(dspy.Module):
    def __init__(self, chain_module: Optional[dspy.Module] = None) -> None:
        super().__init__()
        base = dspy.ChainOfThought(PreferenceClassifier)
        self.classifier = chain_module(base) if chain_module else base

    def forward(self, preference_request: str) -> Dict[str, str]:
        completion = self.classifier(preference_request=preference_request)
        return {"output": completion.normalized_action}


def build_program(chain_module: Optional[dspy.Module] = None) -> PreferenceValidatorModule:
    """Return the DSPy chain-of-thought classifier with optional strategy."""

    return PreferenceValidatorModule(chain_module)


def compute_accuracy(records: List[Dict[str, object]], examples: List[Dict[str, object]]) -> float:
    """Exact-match accuracy between predicted and expected outputs."""

    if not records:
        return 0.0

    correct = 0
    total = min(len(records), len(examples))
    for record, example in zip(records, examples):
        predicted = record.get("output")
        if isinstance(predicted, dict):
            predicted = predicted.get("output")
        expected = example.get("output")
        if predicted == expected:
            correct += 1

    return correct / total if total else 0.0


def per_example_match(prediction: Dict[str, object], example: Dict[str, object]) -> bool:
    """Return True when the model output exactly matches the expected action."""

    predicted = prediction
    if isinstance(predicted, dict):
        predicted = predicted.get("output")
    expected = example.get("output")
    return predicted == expected


