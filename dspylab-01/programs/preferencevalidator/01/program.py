"""DSPy program for the preference validator experiment."""

from __future__ import annotations

from collections import Counter
import math
import re
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


def build_program(
    chain_module: Optional[dspy.Module] = None,
    **extras,
) -> PreferenceValidatorModule:
    """Return the DSPy chain-of-thought classifier with optional strategy."""

    module = PreferenceValidatorModule(chain_module)
    module._dspylab_io = {
        "input_field": extras.get("input_field", "preference_request"),
        "output_field": extras.get("output_field", "normalized_action"),
    }
    return module

_TOKEN_PATTERN = re.compile(r"[\w']+")
COSINE_MATCH_THRESHOLD = 0.9


def _stringify_output(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def _tokenize(text: str) -> Counter[str]:
    tokens = _TOKEN_PATTERN.findall(text.lower())
    return Counter(tokens)


def _cosine_similarity(text_a: str, text_b: str) -> float:
    if not text_a or not text_b:
        return 0.0

    vec_a = _tokenize(text_a)
    vec_b = _tokenize(text_b)

    intersection = set(vec_a) & set(vec_b)
    dot_product = sum(vec_a[token] * vec_b[token] for token in intersection)

    magnitude_a = math.sqrt(sum(count * count for count in vec_a.values()))
    magnitude_b = math.sqrt(sum(count * count for count in vec_b.values()))

    if not magnitude_a or not magnitude_b:
        return 0.0

    return dot_product / (magnitude_a * magnitude_b)


def compute_accuracy(records: List[Dict[str, object]], examples: List[Dict[str, object]]) -> float:
    """Average cosine similarity between predicted and expected outputs."""

    if not records:
        return 0.0

    total_similarity = 0.0
    total = min(len(records), len(examples))
    for record, example in zip(records, examples):
        predicted = record.get("output")
        if isinstance(predicted, dict):
            predicted = predicted.get("output")
        expected = example.get("output")

        predicted_text = _stringify_output(predicted)
        expected_text = _stringify_output(expected)

        total_similarity += _cosine_similarity(predicted_text, expected_text)

    return total_similarity / total if total else 0.0


def per_example_match(prediction: Dict[str, object], example: Dict[str, object]) -> bool:
    """Return True when the cosine similarity exceeds the configured threshold."""

    predicted = prediction
    if isinstance(predicted, dict):
        predicted = predicted.get("output")
    expected = example.get("output")

    predicted_text = _stringify_output(predicted)
    expected_text = _stringify_output(expected)

    return _cosine_similarity(predicted_text, expected_text) >= COSINE_MATCH_THRESHOLD


