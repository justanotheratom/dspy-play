"""Pricing acquisition helpers for DSPyLab."""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Optional

from library.config.models import (
    LegacyPricing,
    ModelConfig,
    PricingOverride,
    PricingReference,
    PriceRate as ConfigPriceRate,
)

from .catalog import (
    DEFAULT_CATALOG_PATH,
    PricingCatalog,
    PricingCatalogError,
    PricingEntry,
    PricingLookupError,
    PriceRate,
    ensure_catalog_exists,
)


DSPYLAB_PRICING_ENV_INPUT = "DSPYLAB_PRICING_INPUT_PER_1M"
DSPYLAB_PRICING_ENV_CACHED_INPUT = "DSPYLAB_PRICING_CACHED_INPUT_PER_1M"
DSPYLAB_PRICING_ENV_OUTPUT = "DSPYLAB_PRICING_OUTPUT_PER_1M"
DSPYLAB_PRICING_ENV_TRAINING = "DSPYLAB_PRICING_TRAINING_PER_1M"
DSPYLAB_PRICING_NONINTERACTIVE = "DSPYLAB_PRICING_NONINTERACTIVE"
DSPYLAB_PRICING_CATALOG_PATH = "DSPYLAB_PRICING_CATALOG_PATH"


class PricingAcquisitionError(PricingCatalogError):
    """Raised when pricing could not be obtained interactively."""


@dataclass
class _PricingSpec:
    pricing_id: Optional[str]
    price_version: Optional[int]
    fine_tuned: bool
    override: Optional[PricingOverride]
    legacy: Optional[LegacyPricing]


def ensure_pricing_for_models(
    *,
    catalog: PricingCatalog | None = None,
    models: Iterable[ModelConfig],
    non_interactive: bool = False,
) -> List[PricingEntry]:
    catalog = catalog or _create_catalog_from_env()
    entries: List[PricingEntry] = []
    for model in models:
        spec = _extract_pricing_spec(model)
        try:
            entry = catalog.lookup(
                provider=model.provider,
                model=model.model,
                pricing_id=spec.pricing_id,
                price_version=spec.price_version,
                fine_tuned=spec.fine_tuned,
            )
        except PricingLookupError as exc:
            if spec.legacy is not None:
                entry = _entry_from_legacy(model, spec, catalog)
            else:
                if non_interactive or os.environ.get(DSPYLAB_PRICING_NONINTERACTIVE):
                    raise PricingAcquisitionError(str(exc)) from exc
                entry = _prompt_for_pricing(model, spec, catalog)
                catalog.add_entry(entry)
        if spec.override:
            entry = _apply_override(entry, spec.override)
        entries.append(entry)
    return entries


def _prompt_for_pricing(model: ModelConfig, spec: _PricingSpec, catalog: PricingCatalog) -> PricingEntry:
    provider = model.provider
    model_name = model.model
    print(f"Pricing data required for provider={provider} model={model_name}")

    def _get_env_or_input(env_var: str, prompt: str) -> float:
        if env_var in os.environ:
            return float(os.environ[env_var])
        while True:
            raw = input(prompt).strip()
            try:
                return float(raw)
            except ValueError:
                print("Please enter a valid numeric USD per 1M tokens value.")

    input_rate = _override_or_prompt(
        override_rate=spec.override.input if spec.override else None,
        env_var=DSPYLAB_PRICING_ENV_INPUT,
        prompt="Input USD per 1M tokens: ",
    )
    cached_input_rate = _maybe_get_cached_input(spec.override)
    if not cached_input_rate:
        cached_env = os.environ.get(DSPYLAB_PRICING_ENV_CACHED_INPUT)
        if cached_env or (cached_env is None and input("Cached input rate available? (y/N) ").lower().startswith("y")):
            cached_input_rate = PriceRate(
                usd_per_1m=_get_env_or_input(
                    DSPYLAB_PRICING_ENV_CACHED_INPUT,
                    "Cached input USD per 1M tokens: ",
                )
            )

    output_rate = _override_or_prompt(
        override_rate=spec.override.output if spec.override else None,
        env_var=DSPYLAB_PRICING_ENV_OUTPUT,
        prompt="Output USD per 1M tokens: ",
    )

    training_rate = None
    training_env = os.environ.get(DSPYLAB_PRICING_ENV_TRAINING)
    if training_env or (training_env is None and input("Training cost per 1M tokens? (y/N) ").lower().startswith("y")):
        training_rate = PriceRate(
            usd_per_1m=_get_env_or_input(
                DSPYLAB_PRICING_ENV_TRAINING,
                "Training USD per 1M tokens: ",
            )
        )

    tier = input("Tier (press enter for default): ").strip() or None
    notes = input("Notes (optional): ").strip() or None
    pricing_id = spec.pricing_id or _default_pricing_id(model)
    version = input("Price version (press enter for next version): ").strip()
    price_version = int(version) if version else _next_version(catalog, provider, model_name, tier)

    return PricingEntry(
        provider=provider,
        model=model_name,
        tier=tier,
        fine_tuned=spec.fine_tuned,
        price_version=price_version,
        effective_at=datetime.now(timezone.utc),
        expires_at=None,
        input=input_rate,
        cached_input=cached_input_rate,
        output=output_rate,
        training=training_rate,
        notes=notes,
        pricing_id=pricing_id,
        was_estimated=False,
    )


def _next_version(catalog: PricingCatalog, provider: str, model_name: str, tier: str | None) -> int:
    versions = [
        entry.price_version
        for entry in catalog.entries
        if entry.provider.lower() == provider.lower()
        and entry.model.lower() == model_name.lower()
        and (tier is None or entry.tier == tier)
    ]
    return max(versions, default=0) + 1


def _extract_pricing_spec(model: ModelConfig) -> _PricingSpec:
    pricing = model.pricing
    reference: Optional[PricingReference] = None
    legacy: Optional[LegacyPricing] = None
    if pricing is not None:
        reference = pricing.reference
        legacy = pricing.legacy

    pricing_id = reference.pricing_id if reference else None
    price_version = reference.price_version if reference else None
    fine_tuned = reference.fine_tuned if reference else False
    override = reference.override if reference else None

    return _PricingSpec(
        pricing_id=pricing_id,
        price_version=price_version,
        fine_tuned=fine_tuned,
        override=override,
        legacy=legacy,
    )


def _entry_from_legacy(model: ModelConfig, spec: _PricingSpec, catalog: PricingCatalog) -> PricingEntry:
    legacy = spec.legacy
    if legacy is None:
        raise PricingAcquisitionError("legacy pricing data missing")

    input_rate = _per_1k_to_rate(legacy.input_per_1k, required=True)
    output_rate = _per_1k_to_rate(legacy.output_per_1k, required=True)
    cached_rate = _per_1k_to_rate(legacy.cached_input_per_1k)

    price_version = spec.price_version or _next_version(catalog, model.provider, model.model, None)
    pricing_id = spec.pricing_id or _default_pricing_id(model)

    entry = PricingEntry(
        provider=model.provider,
        model=model.model,
        tier=None,
        fine_tuned=spec.fine_tuned,
        price_version=price_version,
        effective_at=datetime.now(timezone.utc),
        expires_at=None,
        input=input_rate,
        cached_input=cached_rate,
        output=output_rate,
        training=None,
        notes="legacy-config",
        pricing_id=pricing_id,
        was_estimated=False,
    )

    catalog.add_entry(entry)
    return entry


def _per_1k_to_rate(value: Optional[float], *, required: bool = False) -> Optional[PriceRate]:
    if value is None:
        if required:
            raise PricingAcquisitionError("legacy pricing requires both input_per_1k and output_per_1k")
        return None
    return PriceRate(usd_per_1m=float(value) * 1000)


def _default_pricing_id(model: ModelConfig) -> str:
    return f"{model.provider}/{model.model}"


def _maybe_get_cached_input(override: Optional[PricingOverride]) -> Optional[PriceRate]:
    converted = _convert_config_rate(override.cached_input) if override else None
    if converted:
        return converted
    if DSPYLAB_PRICING_ENV_CACHED_INPUT in os.environ:
        return PriceRate(usd_per_1m=float(os.environ[DSPYLAB_PRICING_ENV_CACHED_INPUT]))
    return None


def _apply_override(entry: PricingEntry, override: PricingOverride) -> PricingEntry:
    return replace(
        entry,
        input=_convert_config_rate(override.input) or entry.input,
        cached_input=_convert_config_rate(override.cached_input) or entry.cached_input,
        output=_convert_config_rate(override.output) or entry.output,
    )


def _convert_config_rate(rate: Optional[ConfigPriceRate]) -> Optional[PriceRate]:
    if rate is None:
        return None
    return PriceRate(usd_per_1m=float(rate.usd_per_1m))


def _override_or_prompt(
    *,
    override_rate: Optional[ConfigPriceRate],
    env_var: str,
    prompt: str,
) -> PriceRate:
    converted = _convert_config_rate(override_rate)
    if converted:
        return converted
    return PriceRate(usd_per_1m=_get_env_or_input(env_var, prompt))


def _create_catalog_from_env() -> PricingCatalog:
    env_path = os.environ.get(DSPYLAB_PRICING_CATALOG_PATH)
    catalog_path = Path(env_path).expanduser() if env_path else DEFAULT_CATALOG_PATH
    ensure_catalog_exists(catalog_path)
    lock_path = catalog_path.with_suffix(".lock") if not catalog_path.name.endswith(".lock") else catalog_path
    if lock_path == catalog_path:
        lock_path = catalog_path.parent / (catalog_path.name + ".lock")
    return PricingCatalog(path=catalog_path, lock_path=lock_path)


