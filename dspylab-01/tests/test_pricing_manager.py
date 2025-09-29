from datetime import datetime, timezone

import pytest

from library.config.models import ModelConfig, Pricing, PricingReference, LegacyPricing
from library.pricing.catalog import PricingCatalog, PricingEntry, PriceRate
from library.pricing.manager import ensure_pricing_for_models, PricingAcquisitionError


def _model_with_pricing(pricing_id: str | None) -> ModelConfig:
    pricing = None
    if pricing_id:
        pricing = Pricing(reference=PricingReference(pricing_id=pricing_id))
    return ModelConfig(
        id="model-a",
        provider="openai",
        model="gpt-4o",
        api_key_env="OPENAI_API_KEY",
        pricing=pricing,
    )


def test_ensure_pricing_noninteractive_error(tmp_path):
    catalog = PricingCatalog(path=tmp_path / "catalog.json", lock_path=tmp_path / "catalog.lock")
    model = _model_with_pricing("openai/gpt-4o")

    with pytest.raises(PricingAcquisitionError):
        ensure_pricing_for_models(catalog=catalog, models=[model], non_interactive=True)


def test_ensure_pricing_returns_existing_entry(tmp_path):
    catalog = PricingCatalog(path=tmp_path / "catalog.json", lock_path=tmp_path / "catalog.lock")
    entry = PricingEntry(
        provider="openai",
        model="gpt-4o",
        tier=None,
        fine_tuned=False,
        price_version=1,
        effective_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        expires_at=None,
        input=PriceRate(usd_per_1m=3.0),
        cached_input=None,
        output=PriceRate(usd_per_1m=6.0),
        training=None,
        notes=None,
        pricing_id="openai/gpt-4o",
        was_estimated=False,
    )
    catalog.add_entry(entry)

    result = ensure_pricing_for_models(catalog=catalog, models=[_model_with_pricing("openai/gpt-4o")], non_interactive=True)

    assert result[0].pricing_id == "openai/gpt-4o"


def test_ensure_pricing_legacy_conversion(tmp_path, monkeypatch):
    catalog = PricingCatalog(path=tmp_path / "catalog.json", lock_path=tmp_path / "catalog.lock")
    model = ModelConfig(
        id="model-legacy",
        provider="openai",
        model="gpt-3.5",
        api_key_env="OPENAI_API_KEY",
        pricing=Pricing(legacy=LegacyPricing(input_per_1k=0.01, output_per_1k=0.02, cached_input_per_1k=0.005)),
    )

    entries = ensure_pricing_for_models(catalog=catalog, models=[model], non_interactive=True)

    entry = entries[0]
    assert entry.input.usd_per_1m == pytest.approx(10.0)
    assert entry.output.usd_per_1m == pytest.approx(20.0)
    assert entry.cached_input.usd_per_1m == pytest.approx(5.0)


