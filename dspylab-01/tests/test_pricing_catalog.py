from datetime import datetime, timezone
from pathlib import Path

import pytest

from library.pricing.catalog import (
    PricingCatalog,
    PricingEntry,
    PricingLookupError,
    PriceRate,
)


def _make_entry(price_version: int, effective: datetime) -> PricingEntry:
    return PricingEntry(
        provider="openai",
        model="gpt-4o",
        tier=None,
        fine_tuned=False,
        price_version=price_version,
        effective_at=effective,
        expires_at=None,
        input=PriceRate(usd_per_1m=3.0),
        cached_input=None,
        output=PriceRate(usd_per_1m=6.0),
        training=None,
        notes=None,
        pricing_id="openai/gpt-4o",
        was_estimated=False,
    )


def test_lookup_latest_version(tmp_path: Path) -> None:
    catalog_path = tmp_path / "catalog.json"
    catalog = PricingCatalog(path=catalog_path, lock_path=tmp_path / "catalog.lock")

    entry_v1 = _make_entry(1, datetime(2024, 1, 1, tzinfo=timezone.utc))
    entry_v2 = _make_entry(2, datetime(2024, 6, 1, tzinfo=timezone.utc))

    catalog.add_entry(entry_v1)
    catalog.add_entry(entry_v2)

    result = catalog.lookup(provider="openai", model="gpt-4o")

    assert result.price_version == 2
    assert result.input.usd_per_1m == pytest.approx(3.0)


def test_lookup_specific_version(tmp_path: Path) -> None:
    catalog_path = tmp_path / "catalog.json"
    catalog = PricingCatalog(path=catalog_path, lock_path=tmp_path / "catalog.lock")

    entry_v1 = _make_entry(1, datetime(2023, 12, 1, tzinfo=timezone.utc))
    entry_v2 = _make_entry(2, datetime(2024, 7, 1, tzinfo=timezone.utc))

    catalog.add_entry(entry_v1)
    catalog.add_entry(entry_v2)

    result = catalog.lookup(provider="openai", model="gpt-4o", price_version=1)

    assert result.price_version == 1


def test_lookup_missing_entry(tmp_path: Path) -> None:
    catalog_path = tmp_path / "catalog.json"
    catalog = PricingCatalog(path=catalog_path, lock_path=tmp_path / "catalog.lock")

    with pytest.raises(PricingLookupError):
        catalog.lookup(provider="anthropic", model="claude-xyz")


