"""Pricing catalog storage and lookup utilities."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from contextlib import contextmanager
import os


DEFAULT_CATALOG_PATH = Path(__file__).resolve().with_name("catalog.json")
CATALOG_LOCK_PATH = Path(__file__).resolve().with_name("catalog.lock")


class PricingCatalogError(Exception):
    """Base class for catalog errors."""


class PricingLookupError(PricingCatalogError):
    """Raised when a pricing entry cannot be found."""


@dataclass
class PriceRate:
    usd_per_1m: float

    def to_json(self) -> Dict[str, float]:
        return {"usd_per_1m": self.usd_per_1m}

    @classmethod
    def from_json(cls, data: Dict[str, float]) -> "PriceRate":
        return cls(usd_per_1m=float(data["usd_per_1m"]))


@dataclass
class PricingEntry:
    provider: str
    model: str
    tier: Optional[str]
    fine_tuned: bool
    price_version: int
    effective_at: datetime
    expires_at: Optional[datetime]
    input: PriceRate
    cached_input: Optional[PriceRate]
    output: PriceRate
    training: Optional[PriceRate] = None
    notes: Optional[str] = None
    pricing_id: Optional[str] = None
    was_estimated: bool = False

    def matches(self, *, timestamp: datetime, tier: Optional[str], fine_tuned: bool) -> bool:
        if tier and self.tier and tier.lower() != self.tier.lower():
            return False
        if fine_tuned and not self.fine_tuned and self.tier:
            # Base pricing entry with provider premium may still work; fine-tune entries override.
            pass
        if timestamp < self.effective_at:
            return False
        if self.expires_at and timestamp > self.expires_at:
            return False
        return True

    def to_json(self) -> Dict[str, object]:
        return {
            "provider": self.provider,
            "model": self.model,
            "tier": self.tier,
            "fine_tuned": self.fine_tuned,
            "price_version": self.price_version,
            "effective_at": self.effective_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "input": self.input.to_json(),
            "cached_input": self.cached_input.to_json() if self.cached_input else None,
            "output": self.output.to_json(),
            "training": self.training.to_json() if self.training else None,
            "notes": self.notes,
            "pricing_id": self.pricing_id,
            "was_estimated": self.was_estimated,
        }

    @classmethod
    def from_json(cls, data: Dict[str, object]) -> "PricingEntry":
        return cls(
            provider=str(data["provider"]),
            model=str(data["model"]),
            tier=str(data["tier"]) if data.get("tier") else None,
            fine_tuned=bool(data.get("fine_tuned", False)),
            price_version=int(data["price_version"]),
            effective_at=_parse_datetime(data["effective_at"]),
            expires_at=_parse_datetime(data.get("expires_at")) if data.get("expires_at") else None,
            input=PriceRate.from_json(data["input"]),
            cached_input=PriceRate.from_json(data["cached_input"]) if data.get("cached_input") else None,
            output=PriceRate.from_json(data["output"]),
            training=PriceRate.from_json(data["training"]) if data.get("training") else None,
            notes=data.get("notes"),
            pricing_id=data.get("pricing_id"),
            was_estimated=bool(data.get("was_estimated", False)),
        )


def _parse_datetime(value: str | None) -> Optional[datetime]:
    if not value:
        return None
    return datetime.fromisoformat(value).astimezone(timezone.utc)


class PricingCatalog:
    def __init__(self, path: Path = DEFAULT_CATALOG_PATH, lock_path: Path = CATALOG_LOCK_PATH) -> None:
        self._path = path
        self._lock_path = lock_path
        self._entries: List[PricingEntry] = []
        if self._path.exists():
            self._entries = self._load()

    @property
    def entries(self) -> List[PricingEntry]:
        return list(self._entries)

    def add_entry(self, entry: PricingEntry) -> None:
        self._entries.append(entry)
        self._entries.sort(key=lambda e: (e.provider, e.model, e.tier or "", e.price_version))
        self._save()

    def lookup(
        self,
        *,
        provider: str,
        model: str,
        timestamp: Optional[datetime] = None,
        tier: Optional[str] = None,
        fine_tuned: bool = False,
        pricing_id: Optional[str] = None,
        price_version: Optional[int] = None,
    ) -> PricingEntry:
        timestamp = timestamp or datetime.now(timezone.utc)
        candidates: List[PricingEntry] = []
        for entry in self._entries:
            if pricing_id and entry.pricing_id != pricing_id:
                continue
            if entry.provider.lower() != provider.lower():
                continue
            if entry.model.lower() != model.lower():
                continue
            if price_version and entry.price_version != price_version:
                continue
            if entry.matches(timestamp=timestamp, tier=tier, fine_tuned=fine_tuned):
                candidates.append(entry)

        if not candidates:
            raise PricingLookupError(
                f"No pricing entry for provider={provider} model={model} tier={tier or 'default'}"
            )

        candidates.sort(key=lambda e: (e.price_version, e.effective_at), reverse=True)
        return candidates[0]

    def _load(self) -> List[PricingEntry]:
        raw = json.loads(self._path.read_text(encoding="utf-8"))
        return [PricingEntry.from_json(item) for item in raw]

    def _save(self) -> None:
        with _locked_file(self._lock_path):
            self._path.parent.mkdir(parents=True, exist_ok=True)
            serialized = [entry.to_json() for entry in self._entries]
            self._path.write_text(json.dumps(serialized, indent=2), encoding="utf-8")


def ensure_catalog_exists(path: Path = DEFAULT_CATALOG_PATH) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("[]", encoding="utf-8")


@contextmanager
def _locked_file(lock_path: Path, retries: int = 10, delay: float = 0.1):
    lock_file = lock_path
    for attempt in range(retries):
        try:
            fd = os.open(lock_file, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            break
        except FileExistsError:
            if attempt == retries - 1:
                raise
            time.sleep(delay)
    try:
        yield
    finally:
        try:
            os.unlink(lock_file)
        except FileNotFoundError:
            pass

