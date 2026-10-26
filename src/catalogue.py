"""Curated laptop catalogue for the Find My Laptop feature.

The data lives in catalogue/laptops.json. Every entry was checked on the manufacturer's official Indian store;
`load_catalogue()` refuses entries that lack a source, a verification date or an official link, so unverified
data cannot reach the UI. This module is independent of the historical price model in src/predict.py.
"""
import json
from dataclasses import dataclass, fields
from datetime import date
from functools import lru_cache
from pathlib import Path

CATALOGUE_PATH = Path(__file__).resolve().parents[1] / 'catalogue' / 'laptops.json'

# Only links to these official stores are accepted.
OFFICIAL_URL_PREFIXES = ('https://www.hp.com/in-en/', 'https://www.apple.com/in/')
OS_VALUES = ('Windows', 'macOS')
POWER_CLASSES = ('efficient', 'performance', 'gaming')


class CatalogueError(ValueError):
    """Raised when a catalogue entry is missing required provenance or has invalid values."""


@dataclass(frozen=True)
class Laptop:
    id: str
    brand: str
    model: str
    part_number: str
    os: str
    cpu: str
    cpu_tier: int
    gpu: str
    gpu_dedicated: bool
    gpu_vram_gb: int | None
    gpu_tier: int
    cuda: bool
    ram_gb: int
    ram_upgradeable: bool | None
    storage_gb: int
    display_in: float
    resolution_w: int
    resolution_h: int
    refresh_hz: int | None
    weight_kg: float
    battery_wh: float
    power_class: str
    price_inr: float
    price_source: str
    url: str
    verified_on: str

    @property
    def display_name(self):
        return self.model if self.model.startswith(self.brand) else f'{self.brand} {self.model}'

    @property
    def storage_label(self):
        return f'{self.storage_gb // 1024} TB SSD' if self.storage_gb >= 1024 else f'{self.storage_gb} GB SSD'


def _validate(entry):
    names = {f.name for f in fields(Laptop)}
    missing = sorted(names - set(entry))
    if missing:
        raise CatalogueError(f"{entry.get('id', '?')}: missing fields {missing}")
    lid = entry['id']
    if not str(entry['url']).startswith(OFFICIAL_URL_PREFIXES):
        raise CatalogueError(f'{lid}: url must point to an official store {OFFICIAL_URL_PREFIXES}')
    if not entry['price_source'] or not entry['part_number']:
        raise CatalogueError(f'{lid}: price_source and part_number are required')
    try:
        verified = date.fromisoformat(entry['verified_on'])
    except (TypeError, ValueError):
        raise CatalogueError(f'{lid}: verified_on must be an ISO date') from None
    if verified > date.today():
        raise CatalogueError(f'{lid}: verified_on is in the future')
    if entry['os'] not in OS_VALUES or entry['power_class'] not in POWER_CLASSES:
        raise CatalogueError(f'{lid}: invalid os or power_class')
    for key in ('cpu_tier', 'gpu_tier'):
        if entry[key] not in (1, 2, 3, 4, 5):
            raise CatalogueError(f'{lid}: {key} must be 1-5')
    for key in ('ram_gb', 'storage_gb', 'display_in', 'weight_kg', 'battery_wh', 'price_inr'):
        if not isinstance(entry[key], (int, float)) or entry[key] <= 0:
            raise CatalogueError(f'{lid}: {key} must be a positive number')
    if entry['cuda'] and not entry['gpu_dedicated']:
        raise CatalogueError(f'{lid}: cuda requires a dedicated NVIDIA GPU')
    return Laptop(**{k: entry[k] for k in names})


@lru_cache(maxsize=2)
def load_catalogue(path=None):
    """Return (metadata dict, tuple of Laptop). Raises CatalogueError for any invalid entry."""
    data = json.loads(Path(path or CATALOGUE_PATH).read_text())
    laptops = tuple(_validate(e) for e in data['laptops'])
    ids = [l.id for l in laptops]
    if len(ids) != len(set(ids)):
        raise CatalogueError('duplicate laptop ids')
    parts = [l.part_number for l in laptops]
    if len(parts) != len(set(parts)):
        raise CatalogueError('duplicate part numbers')
    meta = {k: v for k, v in data.items() if k != 'laptops'}
    return meta, laptops
