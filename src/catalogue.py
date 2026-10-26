"""Curated laptop catalogue for the Find My Laptop feature.

The data lives in catalogue/laptops.json. Every entry was checked on the manufacturer's official Indian store or
on an authorised Indian retailer; `load_catalogue()` refuses entries that lack a source, a verification date or an
allowed link, so unverified data cannot reach the UI. Weight, battery and price may be unknown (null) and are then
treated as unknown, never guessed. This module is independent of the historical price model in src/predict.py.
"""
import json
from dataclasses import dataclass, fields
from datetime import date
from functools import lru_cache
from pathlib import Path

CATALOGUE_PATH = Path(__file__).resolve().parents[1] / 'catalogue' / 'laptops.json'

# Only links to these official brand stores and authorised retailers are accepted.
MANUFACTURER_URL_PREFIXES = ('https://www.hp.com/in-en/', 'https://www.apple.com/in/')
RETAILER_URL_PREFIXES = ('https://www.vijaysales.com/p/',)
OFFICIAL_URL_PREFIXES = MANUFACTURER_URL_PREFIXES + RETAILER_URL_PREFIXES
OS_VALUES = ('Windows', 'macOS')
POWER_CLASSES = ('efficient', 'performance', 'gaming')
AVAILABILITY = ('in_stock', 'out_of_stock', 'unknown')
SOURCE_TYPES = ('manufacturer', 'authorised_retailer')
OPTIONAL_NUMBERS = ('weight_kg', 'battery_wh', 'price_inr')     # may be null = unknown


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
    weight_kg: float | None
    battery_wh: float | None
    power_class: str
    price_inr: float | None
    price_source: str
    url: str
    verified_on: str
    availability: str = 'unknown'
    source_type: str = 'manufacturer'

    @property
    def display_name(self):
        return self.model if self.model.lower().startswith(self.brand.lower()) else f'{self.brand} {self.model}'

    @property
    def storage_label(self):
        return f'{self.storage_gb // 1024} TB SSD' if self.storage_gb >= 1024 else f'{self.storage_gb} GB SSD'


def normalise_part(part):
    """'E5SM2PA#ACJ' -> 'E5SM2PAACJ'; used for duplicate detection across sources."""
    return ''.join(ch for ch in str(part).upper() if ch.isalnum())


def _validate(entry):
    names = {f.name for f in fields(Laptop)}
    entry = {'availability': 'unknown', 'source_type': 'manufacturer', **entry}
    missing = sorted(names - set(entry))
    if missing:
        raise CatalogueError(f"{entry.get('id', '?')}: missing fields {missing}")
    lid = entry['id']
    prefixes = MANUFACTURER_URL_PREFIXES if entry['source_type'] == 'manufacturer' else RETAILER_URL_PREFIXES
    if entry['source_type'] not in SOURCE_TYPES or not str(entry['url']).startswith(prefixes):
        raise CatalogueError(f'{lid}: url must point to an allowed source {prefixes}')
    if entry['availability'] not in AVAILABILITY:
        raise CatalogueError(f'{lid}: availability must be one of {AVAILABILITY}')
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
        if key in OPTIONAL_NUMBERS and entry[key] is None:
            continue
        if isinstance(entry[key], bool) or not isinstance(entry[key], (int, float)) or entry[key] <= 0:
            raise CatalogueError(f'{lid}: {key} must be a positive number' + (' or null' if key in OPTIONAL_NUMBERS else ''))
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
    parts = [normalise_part(l.part_number) for l in laptops]
    dupes = sorted({p for p in parts if parts.count(p) > 1})
    if dupes:
        raise CatalogueError(f'duplicate part numbers: {dupes}')
    urls = [l.url.lower().rstrip('/') for l in laptops]
    if len(urls) != len(set(urls)):
        raise CatalogueError('duplicate product urls')
    meta = {k: v for k, v in data.items() if k != 'laptops'}
    return meta, laptops
