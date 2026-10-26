"""Tests for the verified laptop catalogue (catalogue/laptops.json + src/catalogue.py)."""
import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.catalogue import CATALOGUE_PATH, OFFICIAL_URL_PREFIXES, CatalogueError, load_catalogue, normalise_part


class TestCatalogue(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.meta, cls.laptops = load_catalogue()
        cls.raw = json.loads(CATALOGUE_PATH.read_text())

    def test_size_and_unique_ids(self):
        self.assertGreaterEqual(len(self.laptops), 50)
        self.assertEqual(len({l.id for l in self.laptops}), len(self.laptops))
        self.assertEqual(len({l.part_number for l in self.laptops}), len(self.laptops))

    def test_every_entry_has_provenance(self):
        for l in self.laptops:
            with self.subTest(l.id):
                self.assertTrue(l.url.startswith(OFFICIAL_URL_PREFIXES))
                self.assertTrue(l.price_source)
                self.assertRegex(l.verified_on, r'^\d{4}-\d{2}-\d{2}$')
                self.assertGreater(l.price_inr, 10_000)

    def test_specs_are_plausible(self):
        for l in self.laptops:
            with self.subTest(l.id):
                self.assertIn(l.ram_gb, (8, 12, 16, 24, 32, 48, 64))
                self.assertTrue(l.weight_kg is None or 0.8 <= l.weight_kg <= 3.5)
                self.assertTrue(l.battery_wh is None or 30 <= l.battery_wh <= 100)
                self.assertTrue(l.price_inr is None or 15_000 <= l.price_inr <= 1_500_000)
                self.assertTrue(11 <= l.display_in <= 18.5)
                self.assertEqual(l.cuda, l.gpu.startswith('NVIDIA'))      # CUDA flag only for NVIDIA GPUs

    def test_links_point_to_the_exact_configuration(self):
        for l in self.laptops:
            with self.subTest(l.id):
                if l.source_type == 'authorised_retailer':   # Vijay Sales slugs end with the maker's part number
                    slug = ''.join(ch for ch in l.url.lower() if ch.isalnum())
                    self.assertIn(normalise_part(l.part_number).lower(), slug)
                elif l.brand == 'HP':    # HP product URLs end with the part number, e.g. ...-e5sm2pa-acj
                    self.assertTrue(l.url.endswith('-' + l.part_number.replace('#', '-').lower()))
                else:                    # Apple part-number product URLs
                    self.assertTrue(l.url.endswith('/shop/product/' + l.part_number))

    def test_budget_coverage(self):
        prices = [l.price_inr for l in self.laptops if l.price_inr is not None]
        self.assertGreaterEqual(sum(p < 50_000 for p in prices), 3)
        self.assertGreaterEqual(sum(50_000 <= p < 70_000 for p in prices), 10)
        self.assertGreaterEqual(sum(70_000 <= p < 100_000 for p in prices), 10)

    def test_brand_diversity(self):
        from collections import Counter
        counts = Counter(l.brand for l in self.laptops)
        self.assertGreaterEqual(len(counts), 5)
        for brand in ('HP', 'Lenovo', 'ASUS', 'Dell', 'Acer', 'Apple'):
            self.assertIn(brand, counts)

    def test_retailer_entries_were_in_stock(self):
        for l in self.laptops:
            if l.source_type == 'authorised_retailer':
                self.assertEqual(l.availability, 'in_stock')
                self.assertNotIn('display unit', l.model.lower())

    def test_coverage_of_key_segments(self):
        self.assertTrue(any(l.os == 'macOS' for l in self.laptops))
        self.assertTrue(any(l.gpu_dedicated for l in self.laptops))
        self.assertTrue(any(not l.gpu_dedicated and l.weight_kg and l.weight_kg < 1.5 for l in self.laptops))
        self.assertTrue(any(l.price_inr and l.price_inr < 100_000 for l in self.laptops))

    def _load_modified(self, change):
        data = copy.deepcopy(self.raw)
        change(data['laptops'][0])
        with tempfile.NamedTemporaryFile('w', suffix='.json', delete=False) as f:
            json.dump(data, f)
        load_catalogue.cache_clear()
        try:
            return load_catalogue(f.name)
        finally:
            load_catalogue.cache_clear()

    def test_rejects_unverified_entries(self):
        bad_changes = {
            'non-official url': lambda e: e.update(url='https://www.amazon.in/some-laptop'),
            'missing price source': lambda e: e.update(price_source=''),
            'missing verification date': lambda e: e.update(verified_on=None),
            'future verification date': lambda e: e.update(verified_on='2999-01-01'),
            'missing field': lambda e: e.pop('battery_wh'),
            'invalid tier': lambda e: e.update(cpu_tier=7),
            'negative price': lambda e: e.update(price_inr=-5),
            'cuda without dedicated gpu': lambda e: e.update(cuda=True, gpu_dedicated=False),
            'retailer url for manufacturer source': lambda e: e.update(url='https://www.vijaysales.com/p/1/x', source_type='manufacturer'),
            'unknown source type': lambda e: e.update(source_type='marketplace'),
            'invalid availability': lambda e: e.update(availability='maybe'),
        }
        for name, change in bad_changes.items():
            with self.subTest(name), self.assertRaises(CatalogueError):
                self._load_modified(change)


    def test_unknown_optional_values_allowed(self):
        _, laptops = self._load_modified(lambda e: e.update(weight_kg=None, battery_wh=None, price_inr=None))
        self.assertIsNone(laptops[0].weight_kg)

    def test_duplicate_part_numbers_rejected(self):
        def dup(data):
            twin = dict(data['laptops'][1]); twin['id'] = 'twin'; twin['url'] = data['laptops'][1]['url'] + '?x'
            twin['part_number'] = data['laptops'][1]['part_number'].lower().replace('#', ' ')   # same part, different formatting
            data['laptops'].append(twin)
        data = copy.deepcopy(self.raw); dup(data)
        with tempfile.NamedTemporaryFile('w', suffix='.json', delete=False) as f:
            json.dump(data, f)
        load_catalogue.cache_clear()
        try:
            with self.assertRaises(CatalogueError):
                load_catalogue(f.name)
        finally:
            load_catalogue.cache_clear()


if __name__ == '__main__':
    unittest.main()
