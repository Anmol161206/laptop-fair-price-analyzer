"""Website tests: UI helpers + headless runs of app.py with Streamlit's AppTest.

The model itself is covered by test_pipeline.py; these tests only check that the website passes the right
specs to predict_laptop_price(), displays its output faithfully and handles errors.

Run from the project root:
    conda run -n laptop_price python -m unittest discover -s tests -v
"""
import math
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.app_helpers import (build_specs, compare_listed_price, format_inr, format_resolution, model_summary,
                             parse_resolution, resolution_label, run_prediction)
from src.predict import REQUIRED_FIELDS, load_artifacts, predict_laptop_price

APP_PATH = str(PROJECT_ROOT / 'app.py')


class TestFormatInr(unittest.TestCase):
    def test_indian_grouping(self):
        self.assertEqual(format_inr(0), '₹0')
        self.assertEqual(format_inr(999), '₹999')
        self.assertEqual(format_inr(1000), '₹1,000')
        self.assertEqual(format_inr(48000), '₹48,000')
        self.assertEqual(format_inr(123456), '₹1,23,456')
        self.assertEqual(format_inr(12345678), '₹1,23,45,678')
        self.assertEqual(format_inr(47971.6), '₹47,972')
        self.assertEqual(format_inr(-12000), '-₹12,000')
        self.assertEqual(format_inr(1234.5, decimals=2), '₹1,234.50')

    def test_rejects_non_finite(self):
        for bad in (None, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                format_inr(bad)


class TestResolution(unittest.TestCase):
    def test_round_trip(self):
        self.assertEqual(format_resolution(1920, 1080), '1920 × 1080')
        self.assertEqual(parse_resolution('1920 × 1080'), (1920, 1080))
        self.assertEqual(parse_resolution(format_resolution(2736, 1824)), (2736, 1824))

    def test_labels(self):
        self.assertEqual(resolution_label('1920 × 1080'), '1920 × 1080  (Full HD)')
        self.assertEqual(resolution_label('2736 × 1824'), '2736 × 1824')

    def test_invalid(self):
        for bad in ('1920x1080', '1920 × ', 'abc × 1080', '1920 × 1080 × 3', '0 × 1080'):
            with self.assertRaises(ValueError):
                parse_resolution(bad)


class TestListedPriceComparison(unittest.TestCase):
    def test_above(self):
        c = compare_listed_price(80000, 92000)
        self.assertEqual(c['direction'], 'above')
        self.assertAlmostEqual(c['percent'], 15.0)
        self.assertEqual(c['sentence'], "The listed price is 15% above this model's estimate.")
        self.assertEqual(c['amount_text'], '₹12,000 above the model estimate')

    def test_below(self):
        c = compare_listed_price(80000, 73000)
        self.assertEqual(c['direction'], 'below')
        self.assertAlmostEqual(c['percent'], -8.75)
        self.assertEqual(c['sentence'], "The listed price is 9% below this model's estimate.")
        self.assertEqual(c['amount_text'], '₹7,000 below the model estimate')

    def test_about_same(self):
        self.assertEqual(compare_listed_price(80000, 80300)['direction'], 'about the same')

    def test_neutral_wording(self):
        for listed in (40000, 80000, 160000):
            text = ' '.join(compare_listed_price(80000, listed)[k] for k in ('sentence', 'amount_text')).lower()
            for word in ('deal', 'overpriced', 'underpriced', 'buy', 'cheap', 'expensive'):
                self.assertNotIn(word, text)

    def test_invalid(self):
        for bad in (0, -5000, None, float('nan'), True):
            with self.assertRaises(ValueError):
                compare_listed_price(80000, bad)


class TestSpecsAndSummary(unittest.TestCase):
    def test_build_specs_types(self):
        specs = build_specs(company='Dell', type_name='Notebook', ram_gb=8, weight_kg=2, os='Windows 10',
                            cpu='Intel Core i5 7200U 2.5GHz', gpu='Intel HD Graphics 620', ssd_gb=256, hdd_gb=0,
                            inches=15.6, resolution='1920 × 1080', touchscreen=False, ips=1)
        self.assertEqual(sorted(specs), sorted(REQUIRED_FIELDS))
        self.assertIs(type(specs['ips']), bool)
        self.assertIs(type(specs['weight_kg']), float)
        self.assertEqual((specs['resolution_width'], specs['resolution_height']), (1920, 1080))

    def test_run_prediction_reports_invalid_input(self):
        specs = build_specs(company='Dell', type_name='Notebook', ram_gb=8, weight_kg=2.0, os='Windows 10',
                            cpu='Intel Core i5 7200U 2.5GHz', gpu='Intel HD Graphics 620', ssd_gb=256, hdd_gb=0,
                            inches=15.6, resolution='1920 × 1080', touchscreen=False, ips=False)
        result, error = run_prediction({**specs, 'weight_kg': -1.0})
        self.assertIsNone(result)
        self.assertIn('weight_kg', error)
        result, error = run_prediction(specs)
        self.assertIsNone(error)
        self.assertGreater(result['predicted_price'], 0)

    def test_summary_from_metadata(self):
        _, meta = load_artifacts()
        s = model_summary(meta)
        self.assertEqual(s['model_label'], 'XGBoost Regressor')
        self.assertEqual(s['training_rows'], 1273)
        self.assertEqual(s['mae_rupees'], meta['evaluation_before_final_refit']['holdout']['mae_rupees'])


# ---------------------------------------------------------------------------------------------- headless app
SCENARIOS = {
    'Mac': dict(company='Apple', type_name='Ultrabook', ram_gb=8, weight_kg=1.37, os='macOS',
                cpu='Intel Core i5 2.3GHz', gpu='Intel Iris Plus Graphics 640', ssd_gb=128, hdd_gb=0,
                inches=13.3, resolution='2560 × 1600', touchscreen=False, ips=True),
    'Budget Windows notebook': dict(company='Acer', type_name='Notebook', ram_gb=4, weight_kg=2.1, os='Windows 10',
                                    cpu='Intel Celeron Dual Core N3060 1.6GHz', gpu='Intel HD Graphics 400',
                                    ssd_gb=0, hdd_gb=500, inches=15.6, resolution='1366 × 768',
                                    touchscreen=False, ips=False),
    'Gaming laptop': dict(company='MSI', type_name='Gaming', ram_gb=16, weight_kg=2.4, os='Windows 10',
                          cpu='Intel Core i7 7700HQ 2.8GHz', gpu='Nvidia GeForce GTX 1060', ssd_gb=256, hdd_gb=1000,
                          inches=15.6, resolution='1920 × 1080', touchscreen=False, ips=True),
    'No HDD': dict(company='Dell', type_name='Ultrabook', ram_gb=8, weight_kg=1.2, os='Windows 10',
                   cpu='Intel Core i5 8250U 1.6GHz', gpu='Intel UHD Graphics 620', ssd_gb=512, hdd_gb=0,
                   inches=13.3, resolution='1920 × 1080', touchscreen=False, ips=True),
    'Touchscreen 2-in-1': dict(company='HP', type_name='2 in 1 Convertible', ram_gb=8, weight_kg=1.28,
                               os='Windows 10', cpu='Intel Core i7 8550U 1.8GHz', gpu='Intel UHD Graphics 620',
                               ssd_gb=256, hdd_gb=0, inches=13.3, resolution='1920 × 1080',
                               touchscreen=True, ips=True),
}


def fill_form(at, values, listed_price=None):
    for key in ('company', 'type_name', 'ram_gb', 'os', 'cpu', 'gpu', 'ssd_gb', 'hdd_gb', 'resolution'):
        at.selectbox(key=key).set_value(values[key])
    at.number_input(key='weight_kg').set_value(values['weight_kg'])
    at.number_input(key='inches').set_value(values['inches'])
    at.toggle(key='touchscreen').set_value(values['touchscreen'])
    at.toggle(key='ips').set_value(values['ips'])
    at.number_input(key='listed_price').set_value(listed_price)
    at.button(key='submit').click()
    return at.run()


class TestAppHeadless(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from streamlit.testing.v1 import AppTest
        cls.AppTest = AppTest

    def new_app(self):
        at = self.AppTest.from_file(APP_PATH, default_timeout=60).run()
        self.assertEqual(len(at.exception), 0, [e.value for e in at.exception])
        return at

    def test_page_loads(self):
        at = self.new_app()
        self.assertEqual(at.title[0].value, 'Laptop Fair-Price Analyzer')
        self.assertEqual(len(at.error), 0)
        self.assertIn('Estimate Price', at.info[0].value)           # prompt before the first estimate
        sidebar = ' '.join(m.value for m in at.sidebar.markdown)
        self.assertIn('Applied AI & Data Science Project', sidebar)

    def test_scenarios_match_direct_prediction(self):
        at = self.new_app()
        misses_before = load_artifacts.cache_info().misses
        for name, values in SCENARIOS.items():
            with self.subTest(name):
                at = fill_form(at, values)
                self.assertEqual(len(at.exception), 0)
                self.assertEqual(len(at.error), 0)
                site = at.session_state['last']
                direct = predict_laptop_price(build_specs(**values))
                self.assertEqual(site['specs'], build_specs(**values))
                self.assertEqual(site['result']['predicted_price'], direct['predicted_price'])   # bit-identical
                self.assertEqual(site['result']['features'], direct['features'])
                shown = ' '.join(m.value for m in at.markdown)
                self.assertIn(format_inr(direct['predicted_price_rounded']), shown)
                self.assertEqual(len(at.warning), len(direct['warnings']))
        # the model/metadata were not reloaded during any of these reruns
        self.assertEqual(load_artifacts.cache_info().misses, misses_before)

    def test_derived_values_come_from_pipeline(self):
        at = fill_form(self.new_app(), SCENARIOS['Gaming laptop'])
        f = at.session_state['last']['result']['features']
        self.assertEqual((f['cpu_type'], f['cpu_gen'], f['gpu_tier']), ('High-performance (H)', 'Gen 7', 'Nvidia GTX'))

    def test_listed_price_comparison_display(self):
        values = SCENARIOS['Gaming laptop']
        at = fill_form(self.new_app(), values, listed_price=92000)
        estimate = predict_laptop_price(build_specs(**values))['predicted_price']
        expected = compare_listed_price(estimate, 92000)
        info = ' '.join(i.value for i in at.info)
        self.assertIn(expected['sentence'], info)
        self.assertIn(expected['amount_text'], info)
        self.assertEqual(at.metric[1].value, '₹92,000')

    def test_blank_listed_price_skips_comparison(self):
        at = fill_form(self.new_app(), SCENARIOS['Mac'])
        self.assertEqual(len(at.metric), 0)

    def test_invalid_listed_price_handled(self):
        at = fill_form(self.new_app(), SCENARIOS['Mac'], listed_price=0)
        self.assertEqual(len(at.exception), 0)
        self.assertTrue(any('listed price must be a positive amount' in e.value for e in at.error))


if __name__ == '__main__':
    unittest.main()
