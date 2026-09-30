"""Tests for the production feature engineering, saved model and prediction interface.

Run from the project root:
    conda run -n laptop_price python -m unittest discover -s tests -v
"""
import json
import math
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.features import (MODEL_FEATURES, PIPELINE_INPUT_COLUMNS, CompanyGrouper, build_features_from_raw,
                          engineer_features, specs_from_raw)
from src.predict import InvalidSpecError, load_artifacts, predict_laptop_price
from src.train import LOCKED_EVALUATION, MODEL_PATH, load_clean_raw, reproduce_locked_evaluation

SANITY_RANGE = (3_000, 1_000_000)   # rupees; deliberately wide (training prices: ~9.3k - 325k)


def user_specs_from_raw_row(raw_row):
    """A raw dataset row expressed as the human-readable dict a website would send (no cpu_ghz: parsed from cpu)."""
    spec = specs_from_raw(raw_row.to_frame().T).iloc[0].to_dict()
    spec.pop('cpu_ghz')
    return {k: (v.item() if hasattr(v, 'item') else v) for k, v in spec.items()}


def run_fresh_python(code, cwd):
    """Run code in a brand-new Python process (same interpreter/env) and return its stdout."""
    out = subprocess.run([sys.executable, '-c', code], cwd=cwd, capture_output=True, text=True, timeout=120)
    if out.returncode != 0:
        raise AssertionError(f'fresh process failed:\n{out.stderr}')
    return out.stdout


class Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not MODEL_PATH.exists():
            raise unittest.SkipTest('artifacts missing - run `python -m src.train` first')
        cls.raw = load_clean_raw()
        cls.pipeline, cls.metadata = load_artifacts()
        cls.example = user_specs_from_raw_row(cls.raw.iloc[0])


class TestA_KnownRow(Base):
    def test_known_row_predicts(self):
        result = predict_laptop_price(self.example)
        self.assertIn('predicted_price', result)
        self.assertEqual(result['warnings'], [])                      # a training row has nothing unseen
        self.assertEqual(list(result['features']), MODEL_FEATURES)

    def test_user_path_equals_training_path(self):
        """Specs dict -> predict path must produce the same features and prediction as the training path."""
        rows = self.raw.sample(25, random_state=0)
        X_train_path = build_features_from_raw(rows.drop(columns=['Price']))
        expected = np.exp(self.pipeline.predict(X_train_path))
        for (idx, row), exp_price in zip(rows.iterrows(), expected):
            result = predict_laptop_price(user_specs_from_raw_row(row))
            grouped = self.pipeline.named_steps['company_grouper'].transform(X_train_path.loc[[idx]])
            self.assertEqual(result['features'], {k: (v.item() if hasattr(v, 'item') else v)
                                                  for k, v in grouped[MODEL_FEATURES].iloc[0].items()})
            self.assertAlmostEqual(result['predicted_price'], exp_price, places=6)


class TestNoTargetLeakage(Base):
    def test_features_without_price(self):
        no_price = self.raw.drop(columns=['Price'])
        X = build_features_from_raw(no_price)
        self.assertNotIn('Price', X.columns)
        self.assertEqual(list(X.columns), PIPELINE_INPUT_COLUMNS)
        # Price present or absent must make no difference
        pd.testing.assert_frame_equal(X, build_features_from_raw(self.raw))
        # Changing Price must not change any feature
        shuffled = self.raw.assign(Price=self.raw['Price'].sample(frac=1, random_state=0).values)
        pd.testing.assert_frame_equal(X, build_features_from_raw(shuffled))

    def test_company_grouper_uses_counts_only(self):
        X = build_features_from_raw(self.raw.drop(columns=['Price']))
        a = CompanyGrouper().fit(X).rare_companies_
        b = CompanyGrouper().fit(X, np.random.default_rng(0).normal(size=len(X))).rare_companies_
        self.assertEqual(a, b)


class TestB_MatchesNotebook05(Base):
    def test_features_match_notebook05_code(self):
        """Execute the build_features() code stored in 05_final_evaluation.ipynb and compare on all rows."""
        nb = json.loads((PROJECT_ROOT / '05_final_evaluation.ipynb').read_text())
        src = next(''.join(c['source']) for c in nb['cells']
                   if c['cell_type'] == 'code' and 'def build_features(r):' in ''.join(c['source']))
        src = src[src.index('def cpu_brand'):src.index('d = build_features(raw_X)')]   # function definitions only
        ns = {'re': re, 'np': np, 'pd': pd}
        exec(src, ns)
        raw_X = self.raw.drop(columns=['Price'])
        nb05 = ns['build_features'](raw_X)
        prod = build_features_from_raw(raw_X)
        pd.testing.assert_frame_equal(prod, nb05[PIPELINE_INPUT_COLUMNS], check_dtype=False)

    def test_known_rows_spot_check(self):
        """Hand-checked values for a few rows of the raw data."""
        X = build_features_from_raw(self.raw.drop(columns=['Price']))
        r0 = X.iloc[0]      # Apple MacBook Pro 13.3" IPS 2560x1600, i5 2.3GHz, Iris Plus 640, 128GB SSD, macOS
        self.assertEqual((r0['Cpu brand'], r0['cpu_gen'], r0['gpu_tier'], r0['os']),
                         ('Intel Core i5', 'Unknown', 'Intel Iris', 'Mac'))
        self.assertEqual((r0['Ram'], r0['SSD'], r0['HDD'], r0['Ips'], r0['X_res'], r0['Y_res']), (8, 128, 0, 1, 2560, 1600))
        self.assertAlmostEqual(r0['ppi'], math.hypot(2560, 1600) / 13.3)
        self.assertAlmostEqual(r0['cpu_ghz'], 2.3)
        r = X[self.raw['Cpu'].eq('Intel Core i7 7700HQ 2.8GHz') & self.raw['Gpu'].eq('Nvidia GeForce GTX 1050')].iloc[0]
        self.assertEqual((r['cpu_type'], r['cpu_gen'], r['gpu_tier']), ('High-performance (H)', 'Gen 7', 'Nvidia GTX'))
        r = X[self.raw['Cpu'].str.startswith('AMD')].iloc[0]
        self.assertEqual((r['Cpu brand'], r['cpu_type'], r['cpu_gen']), ('AMD Processor', 'Other', 'Unknown'))
        r = X[self.raw['Memory'].eq('128GB SSD +  1TB HDD')].iloc[0]
        self.assertEqual((r['SSD'], r['HDD']), (128, 1000))

    def test_reproduces_locked_holdout_metrics(self):
        raw_X = self.raw.drop(columns=['Price'])
        check = reproduce_locked_evaluation(self.raw, build_features_from_raw(raw_X), np.log(self.raw['Price']))
        self.assertAlmostEqual(check['r2_log_price'], LOCKED_EVALUATION['holdout']['r2_log_price'], places=5)
        self.assertAlmostEqual(check['mae_rupees'], LOCKED_EVALUATION['holdout']['mae_rupees'], places=2)
        # notebook 05's rare-company list (fitted on its 1,015 training rows)
        self.assertEqual(set(check['rare_companies']), {'Microsoft', 'Razer', 'Mediacom', 'Xiaomi', 'Vero', 'Samsung',
                                                        'Google', 'LG', 'Huawei', 'Chuwi', 'Fujitsu'})


class TestC_ReloadInFreshProcess(Base):
    CODE = ('import json, sys; from src.predict import predict_laptop_price; '
            'print(json.dumps(predict_laptop_price(json.loads(sys.argv[1] if len(sys.argv) > 1 else {spec!r}))))')

    def _fresh_prediction(self, cwd):
        return json.loads(run_fresh_python(self.CODE.format(spec=json.dumps(self.example)), cwd))

    def test_same_prediction_in_new_process(self):
        here = predict_laptop_price(self.example)
        there = self._fresh_prediction(PROJECT_ROOT)
        self.assertEqual(there['predicted_price'], here['predicted_price'])     # bit-identical
        self.assertEqual(there['features'], json.loads(json.dumps(here['features'])))

    def test_artifact_independence(self):
        """Only src/ + artifacts/ copied to an empty folder: no notebooks, no CSV, no project files."""
        here = predict_laptop_price(self.example)
        with tempfile.TemporaryDirectory() as tmp:
            shutil.copytree(PROJECT_ROOT / 'src', Path(tmp) / 'src', ignore=shutil.ignore_patterns('__pycache__'))
            shutil.copytree(PROJECT_ROOT / 'artifacts', Path(tmp) / 'artifacts')
            self.assertEqual(sorted(p.name for p in Path(tmp).iterdir()), ['artifacts', 'src'])
            there = self._fresh_prediction(tmp)
        self.assertEqual(there['predicted_price'], here['predicted_price'])

    def test_metadata_consistent_with_model(self):
        grouper = self.pipeline.named_steps['company_grouper']
        self.assertEqual(self.metadata['company_grouping']['rare_companies'], grouper.rare_companies_)
        self.assertEqual(self.metadata['company_grouping']['frequent_companies'], grouper.frequent_companies_)
        self.assertEqual(self.metadata['training_data']['n_training_observations'], 1273)
        self.assertEqual(self.metadata['final_feature_names'], MODEL_FEATURES)
        self.assertIn('BEFORE the final refit', self.metadata['evaluation_before_final_refit']['note'])


class TestD_InvalidInputs(Base):
    def assert_invalid(self, field_in_message, **changes):
        spec = {**self.example, **changes}
        with self.assertRaises(InvalidSpecError) as ctx:
            predict_laptop_price(spec)
        self.assertIn(field_in_message, str(ctx.exception))

    def test_invalid_values(self):
        self.assert_invalid('ram_gb', ram_gb=-8)
        self.assert_invalid('ram_gb', ram_gb=0)
        self.assert_invalid('ram_gb', ram_gb=8.5)
        self.assert_invalid('ram_gb', ram_gb='8GB')
        self.assert_invalid('ram_gb', ram_gb=True)
        self.assert_invalid('weight_kg', weight_kg=-1.2)
        self.assert_invalid('weight_kg', weight_kg=0)
        self.assert_invalid('weight_kg', weight_kg=50)
        self.assert_invalid('weight_kg', weight_kg=float('nan'))
        self.assert_invalid('inches', inches=0)
        self.assert_invalid('resolution_width', resolution_width=-1920)
        self.assert_invalid('resolution_height', resolution_height=1080.5)
        self.assert_invalid('ssd_gb', ssd_gb=-256)
        self.assert_invalid('hdd_gb', hdd_gb=-1)
        self.assert_invalid('touchscreen', touchscreen='yes')
        self.assert_invalid('company', company='  ')
        self.assert_invalid('cpu_ghz', cpu='Intel Core i5 7200U', cpu_ghz=-2.5)
        self.assert_invalid('cpu_ghz', cpu='Intel Core i5 7200U')                     # no GHz anywhere
        self.assert_invalid('contradicts', cpu='Intel Core i5 7200U 2.5GHz', cpu_ghz=3.1)

    def test_missing_and_unknown_fields(self):
        spec = dict(self.example); spec.pop('ram_gb')
        with self.assertRaisesRegex(InvalidSpecError, 'Missing required'):
            predict_laptop_price(spec)
        with self.assertRaisesRegex(InvalidSpecError, 'Unknown field'):
            predict_laptop_price({**self.example, 'ram': 8})
        with self.assertRaises(InvalidSpecError):
            predict_laptop_price(['not', 'a', 'dict'])

    def test_valid_edge_inputs_accepted(self):
        ok = predict_laptop_price({**self.example, 'cpu': 'Intel Core i5 7200U', 'cpu_ghz': 2.5, 'touchscreen': 1,
                                   'ram_gb': 8.0})
        self.assertGreater(ok['predicted_price'], 0)


class TestE_UnknownCategories(Base):
    def test_unseen_values_do_not_crash(self):
        spec = {**self.example, 'company': 'Framework', 'type_name': 'Tablet', 'os': 'FreeBSD',
                'cpu': 'Apple M2 3.5GHz', 'gpu': 'Nvidia GeForce RTX 4090'}
        result = predict_laptop_price(spec)
        self.assertTrue(math.isfinite(result['predicted_price']))
        self.assertEqual(result['features']['Company_grouped'], 'Other')
        self.assertEqual(result['features']['os'], 'Others/No OS/Linux')
        self.assertEqual(len(result['warnings']), 5)

    def test_rare_training_company_grouped(self):
        result = predict_laptop_price({**self.example, 'company': 'Razer'})
        self.assertEqual(result['features']['Company_grouped'], 'Other')
        self.assertEqual(result['warnings'], [])        # Razer is known, just rare

    def test_out_of_range_numbers_warn(self):
        result = predict_laptop_price({**self.example, 'ram_gb': 128})
        self.assertTrue(any('ram_gb' in w for w in result['warnings']))


class TestF_SanityOfPredictions(Base):
    def test_all_training_rows_sane(self):
        X = build_features_from_raw(self.raw.drop(columns=['Price']))
        prices = np.exp(self.pipeline.predict(X))
        self.assertTrue(np.isfinite(prices).all())
        self.assertTrue((prices > 0).all())
        self.assertTrue(((prices > SANITY_RANGE[0]) & (prices < SANITY_RANGE[1])).all())

    def test_single_prediction_sane(self):
        result = predict_laptop_price(self.example)
        p = result['predicted_price']
        self.assertTrue(math.isfinite(p) and SANITY_RANGE[0] < p < SANITY_RANGE[1])
        self.assertAlmostEqual(p, math.exp(result['log_price_prediction']))
        self.assertEqual(result['predicted_price_rounded'] % 100, 0)


if __name__ == '__main__':
    unittest.main()
