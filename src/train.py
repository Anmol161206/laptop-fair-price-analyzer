"""Train and save the frozen production model.

    conda run -n laptop_price python -m src.train

Steps
1. Load the raw CSV, remove the ARM row and the 29 exact duplicate copies (-> 1,273 rows), as in notebook 05.
2. Build features with src.features (Price is dropped before feature engineering).
3. Safety check: re-run notebook 05's locked XGBoost holdout evaluation with THIS code and assert that it
   reproduces the numbers reported there. This proves the refactored features/pipeline are equivalent.
4. Refit the same pipeline (no new tuning) on all 1,273 rows and save the model plus metadata.
"""
import hashlib
import json
import platform
import sys
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

from src.features import (CATEGORICAL_FEATURES, MODEL_FEATURES, NUMERICAL_FEATURES, OTHER_COMPANY,
                          PIPELINE_INPUT_COLUMNS, RARE_COMPANY_MIN_COUNT, SPEC_FIELDS, CompanyGrouper,
                          engineer_features, specs_from_raw)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / 'laptop_data.csv'
ARTIFACT_DIR = PROJECT_ROOT / 'artifacts'
MODEL_PATH = ARTIFACT_DIR / 'laptop_price_model.joblib'
METADATA_PATH = ARTIFACT_DIR / 'model_metadata.json'

MODEL_NAME = 'laptop-price-xgboost'
MODEL_VERSION = '1.0.0'
EXPECTED_ROWS = 1273

# Hyperparameters selected by RandomizedSearchCV in notebook 05 (group-aware CV on the 1,015 training rows).
XGB_PARAMS = {
    'n_estimators': 300, 'learning_rate': 0.02, 'max_depth': 8, 'subsample': 0.8, 'colsample_bytree': 0.4,
    'min_child_weight': 1, 'reg_lambda': 0.5,
    'random_state': 2, 'n_jobs': 1, 'tree_method': 'hist',
}

# Notebook 05 evaluation protocol (split seed 42, 20% of configuration groups held out).
SPLIT_SEED, TEST_SIZE = 42, 0.20
RAW_SPEC_COLUMNS = ['Company', 'TypeName', 'Inches', 'ScreenResolution', 'Cpu', 'Ram', 'Memory', 'Gpu', 'OpSys', 'Weight']

# XGBoost results printed by notebook 05 (locked protocol, model fitted on 1,015 rows, scored on 258 unseen rows).
LOCKED_EVALUATION = {
    'protocol': ('Notebook 05: exact duplicates and ARM row removed (1,273 rows); GroupShuffleSplit grouped by full '
                 'raw specification, test_size=0.20, random_state=42 -> 1,015 train / 258 holdout rows; '
                 'hyperparameters chosen by group-aware 5-fold CV on the training rows only; holdout scored once.'),
    'note': ('These metrics come from the locked evaluation BEFORE the final refit. The saved production model was '
             'refit on all 1,273 rows and therefore has no independent test score of its own; these numbers are the '
             'best estimate of its performance on new laptops.'),
    'train_rows': 1015,
    'holdout_rows': 258,
    'holdout': {
        'r2_log_price': 0.915952,
        'r2_price': 0.8233,
        'mae_rupees': 8700.4769,
        'rmse_rupees': 15275.0533,
        'mape_percent': 13.9160,
        'median_ape_percent': 12.2205,
        'mae_log_price': 0.1367,
        'share_within_10_20_30_percent': [0.45, 0.76, 0.93],
        'bootstrap_95ci_r2_log_price': [0.897, 0.933],
        'bootstrap_95ci_mae_rupees': [7294, 10470],
    },
    'repeated_cv': {
        'scheme': 'GroupKFold 5 folds x 5 repeats on the 1,015 training rows',
        'r2_log_price_mean': 0.8974, 'r2_log_price_std': 0.0138,
        'r2_price_mean': 0.8203, 'r2_price_std': 0.0481,
        'mae_rupees_mean': 9564.8638, 'mae_rupees_std': 1001.5546,
    },
}

KNOWN_LIMITATIONS = [
    'Trained on a single dataset of 1,273 laptops listed around 2017-2018 (Intel 6th-8th gen, GTX 10-series era); '
    'prices of newer hardware (e.g. Intel 10th gen+, Ryzen 5000+, RTX GPUs, Apple Silicon) are extrapolations.',
    'Prices shrink towards the middle: budget laptops are over-predicted (~ -11% mean signed error) and laptops '
    'above Rs 100k under-predicted (~ +12%); errors are largest for very expensive and workstation laptops.',
    'Typical error is about 12% (median absolute percentage error on the holdout); identical configurations in '
    'the data already differ in price by about 4%, so no model can be exact.',
    "Any CPU that does not start with 'Intel' is encoded as 'AMD Processor' (true in the training data, "
    'not for e.g. Apple M-series or Qualcomm).',
    'Companies with fewer than 10 training laptops, and companies never seen in training, are grouped as "Other".',
    'Categories unseen in training (e.g. a new TypeName) are encoded as all zeros by the one-hot encoder '
    '(drop="first"), which makes them look like the first (dropped) category of that feature.',
    'Flash Storage and Hybrid drives are not used as features; a laptop with only flash storage has SSD=0 and HDD=0.',
    'Currency is the dataset\'s price unit (Indian rupees).',
]


def load_clean_raw(path=DATA_PATH):
    """Raw CSV -> drop the exported index column, the ARM row and exact duplicate copies (notebook 05 rules)."""
    raw = pd.read_csv(path)
    assert (raw['Unnamed: 0'] == np.arange(len(raw))).all(), "'Unnamed: 0' is not a plain row index"
    raw = raw.drop(columns=['Unnamed: 0'])
    raw = raw[~raw['Gpu'].str.startswith('ARM')]
    raw = raw[~raw.duplicated(keep='first')].reset_index(drop=True)
    return raw


def make_pipeline():
    """CompanyGrouper -> one-hot encoding of categoricals (numerics passed through) -> XGBoost."""
    preprocess = ColumnTransformer([
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False), CATEGORICAL_FEATURES),
        ('num', 'passthrough', NUMERICAL_FEATURES),
    ])
    return Pipeline([
        ('company_grouper', CompanyGrouper(min_count=RARE_COMPANY_MIN_COUNT)),
        ('preprocess', preprocess),
        ('model', XGBRegressor(**XGB_PARAMS)),
    ])


def reproduce_locked_evaluation(raw, X, y):
    """Re-run notebook 05's holdout evaluation of XGBoost with the production code; return the metrics."""
    groups = raw.groupby(RAW_SPEC_COLUMNS).ngroup()
    train_pos, test_pos = next(GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=SPLIT_SEED)
                               .split(X, y, groups))
    pipe = make_pipeline().fit(X.iloc[train_pos], y.iloc[train_pos])
    pred = pipe.predict(X.iloc[test_pos])
    return {
        'train_rows': len(train_pos), 'holdout_rows': len(test_pos),
        'r2_log_price': r2_score(y.iloc[test_pos], pred),
        'mae_rupees': mean_absolute_error(np.exp(y.iloc[test_pos]), np.exp(pred)),
        'rare_companies': pipe.named_steps['company_grouper'].rare_companies_,
    }


def observed_input_values(specs):
    """Values that actually occur in the training data - used for website dropdowns (with laptop counts)."""
    def counts(col, cast):
        vc = specs[col].value_counts().sort_index()
        return [{'value': cast(v), 'count': int(n)} for v, n in vc.items()]
    res = (specs.groupby(['resolution_width', 'resolution_height']).size()
           .sort_values(ascending=False, kind='stable'))
    return {
        'ram_gb': counts('ram_gb', int),
        'ssd_gb': counts('ssd_gb', int),
        'hdd_gb': counts('hdd_gb', int),
        'inches': counts('inches', float),
        'resolutions': [{'width': int(w), 'height': int(h), 'count': int(n)} for (w, h), n in res.items()],
    }


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(metadata_only=False):
    """Train + save model and metadata. With metadata_only=True, rebuild model_metadata.json for the EXISTING
    saved model (no training); the model file must be unchanged (checksum verified)."""
    raw = load_clean_raw()
    assert len(raw) == EXPECTED_ROWS, f'expected {EXPECTED_ROWS} clean rows, got {len(raw)}'
    specs = specs_from_raw(raw.drop(columns=['Price']))      # Price removed before any feature engineering
    X = engineer_features(specs)
    y = np.log(raw['Price'])
    print(f'Clean rows: {len(raw)} | pipeline input columns: {len(X.columns)}')

    if metadata_only:
        previous = json.loads(METADATA_PATH.read_text())
        if sha256(MODEL_PATH) != previous['model_file_sha256']:
            raise RuntimeError('Saved model does not match the existing metadata; retrain instead.')
        pipe = joblib.load(MODEL_PATH)
        write_metadata(raw, specs, pipe, previous['equivalence_check'], previous['training_date'])
        print('Rebuilt metadata for the existing model (not retrained); checksum unchanged.')
        return

    # ---- 3. equivalence check against notebook 05 ----
    check = reproduce_locked_evaluation(raw, X, y)
    print(f"Reproduced notebook 05 holdout: R2(log) = {check['r2_log_price']:.6f}, MAE = {check['mae_rupees']:.4f} "
          f"({check['train_rows']} train / {check['holdout_rows']} holdout)")
    assert (check['train_rows'], check['holdout_rows']) == (1015, 258)
    assert abs(check['r2_log_price'] - LOCKED_EVALUATION['holdout']['r2_log_price']) < 1e-5
    assert abs(check['mae_rupees'] - LOCKED_EVALUATION['holdout']['mae_rupees']) < 0.01
    print('Production code reproduces the notebook 05 result exactly.')

    # ---- 4. final refit on all rows (no tuning) ----
    pipe = make_pipeline().fit(X, y)
    ARTIFACT_DIR.mkdir(exist_ok=True)
    joblib.dump(pipe, MODEL_PATH)
    equivalence = {
        'description': 'Notebook 05 holdout evaluation re-run with the production code before refitting',
        'r2_log_price': round(check['r2_log_price'], 6), 'mae_rupees': round(check['mae_rupees'], 4),
    }
    write_metadata(raw, specs, pipe, equivalence, date.today().isoformat())
    print(f'Saved {MODEL_PATH.relative_to(PROJECT_ROOT)} ({MODEL_PATH.stat().st_size / 1024:.0f} KB)')
    print(f'Saved {METADATA_PATH.relative_to(PROJECT_ROOT)}')
    print('Rare companies (fitted on all rows):', pipe.named_steps['company_grouper'].rare_companies_)


def write_metadata(raw, specs, pipe, equivalence, training_date):
    grouper = pipe.named_steps['company_grouper']
    encoder = pipe.named_steps['preprocess'].named_transformers_['cat']

    num_ranges = {c: [float(specs[c].min()), float(specs[c].max())]
                  for c in ['ram_gb', 'weight_kg', 'inches', 'resolution_width', 'resolution_height',
                            'cpu_ghz', 'ssd_gb', 'hdd_gb']}
    metadata = {
        'model_name': MODEL_NAME,
        'model_version': MODEL_VERSION,
        'model_type': 'scikit-learn Pipeline: CompanyGrouper -> ColumnTransformer(OneHotEncoder, passthrough) -> XGBRegressor',
        'training_date': training_date,
        'model_file': MODEL_PATH.name,
        'model_file_sha256': sha256(MODEL_PATH),
        'model_file_size_bytes': MODEL_PATH.stat().st_size,
        'training_data': {
            'source': DATA_PATH.name,
            'raw_rows': int(len(pd.read_csv(DATA_PATH))),
            'removed': 'Samsung ARM laptop (1 row) and 29 exact duplicate copies',
            'n_training_observations': int(len(raw)),
            'price_range_rupees': [float(raw['Price'].min()), float(raw['Price'].max())],
        },
        'target_transformation': 'log',
        'target_transformation_detail': 'model is trained on y = numpy.log(Price) (natural log)',
        'price_inverse_transformation': 'exp',
        'price_inverse_transformation_detail': 'predicted Price = numpy.exp(model prediction)',
        'currency': 'INR',
        'input_spec_fields': SPEC_FIELDS,
        'pipeline_input_columns': PIPELINE_INPUT_COLUMNS,
        'final_feature_names': MODEL_FEATURES,
        'categorical_features': CATEGORICAL_FEATURES,
        'numerical_features': NUMERICAL_FEATURES,
        'company_grouping': {
            'rule': f'companies with fewer than {RARE_COMPANY_MIN_COUNT} laptops in the training data, and any '
                    f'company not seen in training, are mapped to "{OTHER_COMPANY}"',
            'min_count': RARE_COMPANY_MIN_COUNT,
            'fitted_on_rows': int(len(raw)),
            'frequent_companies': grouper.frequent_companies_,
            'rare_companies': grouper.rare_companies_,
        },
        'category_levels': {f: [str(v) for v in cats] for f, cats in zip(CATEGORICAL_FEATURES, encoder.categories_)},
        'known_raw_values': {
            'company': sorted(raw['Company'].unique().tolist()),
            'type_name': sorted(raw['TypeName'].unique().tolist()),
            'os': sorted(raw['OpSys'].unique().tolist()),
            'cpu': sorted(raw['Cpu'].str.strip().unique().tolist()),
            'gpu': sorted(raw['Gpu'].str.strip().unique().tolist()),
        },
        'training_numeric_ranges': num_ranges,
        'observed_input_values': observed_input_values(specs),
        'hyperparameters': XGB_PARAMS,
        'evaluation_before_final_refit': LOCKED_EVALUATION,
        'equivalence_check': equivalence,
        'known_limitations': KNOWN_LIMITATIONS,
        'environment': {
            'python': platform.python_version(), 'scikit-learn': sklearn.__version__,
            'xgboost': xgboost.__version__, 'pandas': pd.__version__, 'numpy': np.__version__,
            'joblib': joblib.__version__,
            'note': 'Load the joblib file with the same library versions (see environment.yml); '
                    'the src package must be importable because the pipeline contains src.features.CompanyGrouper.',
        },
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    sys.exit(main(metadata_only='--metadata-only' in sys.argv))
