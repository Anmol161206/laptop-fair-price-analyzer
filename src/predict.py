"""Public prediction interface for the laptop price model.

    from src.predict import predict_laptop_price

    result = predict_laptop_price({
        'company': 'Dell', 'type_name': 'Notebook', 'ram_gb': 8, 'weight_kg': 2.2,
        'inches': 15.6, 'resolution_width': 1920, 'resolution_height': 1080,
        'touchscreen': False, 'ips': True,
        'cpu': 'Intel Core i5 8250U 1.6GHz', 'gpu': 'Intel UHD Graphics 620', 'os': 'Windows 10',
        'ssd_gb': 256, 'hdd_gb': 1000,
    })
    result['predicted_price']            # unrounded rupees

Callers give human-readable specs; all feature engineering happens in src.features (the same code
used for training). Only the saved artifacts and this package are needed - no notebook state.
"""
import hashlib
import json
import math
import numbers
import warnings
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.features import (MODEL_FEATURES, OTHER_COMPANY, SPEC_FIELDS, cpu_brand, engineer_features,
                          extract_cpu_ghz)

DEFAULT_ARTIFACT_DIR = Path(__file__).resolve().parents[1] / 'artifacts'
MODEL_FILENAME = 'laptop_price_model.joblib'
METADATA_FILENAME = 'model_metadata.json'

REQUIRED_FIELDS = [f for f in SPEC_FIELDS if f != 'cpu_ghz']
OPTIONAL_FIELDS = ['cpu_ghz']      # only needed when the CPU description has no 'GHz'
STRING_FIELDS = ['company', 'type_name', 'cpu', 'gpu', 'os']
BOOL_FIELDS = ['touchscreen', 'ips']

# Hard limits: values outside these are physically implausible for a laptop and are rejected.
# (Values inside the limits but outside the training range are accepted with a warning.)
NUMERIC_LIMITS = {
    #  field               (integer?, min allowed, min inclusive?, max allowed)
    'ram_gb':            (True,  1,    True,  256),
    'weight_kg':         (False, 0.3,  True,  10.0),
    'inches':            (False, 7.0,  True,  25.0),
    'resolution_width':  (True,  320,  True,  10000),
    'resolution_height': (True,  200,  True,  10000),
    'ssd_gb':            (True,  0,    True,  16000),
    'hdd_gb':            (True,  0,    True,  16000),
    'cpu_ghz':           (False, 0.0,  False, 10.0),
}


class InvalidSpecError(ValueError):
    """Raised when laptop specs are missing, malformed or physically implausible."""


# --------------------------------------------------------------------------------------------------
# Artifacts
# --------------------------------------------------------------------------------------------------
@lru_cache(maxsize=4)
def load_artifacts(artifact_dir=None):
    """Load (pipeline, metadata) once per process. Verifies the model file checksum and feature list."""
    artifact_dir = Path(artifact_dir) if artifact_dir else DEFAULT_ARTIFACT_DIR
    model_path, metadata_path = artifact_dir / MODEL_FILENAME, artifact_dir / METADATA_FILENAME
    for p in (model_path, metadata_path):
        if not p.exists():
            raise FileNotFoundError(f'Missing model artifact: {p}. Run `python -m src.train` first.')
    metadata = json.loads(metadata_path.read_text())
    digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
    if digest != metadata['model_file_sha256']:
        raise RuntimeError('Model file checksum does not match model_metadata.json - artifacts are inconsistent.')
    if metadata['final_feature_names'] != MODEL_FEATURES:
        raise RuntimeError('src.features feature list differs from the one the saved model was trained with.')
    pipeline = joblib.load(model_path)
    return pipeline, metadata


def get_input_options(artifact_dir=None):
    """Values seen in training - useful for website dropdowns and numeric input hints."""
    _, metadata = load_artifacts(artifact_dir)
    return {
        **metadata['known_raw_values'],
        'numeric_ranges': metadata['training_numeric_ranges'],
        'observed': metadata['observed_input_values'],
        'required_fields': REQUIRED_FIELDS,
        'optional_fields': OPTIONAL_FIELDS,
    }


# --------------------------------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------------------------------
def _check_number(field, value):
    integer, low, low_inclusive, high = NUMERIC_LIMITS[field]
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
        raise InvalidSpecError(f'{field} must be a number, got {value!r} ({type(value).__name__})')
    if not math.isfinite(value):
        raise InvalidSpecError(f'{field} must be a finite number, got {value!r}')
    if integer:
        if value != int(value):
            raise InvalidSpecError(f'{field} must be a whole number, got {value!r}')
        value = int(value)
    else:
        value = float(value)
    if value < low or (value == low and not low_inclusive) or value > high:
        bound = f'>= {low}' if low_inclusive else f'> {low}'
        raise InvalidSpecError(f'{field}={value!r} is not plausible for a laptop (must be {bound} and <= {high})')
    return value


def _check_bool(field, value):
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, numbers.Integral) and value in (0, 1):
        return bool(value)
    raise InvalidSpecError(f'{field} must be True/False (or 1/0), got {value!r}')


def _check_string(field, value):
    if not isinstance(value, str) or not value.strip():
        raise InvalidSpecError(f'{field} must be a non-empty string, got {value!r}')
    return value.strip()


def validate_specs(raw_specs, metadata):
    """Validate a user specs dict. Returns (clean specs dict with all SPEC_FIELDS, list of warnings)."""
    if not isinstance(raw_specs, dict):
        raise InvalidSpecError(f'specs must be a dict, got {type(raw_specs).__name__}')
    unknown = sorted(set(raw_specs) - set(REQUIRED_FIELDS) - set(OPTIONAL_FIELDS))
    if unknown:
        raise InvalidSpecError(f'Unknown field(s): {unknown}. Allowed: {REQUIRED_FIELDS + OPTIONAL_FIELDS}')
    missing = [f for f in REQUIRED_FIELDS if raw_specs.get(f) is None]
    if missing:
        raise InvalidSpecError(f'Missing required field(s): {missing}')

    spec = {}
    for f in STRING_FIELDS:
        spec[f] = _check_string(f, raw_specs[f])
    for f in BOOL_FIELDS:
        spec[f] = _check_bool(f, raw_specs[f])
    for f in ['ram_gb', 'weight_kg', 'inches', 'resolution_width', 'resolution_height', 'ssd_gb', 'hdd_gb']:
        spec[f] = _check_number(f, raw_specs[f])

    # CPU clock: from the CPU description (as in training) or supplied directly - never both disagreeing
    ghz_in_text = extract_cpu_ghz(spec['cpu'])
    supplied = raw_specs.get('cpu_ghz')
    if supplied is not None:
        supplied = _check_number('cpu_ghz', supplied)
        if ghz_in_text is not None and abs(ghz_in_text - supplied) > 1e-9:
            raise InvalidSpecError(f"cpu_ghz={supplied} contradicts the CPU description {spec['cpu']!r} ({ghz_in_text}GHz)")
    elif ghz_in_text is None:
        raise InvalidSpecError(f"cpu_ghz is required because the CPU description {spec['cpu']!r} contains no 'GHz'")
    else:
        _check_number('cpu_ghz', ghz_in_text)
    spec['cpu_ghz'] = supplied if supplied is not None else ghz_in_text

    # Warnings: valid input, but outside what the model learned from
    notes = []
    known = metadata['known_raw_values']
    if spec['company'] not in known['company']:
        notes.append(f"Company {spec['company']!r} was not in the training data; it is treated as '{OTHER_COMPANY}'.")
    if spec['type_name'] not in known['type_name']:
        notes.append(f"TypeName {spec['type_name']!r} was not in the training data; the model cannot use it "
                     f"(encoded like the baseline category {metadata['category_levels']['TypeName'][0]!r}).")
    if spec['os'] not in known['os']:
        notes.append(f"OS {spec['os']!r} was not in the training data; it is treated as 'Others/No OS/Linux'.")
    if spec['cpu'] not in known['cpu']:
        brand = cpu_brand(spec['cpu'])
        extra = (" Note: non-Intel CPUs are always treated as 'AMD Processor'."
                 if brand == 'AMD Processor' and not spec['cpu'].startswith('AMD') else '')
        notes.append(f"CPU {spec['cpu']!r} was not in the training data; derived brand = {brand!r}.{extra}")
    if spec['gpu'] not in known['gpu']:
        notes.append(f"GPU {spec['gpu']!r} was not in the training data; its tier is derived from the name.")
    for f, (lo, hi) in metadata['training_numeric_ranges'].items():
        if not lo <= spec[f] <= hi:
            notes.append(f'{f}={spec[f]} is outside the training range [{lo:g}, {hi:g}]; the prediction is an extrapolation.')
    return spec, notes


# --------------------------------------------------------------------------------------------------
# Prediction
# --------------------------------------------------------------------------------------------------
def predict_laptop_price(raw_specs, artifact_dir=None):
    """Predict the price (INR) of one laptop from human-readable specs.

    Required keys: company, type_name, ram_gb, weight_kg, inches, resolution_width, resolution_height,
                   touchscreen, ips, cpu, gpu, os, ssd_gb, hdd_gb
    Optional key:  cpu_ghz (only needed if the cpu description has no 'GHz')

    Returns a dict:
      predicted_price          - unrounded price in rupees (exp of the model's log-price prediction)
      predicted_price_rounded  - rounded to the nearest 100 rupees, for display
      log_price_prediction     - raw model output (natural log of price)
      currency                 - 'INR'
      features                 - the 18 engineered model features actually used
      warnings                 - notes about unseen categories / out-of-range values (empty if none)
      model_name, model_version

    Raises InvalidSpecError for missing, malformed or implausible input.
    """
    pipeline, metadata = load_artifacts(artifact_dir)
    spec, notes = validate_specs(raw_specs, metadata)

    X = engineer_features(pd.DataFrame([spec], columns=SPEC_FIELDS))
    with warnings.catch_warnings():
        # unseen categories are reported in `notes`; silence sklearn's generic duplicate warning
        warnings.filterwarnings('ignore', message='Found unknown categories', category=UserWarning)
        log_pred = float(pipeline.predict(X)[0])
    price = math.exp(log_pred)
    if not math.isfinite(price) or price <= 0:
        raise RuntimeError(f'Model produced an invalid price: {price!r}')

    model_features = pipeline.named_steps['company_grouper'].transform(X)[MODEL_FEATURES].iloc[0]
    features = {k: (v.item() if hasattr(v, 'item') else v) for k, v in model_features.items()}
    return {
        'predicted_price': price,
        'predicted_price_rounded': int(round(price, -2)),
        'log_price_prediction': log_pred,
        'currency': metadata['currency'],
        'features': features,
        'warnings': notes,
        'model_name': metadata['model_name'],
        'model_version': metadata['model_version'],
    }
