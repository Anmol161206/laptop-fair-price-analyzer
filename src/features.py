"""Feature engineering for the laptop price model - the ONE source of truth.

Every consumer (training script, prediction interface, tests, and later the website) must build
features through this module. The rules are the ones validated in notebook 05 and must not be
changed without re-running the evaluation protocol.

Data flow
---------
    raw dataset row  --specs_from_raw()-->  specs  --engineer_features()-->  pipeline input
    user input dict  --(predict.py validation)-->  specs  --engineer_features()-->  pipeline input

"specs" is a human-readable, parsed description of a laptop (one row per laptop, columns in
SPEC_FIELDS). Both paths meet at `engineer_features()`, so training and inference cannot drift apart.

The pipeline input still contains the raw `Company`; the fitted `CompanyGrouper` (the first step of
the saved pipeline) turns it into `Company_grouped` using the company list learned at training time.

No function here reads `Price`, so features can never contain target information.
"""
import re

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

# --------------------------------------------------------------------------------------------------
# Column definitions
# --------------------------------------------------------------------------------------------------
SPEC_FIELDS = [
    'company',            # manufacturer, e.g. 'Dell'
    'type_name',          # 'Notebook', 'Ultrabook', 'Gaming', '2 in 1 Convertible', 'Workstation', 'Netbook'
    'ram_gb',             # int, RAM in GB
    'weight_kg',          # float, weight in kg
    'inches',             # float, screen diagonal in inches
    'resolution_width',   # int, horizontal pixels, e.g. 1920
    'resolution_height',  # int, vertical pixels, e.g. 1080
    'touchscreen',        # bool
    'ips',                # bool, IPS panel
    'cpu',                # CPU description as in the dataset, e.g. 'Intel Core i5 7200U 2.5GHz'
    'cpu_ghz',            # float, base clock in GHz
    'gpu',                # GPU description as in the dataset, e.g. 'Nvidia GeForce GTX 1050'
    'os',                 # operating system as in the dataset, e.g. 'Windows 10', 'macOS', 'No OS'
    'ssd_gb',             # int, SSD capacity in GB (1TB = 1000GB)
    'hdd_gb',             # int, HDD capacity in GB (1TB = 1000GB)
]

# Order matters: it fixes the column order the model was evaluated with in notebook 05.
CATEGORICAL_FEATURES = ['TypeName', 'Cpu brand', 'os', 'cpu_type', 'cpu_gen', 'gpu_tier', 'Company_grouped']
NUMERICAL_FEATURES = ['Ram', 'Weight', 'Touchscreen', 'Ips', 'ppi', 'SSD', 'HDD', 'cpu_ghz', 'Inches', 'X_res', 'Y_res']
MODEL_FEATURES = CATEGORICAL_FEATURES + NUMERICAL_FEATURES

# What engineer_features() returns: the model features, but with raw 'Company' (grouped inside the pipeline)
PIPELINE_INPUT_COLUMNS = ['Company'] + [c for c in MODEL_FEATURES if c != 'Company_grouped']

RARE_COMPANY_MIN_COUNT = 10   # companies with fewer than 10 training laptops -> 'Other' (rule from notebook 04/05)
OTHER_COMPANY = 'Other'


# --------------------------------------------------------------------------------------------------
# Parsers for the raw dataset's string formats
# --------------------------------------------------------------------------------------------------
def parse_ram(text):
    """'8GB' -> 8. Raises ValueError for any other format."""
    if not re.fullmatch(r'\d+GB', text):
        raise ValueError(f'Unexpected Ram format: {text!r}')
    return int(text[:-2])


def parse_weight(text):
    """'1.37kg' -> 1.37. Raises ValueError for any other format."""
    if not re.fullmatch(r'\d+(\.\d+)?kg', text):
        raise ValueError(f'Unexpected Weight format: {text!r}')
    return float(text[:-2])


def parse_screen_resolution(text):
    """Parse a ScreenResolution string such as 'IPS Panel Full HD / Touchscreen 1920x1080'.

    Returns a dict with:
      touchscreen - True if the text contains 'Touchscreen' (case-insensitive)
      ips         - True if the text contains 'IPS' (case-insensitive)
      width       - horizontal pixels: the digits immediately before the first 'x'
      height      - vertical pixels: everything after the first 'x'
    """
    left, right = text.split('x', 1)
    return {
        'touchscreen': 'touchscreen' in text.lower(),
        'ips': 'ips' in text.lower(),
        'width': int(re.search(r'(\d+)$', left).group(1)),
        'height': int(right),
    }


def parse_memory(text):
    """Parse a Memory string such as '128GB SSD +  1TB HDD' into GB per storage type.

    Returns {'SSD', 'HDD', 'Flash Storage', 'Hybrid'} in GB (1TB = 1000GB). Only SSD and HDD are
    model features; Flash Storage and Hybrid were tested in notebook 04 and did not help.
    """
    sizes = {'SSD': 0, 'HDD': 0, 'Flash Storage': 0, 'Hybrid': 0}
    for part in text.split('+'):
        match = re.fullmatch(r'(\d+(?:\.\d+)?)(GB|TB) (SSD|HDD|Flash Storage|Hybrid)', part.strip())
        if match is None:
            raise ValueError(f'Unexpected Memory format: {text!r}')
        size, unit, kind = match.groups()
        sizes[kind] += int(float(size) * (1000 if unit == 'TB' else 1))
    return sizes


def extract_cpu_ghz(cpu):
    """First '<number>GHz' in a CPU description, e.g. 'Intel Core i5 7200U 2.5GHz' -> 2.5; None if absent."""
    match = re.search(r'([\d.]+)GHz', cpu)
    return float(match.group(1)) if match else None


# --------------------------------------------------------------------------------------------------
# Engineered categorical features (rules exactly as validated in notebook 05)
# --------------------------------------------------------------------------------------------------
def cpu_brand(cpu):
    """CPU family from the first three words of the CPU description.

    'Intel Core i7' / 'Intel Core i5' / 'Intel Core i3' are kept; any other Intel CPU (Celeron,
    Pentium, Atom, Core M, Xeon...) -> 'Other Intel Processor'; anything not starting with 'Intel'
    -> 'AMD Processor' (in the training data every non-Intel CPU was AMD, after removing the ARM row).
    """
    name = ' '.join(cpu.split()[:3])
    if name in ('Intel Core i7', 'Intel Core i5', 'Intel Core i3'):
        return name
    return 'Other Intel Processor' if name.split()[0] == 'Intel' else 'AMD Processor'


def cpu_type(cpu):
    """CPU power class from the Intel model-number suffix.

    'High-performance (H)' - HQ/HK suffix (gaming / workstation chips, e.g. 7700HQ)
    'Low-power (U)'        - U suffix (thin-and-light, e.g. 7200U)
    'Ultra-low-power (Y)'  - Y in the model number (e.g. 7Y54, Core M 6Y75)
    'Other'                - no recognisable suffix (Celeron, Pentium, Atom, AMD, Xeon...)
    """
    if re.search(r'\d{4}H[QK]\b', cpu):
        return 'High-performance (H)'
    if re.search(r'\d{4}U\b', cpu):
        return 'Low-power (U)'
    if re.search(r'\dY\d{2}', cpu):
        return 'Ultra-low-power (Y)'
    return 'Other'


def cpu_gen(cpu):
    """Intel Core generation from the first digit of the model number: 7200U -> 'Gen 7'.

    Only Intel Core i3/i5/i7/M CPUs with a readable model number get a generation; all others
    (AMD, Celeron, Pentium, Atom, Xeon, 'Intel Core i5 1.6GHz' without a model number) -> 'Unknown'.
    """
    match = re.search(r'Intel Core (?:i\d|M) (?:[mM]\d-)?(\d)(?:\d{3}|Y\d{2})', cpu)
    return f'Gen {match.group(1)}' if match else 'Unknown'


def gpu_tier(gpu):
    """GPU performance tier from the GPU description (checked in this order):

    'Nvidia GTX'                 - GeForce GTX gaming cards
    'Nvidia Quadro'              - workstation cards
    'Nvidia other (MX/GT/9xxMX)' - entry-level Nvidia (MX150, 940MX...)
    'AMD Pro/FirePro'            - AMD workstation / Apple 'Radeon Pro' cards
    'AMD other'                  - other AMD Radeon
    'Intel Iris'                 - Intel Iris / Iris Plus / Iris Pro integrated graphics
    'Intel HD/UHD'               - everything else (Intel HD / UHD integrated graphics)
    """
    if 'GTX' in gpu:                              return 'Nvidia GTX'
    if 'Quadro' in gpu:                           return 'Nvidia Quadro'
    if gpu.startswith('Nvidia'):                  return 'Nvidia other (MX/GT/9xxMX)'
    if 'FirePro' in gpu or 'Radeon Pro' in gpu:   return 'AMD Pro/FirePro'
    if gpu.startswith('AMD'):                     return 'AMD other'
    if 'Iris' in gpu:                             return 'Intel Iris'
    return 'Intel HD/UHD'


def os_category(os_name):
    """'Windows' (Windows 10 / 7 / 10 S), 'Mac' (macOS / Mac OS X), else 'Others/No OS/Linux'
    (Linux, No OS, Chrome OS, Android, and anything unrecognised)."""
    if os_name in ('Windows 10', 'Windows 7', 'Windows 10 S'):
        return 'Windows'
    if os_name in ('macOS', 'Mac OS X'):
        return 'Mac'
    return 'Others/No OS/Linux'


# --------------------------------------------------------------------------------------------------
# Raw dataset -> specs -> model input
# --------------------------------------------------------------------------------------------------
def specs_from_raw(raw):
    """Convert rows in the original CSV format (Company, TypeName, Inches, ScreenResolution, Cpu,
    Ram, Memory, Gpu, OpSys, Weight) into the parsed `specs` table (columns = SPEC_FIELDS).

    `Price` is not needed and is ignored if present.
    """
    screen = raw['ScreenResolution'].map(parse_screen_resolution)
    memory = raw['Memory'].map(parse_memory)
    cpu_ghz = raw['Cpu'].map(extract_cpu_ghz)
    if cpu_ghz.isna().any():
        raise ValueError(f'CPU descriptions without GHz: {raw.loc[cpu_ghz.isna(), "Cpu"].tolist()}')
    specs = pd.DataFrame({
        'company': raw['Company'],
        'type_name': raw['TypeName'],
        'ram_gb': raw['Ram'].map(parse_ram),
        'weight_kg': raw['Weight'].map(parse_weight),
        'inches': raw['Inches'].astype(float),
        'resolution_width': screen.map(lambda s: s['width']),
        'resolution_height': screen.map(lambda s: s['height']),
        'touchscreen': screen.map(lambda s: s['touchscreen']),
        'ips': screen.map(lambda s: s['ips']),
        'cpu': raw['Cpu'],
        'cpu_ghz': cpu_ghz.astype(float),
        'gpu': raw['Gpu'],
        'os': raw['OpSys'],
        'ssd_gb': memory.map(lambda m: m['SSD']),
        'hdd_gb': memory.map(lambda m: m['HDD']),
    }, index=raw.index)
    return specs[SPEC_FIELDS]


def engineer_features(specs):
    """Turn parsed laptop specs (columns = SPEC_FIELDS) into the pipeline input (PIPELINE_INPUT_COLUMNS).

    Engineered features:
      Ram, Weight, Inches   - copied from the specs (GB, kg, inches)
      Touchscreen, Ips      - 1/0 flags
      X_res, Y_res          - screen resolution in pixels
      ppi                   - pixels per inch = sqrt(X_res^2 + Y_res^2) / Inches
      SSD, HDD              - storage in GB
      cpu_ghz               - CPU base clock in GHz
      Cpu brand             - see cpu_brand()
      cpu_type              - see cpu_type()
      cpu_gen               - see cpu_gen()
      gpu_tier              - see gpu_tier()
      os                    - see os_category()
      TypeName              - laptop category, unchanged
      Company               - raw manufacturer; becomes Company_grouped inside the fitted pipeline
    """
    missing = [c for c in SPEC_FIELDS if c not in specs.columns]
    if missing:
        raise KeyError(f'specs are missing columns: {missing}')

    d = pd.DataFrame(index=specs.index)
    d['Company'] = specs['company']
    d['TypeName'] = specs['type_name']
    d['Ram'] = specs['ram_gb'].astype(int)
    d['Weight'] = specs['weight_kg'].astype(float)
    d['Touchscreen'] = specs['touchscreen'].astype(int)
    d['Ips'] = specs['ips'].astype(int)
    d['X_res'] = specs['resolution_width'].astype(int)
    d['Y_res'] = specs['resolution_height'].astype(int)
    d['Inches'] = specs['inches'].astype(float)
    d['ppi'] = np.sqrt(d['X_res'] ** 2 + d['Y_res'] ** 2) / d['Inches']
    d['Cpu brand'] = specs['cpu'].map(cpu_brand)
    d['SSD'] = specs['ssd_gb'].astype(int)
    d['HDD'] = specs['hdd_gb'].astype(int)
    d['os'] = specs['os'].map(os_category)
    d['cpu_ghz'] = specs['cpu_ghz'].astype(float)
    d['cpu_type'] = specs['cpu'].map(cpu_type)
    d['cpu_gen'] = specs['cpu'].map(cpu_gen)
    d['gpu_tier'] = specs['gpu'].map(gpu_tier)
    return d[PIPELINE_INPUT_COLUMNS]


def build_features_from_raw(raw):
    """Convenience: raw CSV-format rows -> pipeline input. Never uses `Price`."""
    return engineer_features(specs_from_raw(raw))


# --------------------------------------------------------------------------------------------------
# The one data-dependent rule: rare companies -> 'Other'
# --------------------------------------------------------------------------------------------------
class CompanyGrouper(BaseEstimator, TransformerMixin):
    """Replace the raw `Company` column with `Company_grouped`.

    fit():       counts laptops per company in the TRAINING data (counts only, never prices).
                 Companies with at least `min_count` laptops are "frequent" and keep their name.
    transform(): frequent companies keep their name; every other company -> 'Other'.
                 This includes the rare training companies AND companies never seen in training,
                 so a single prediction never recomputes rarity.
    """

    def __init__(self, min_count=RARE_COMPANY_MIN_COUNT):
        self.min_count = min_count

    def fit(self, X, y=None):
        counts = X['Company'].value_counts()
        self.company_counts_ = {str(k): int(v) for k, v in counts.items()}
        self.frequent_companies_ = sorted(str(c) for c in counts[counts >= self.min_count].index)
        self.rare_companies_ = sorted(str(c) for c in counts[counts < self.min_count].index)
        return self

    def transform(self, X):
        check_is_fitted(self, 'frequent_companies_')
        X = X.copy()
        X['Company_grouped'] = X['Company'].where(X['Company'].isin(self.frequent_companies_), OTHER_COMPANY)
        return X.drop(columns=['Company'])
