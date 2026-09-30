"""Small, UI-only helpers for the Streamlit app (no Streamlit import, so they are easy to test).

Nothing here engineers model features. The app packs the user's form values into the plain specs dict
expected by `src.predict.predict_laptop_price()`; all feature engineering stays in `src.features`.
"""
import math

from src.predict import REQUIRED_FIELDS, InvalidSpecError, predict_laptop_price

# Display names for common resolutions (labels only - the model receives width and height).
RESOLUTION_NAMES = {
    (1366, 768): 'HD',
    (1600, 900): 'HD+',
    (1920, 1080): 'Full HD',
    (1920, 1200): 'WUXGA',
    (2560, 1440): 'QHD',
    (2560, 1600): 'WQXGA',
    (3200, 1800): 'QHD+',
    (3840, 2160): '4K UHD',
}
RESOLUTION_SEPARATOR = ' × '


def format_inr(amount, decimals=0):
    """Format rupees with Indian digit grouping: 123456 -> '₹1,23,456'; 48000 -> '₹48,000'."""
    if amount is None or not math.isfinite(amount):
        raise ValueError(f'Cannot format amount {amount!r}')
    sign = '-' if amount < 0 else ''
    text = f'{abs(amount):.{decimals}f}'
    whole, _, frac = text.partition('.')
    if len(whole) > 3:
        head, tail = whole[:-3], whole[-3:]
        groups = []
        while len(head) > 2:
            groups.insert(0, head[-2:])
            head = head[:-2]
        if head:
            groups.insert(0, head)
        whole = ','.join(groups + [tail])
    return f"{sign}₹{whole}{'.' + frac if frac else ''}"


def format_resolution(width, height):
    """(1920, 1080) -> '1920 × 1080'."""
    return f'{width}{RESOLUTION_SEPARATOR}{height}'


def resolution_label(value):
    """'1920 × 1080' -> '1920 × 1080  (Full HD)' for the dropdown; unnamed resolutions are shown as-is."""
    name = RESOLUTION_NAMES.get(parse_resolution(value))
    return f'{value}  ({name})' if name else value


def parse_resolution(value):
    """'1920 × 1080' -> (1920, 1080). Raises ValueError for anything else."""
    parts = value.split(RESOLUTION_SEPARATOR)
    if len(parts) != 2 or not all(p.isdigit() for p in parts):
        raise ValueError(f'Unexpected resolution format: {value!r}')
    width, height = int(parts[0]), int(parts[1])
    if width <= 0 or height <= 0:
        raise ValueError(f'Resolution must be positive: {value!r}')
    return width, height


def build_specs(*, company, type_name, ram_gb, weight_kg, os, cpu, gpu, ssd_gb, hdd_gb, inches, resolution,
                touchscreen, ips):
    """Pack form values into the exact dict `predict_laptop_price()` expects (plain Python types only)."""
    width, height = parse_resolution(resolution)
    specs = {
        'company': company,
        'type_name': type_name,
        'ram_gb': int(ram_gb),
        'weight_kg': float(weight_kg),
        'inches': float(inches),
        'resolution_width': width,
        'resolution_height': height,
        'touchscreen': bool(touchscreen),
        'ips': bool(ips),
        'cpu': cpu,
        'gpu': gpu,
        'os': os,
        'ssd_gb': int(ssd_gb),
        'hdd_gb': int(hdd_gb),
    }
    assert sorted(specs) == sorted(REQUIRED_FIELDS)
    return specs


def run_prediction(specs):
    """Call the production interface. Returns (result, None) on success or (None, error message) for invalid input."""
    try:
        return predict_laptop_price(specs), None
    except InvalidSpecError as exc:
        return None, str(exc)


def compare_listed_price(estimate, listed):
    """Neutral comparison of a seller's listed price with the model estimate (both in rupees).

    The amount difference is rounded to the nearest 100 rupees, like the displayed estimate.
    Returns {'difference', 'percent', 'direction' ('above'/'below'/'about the same'), 'amount_text', 'sentence'}.
    """
    for name, v in (('estimate', estimate), ('listed price', listed)):
        if v is None or isinstance(v, bool) or not math.isfinite(v) or v <= 0:
            raise ValueError(f'The {name} must be a positive amount, got {v!r}')
    difference = listed - estimate
    percent = difference / estimate * 100
    rounded_diff = int(round(abs(difference), -2))
    if round(abs(percent)) == 0:
        direction = 'about the same'
        amount_text = 'Within 1% of the model estimate'
        sentence = "The listed price is about the same as this model's estimate."
    else:
        direction = 'above' if difference > 0 else 'below'
        amount_text = f'{format_inr(rounded_diff)} {direction} the model estimate'
        sentence = f"The listed price is {abs(percent):.0f}% {direction} this model's estimate."
    return {'difference': difference, 'percent': percent, 'direction': direction,
            'amount_text': amount_text, 'sentence': sentence}


def model_summary(metadata):
    """Numbers shown in the sidebar / reliability panel, read from model_metadata.json."""
    ev = metadata['evaluation_before_final_refit']
    model_type = metadata['model_type']
    return {
        'model_label': 'XGBoost Regressor' if 'XGBRegressor' in model_type else model_type,
        'model_version': metadata['model_version'],
        'training_rows': metadata['training_data']['n_training_observations'],
        'holdout_rows': ev['holdout_rows'],
        'r2_log_price': ev['holdout']['r2_log_price'],
        'mae_rupees': ev['holdout']['mae_rupees'],
        'median_ape_percent': ev['holdout']['median_ape_percent'],
        'mape_percent': ev['holdout']['mape_percent'],
    }
