# Laptop Fair-Price Analyzer

Estimates a **reference price (INR)** for a laptop from its hardware specifications, using a machine-learning
model trained on a historical dataset of 1,273 laptops. You can also enter a seller's listed price to see how it
compares with the model's estimate.

> The app predicts a reference price from **historical** laptop data. It does **not** fetch live prices from
> Amazon, Flipkart or any other retailer, and its estimate is not a guaranteed market price.

**Live app:** _link added after deployment_

## Run locally

```bash
conda env create -f environment.yml      # first time only
conda activate laptop_price
streamlit run app.py
```

Then open http://localhost:8501.

## What the website does

- You pick a laptop's specifications from values present in the training data: company, type, RAM, weight, OS,
  CPU, GPU, SSD/HDD, screen size, resolution, touchscreen and IPS.
- It shows the model's estimated reference price.
- Optionally, it compares a seller's listed price with that estimate, using neutral wording and no buy/sell advice.
- It shows the model's measured error on unseen laptops, any input warnings, and the characteristics the model
  derived (CPU generation/class, GPU tier, pixel density).

## Model summary

| | |
|---|---|
| Model | XGBoost regressor inside a scikit-learn pipeline (company grouping, then one-hot encoding, then XGBoost) |
| Target | `log(Price)`; predictions are converted back with `exp()` |
| Training data | 1,273 laptops (raw 1,303 minus 1 ARM laptop and 29 exact duplicate rows) |
| Features (18) | TypeName, Company_grouped, Cpu brand, cpu_type, cpu_gen, gpu_tier, os, Ram, Weight, Touchscreen, Ips, ppi, SSD, HDD, cpu_ghz, Inches, X_res, Y_res |

### Evaluation

The metrics below were measured with a locked protocol (notebook 05), **before** the final refit on all data:

- a 258-laptop holdout, grouped so that identical configurations never appear in both train and test;
- hyperparameters tuned only by cross-validation on the training portion.

| Metric (holdout) | Value |
|---|---|
| R² on log(price) | 0.916 |
| R² on price | 0.823 |
| Mean absolute error | about ₹8,700 |
| Median absolute percentage error | about 12% |
| Repeated-CV R² on log(price) (5×5, training portion) | 0.897 ± 0.014 |

## Architecture

```text
Streamlit UI (app.py)
      ↓  plain specs dict
predict_laptop_price()          src/predict.py    validation, warnings, loads artifacts once
      ↓
engineer_features()             src/features.py   single source of truth for feature engineering
      ↓
saved pipeline                  artifacts/laptop_price_model.joblib
      ↓
exp(prediction) → estimated price
```

The app contains no feature-engineering logic of its own.

## Limitations

- The dataset is historical (around 2017–2018 hardware), not live retailer data. Newer CPUs and GPUs are extrapolations.
- Discounts, promotions, laptop age and condition are not modelled.
- The exact model and build quality may carry price information that is not in the dataset.
- Errors are larger for very expensive and unusual laptops (workstations, laptops above ₹1 lakh).
  Budget laptops tend to be slightly over-estimated and premium laptops under-estimated.
- The estimate is a reference price, not a guaranteed market price.

The full list is in `known_limitations` in `artifacts/model_metadata.json`.

## Project structure

### Runtime files (needed by the deployed web app)

| Path | Purpose |
|---|---|
| `app.py` | Streamlit website (entrypoint) |
| `.streamlit/config.toml` | Streamlit settings (theme, no usage statistics) |
| `src/predict.py` | Public interface `predict_laptop_price(specs)`, with input validation |
| `src/features.py` | **Single source of truth** for feature engineering |
| `src/app_helpers.py` | UI-only helpers (INR formatting, resolution labels, listed-price comparison) |
| `artifacts/laptop_price_model.joblib` | Fitted pipeline (company grouping, one-hot encoding, XGBoost) |
| `artifacts/model_metadata.json` | Features, company grouping, dropdown values, metrics, limitations, library versions |
| `environment.yml` | Dependency file (conda, Python 3.11, pinned versions that created the model) |

### Research and project-history files (not needed at runtime)

| Path | Purpose |
|---|---|
| `01_data_inspection.ipynb` … `05_final_evaluation.ipynb` | Inspection, cleaning/EDA, model comparison, feature experiments, final evaluation |
| `06_production_pipeline.ipynb` | Sanity check of the production package |
| `src/train.py` | Frozen training script (reproduces the notebook 05 result, then refits and saves artifacts) |
| `tests/` | Automated tests for the model pipeline and the website |
| `laptop_data.csv`, `laptop_clean.csv` | Raw and cleaned dataset |
| `*.png` | Charts from the analysis notebooks |

## Using the prediction interface in Python

```python
from src.predict import predict_laptop_price

result = predict_laptop_price({
    'company': 'Dell', 'type_name': 'Notebook', 'ram_gb': 8, 'weight_kg': 2.2,
    'inches': 15.6, 'resolution_width': 1920, 'resolution_height': 1080,
    'touchscreen': False, 'ips': False,
    'cpu': 'Intel Core i5 8250U 1.6GHz', 'gpu': 'Nvidia GeForce MX150', 'os': 'Windows 10',
    'ssd_gb': 256, 'hdd_gb': 1000,
})
result['predicted_price']          # 47971.18 (unrounded rupees)
result['predicted_price_rounded']  # 48000
result['warnings']                 # notes about unseen categories / out-of-range values
```

`cpu_ghz` is optional and only needed if the `cpu` text has no `...GHz`. Invalid input raises `InvalidSpecError`.
`src.predict.get_input_options()` returns the values seen in training (used for the website's dropdowns).

## Retrain / test

```bash
python -m src.train                          # rebuilds artifacts/ (no hyperparameter tuning)
python -m src.train --metadata-only          # refresh model_metadata.json for the existing model (no training)
python -m unittest discover -s tests -v      # run from the project root
```

## Deployment (Streamlit Community Cloud)

- **Entrypoint:** `app.py`. **Branch:** `main`.
- **Python version:** select **3.11** under *Advanced settings*. Community Cloud defaults to a newer version, and
  the saved model was created and tested with Python 3.11.
- **Dependencies:** `environment.yml` is the only dependency file. Do not add a `requirements.txt`; Community Cloud
  uses only one dependency file.
- **Secrets:** none are needed.
