# Laptop Fair-Price Analyzer

## Live demo

- **App:** https://laptop-fair-price-analyzer.streamlit.app
- **Code:** https://github.com/Anmol161206/laptop-fair-price-analyzer

## Project overview

The Laptop Fair-Price Analyzer estimates a **reference price** (in INR) for a laptop from its hardware
specifications. It uses an XGBoost model trained on a historical dataset of 1,273 laptops.

You choose a laptop's specifications: company, type, RAM, CPU, GPU, storage, display and operating system. The app
returns the price the model would expect for that configuration. You can also enter a seller's listed price to see
how far it is from that estimate.

## Why this project exists

The model answers one question:

> Given a laptop's hardware characteristics, what reference price would the model expect based on historical laptop data?

This is different from searching Amazon or Flipkart:

- it estimates a price **from specifications**, so the exact laptop listing does not need to exist anywhere;
- it can **compare a seller's listed price** with the model's estimate, as a neutral reference point;
- it is **not a live price tracker**: it does not fetch current prices, discounts or stock from any retailer.

Because the training data is historical, the estimate reflects how prices related to hardware in that data. It is
not a statement about today's market.

## Features

- A form with dropdowns limited to values present in the training data (117 CPUs, 105 GPUs, observed RAM and storage sizes, 15 resolutions).
- The estimated reference price, shown with Indian digit grouping (e.g. ₹1,23,400).
- An optional listed-price comparison, worded neutrally (e.g. "The listed price is 6% above this model's estimate"), with no buy/sell advice.
- The model's measured error on unseen laptops, shown next to every estimate.
- Derived characteristics shown for transparency: CPU generation and power class, GPU tier, pixel density.
- Clear warnings for any input the model has not seen before.

## Model and evaluation

| | |
|---|---|
| Model | XGBoost regressor in a scikit-learn pipeline (company grouping, then one-hot encoding, then XGBoost) |
| Target | `log(Price)`; predictions are converted back to rupees with `exp()` |
| Training data | 1,273 laptops (1,303 raw rows minus 1 ARM laptop and 29 exact duplicate rows) |
| Features | 18: TypeName, Company_grouped, Cpu brand, cpu_type, cpu_gen, gpu_tier, os, Ram, Weight, Touchscreen, Ips, ppi, SSD, HDD, cpu_ghz, Inches, X_res, Y_res |

**Why log(Price)?** Raw prices are strongly right-skewed (skewness 1.52). A few very expensive laptops would dominate
the training loss. On the log scale the skewness is −0.18.

These metrics come from a **locked evaluation protocol** (notebook 05), run **before** the final refit on all 1,273 rows:

- a 258-laptop holdout that was never used for model selection;
- grouping so that identical configurations never appear in both train and test.

### Log-price metric

| Metric | Value |
|---|---|
| R² on log(Price), holdout | 0.916 |
| R² on log(Price), repeated group-aware CV (5 folds × 5 repeats, training portion) | 0.897 ± 0.014 |

### Actual rupee-price metrics

| Metric (holdout, predictions converted back to rupees) | Value |
|---|---|
| R² on Price | 0.823 |
| Mean absolute error | about ₹8,700 |
| Median absolute percentage error | about 12% |

R² on rupees is lower than R² on log(Price) because a few expensive laptops have large absolute errors.
Historically, half of the holdout predictions were within about 12% of the actual price.

### Where the model is less reliable

The model is not equally reliable everywhere (holdout error analysis, notebook 05):

- **Ordinary notebooks are predicted most accurately:** about ₹6,000 average error.
- **Unusual and high-end laptops have larger errors.** Workstations average about ₹19,000 error, and laptops above ₹1 lakh about ₹26,000.
- **Predictions are pulled towards the middle.** Budget laptops tend to be slightly over-estimated, and very expensive laptops under-estimated.

## ML workflow

What was actually done in this repository:

1. **Raw-data inspection (notebook 01).** Shape, types, missing values, duplicate check, target statistics.
2. **Cleaning, EDA and feature engineering (notebook 02).**
   - Parsed RAM, weight, resolution, touchscreen/IPS, pixel density (PPI), CPU family, SSD/HDD capacity, GPU brand and OS group.
   - Explored price against every feature and moved the target to a log scale.
3. **Model comparison (notebook 03).**
   - Linear models (Linear, Ridge, Lasso, ElasticNet), KNN, SVR, decision tree, random forest, extra trees, AdaBoost, gradient boosting, HistGradientBoosting and XGBoost.
   - Also voting and stacking ensembles, all tuned with cross-validated grid or random search.
4. **Feature experiments (notebook 04).** Added CPU clock speed, CPU power class (H/U/Y), CPU generation, GPU tier and rare-company grouping, each tested with repeated cross-validation. This added about +0.02 CV R².
5. **Evaluation-integrity audit (notebook 05).**
   - Removed exact duplicates before splitting.
   - Used a **group-aware** holdout and **group-aware** cross-validation, so identical specifications never leak between train and test.
   - Re-tuned the final candidates and checked the repeated-CV spread.
   - Ran error analysis.
   - **Selected XGBoost:** accuracy was statistically tied with a 4-model voting ensemble, while XGBoost is about 20× smaller and 30× faster.
6. **Production refactor (notebook 06, `src/`).**
   - One feature-engineering module shared by training and prediction.
   - A frozen training script that first reproduces the notebook 05 result exactly.
   - A validated prediction interface.
   - Automated tests: 37 in total, covering the model pipeline and the website.
7. **Streamlit deployment.** `app.py` on Streamlit Community Cloud.

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

The app contains no feature-engineering logic of its own. Training, prediction and tests all use `src/features.py`.

## Run locally

There are two environments.

### Run the deployed/runtime app

This uses the same minimal pip dependencies as the deployed app. It needs Python 3.12 or newer, because PyPI's
XGBoost 3.4 requires Python ≥ 3.12.

```bash
python -m pip install -r requirements.txt
streamlit run app.py                     # then open http://localhost:8501
```

On macOS, pip's XGBoost wheel needs the OpenMP runtime: `brew install libomp`.

### Full development/notebook environment

This is the conda environment the notebooks, training and tests were built with (Python 3.11, XGBoost 3.4.2 from
conda-forge).

```bash
conda env create -f environment-dev.yml
conda activate laptop_price
python -m unittest discover -s tests -v  # run the test suite
```

**Why the two environments use different Python/XGBoost versions:**

- The model was saved with XGBoost 3.4.2 from conda-forge. That exact version is not published on PyPI, and PyPI's
  XGBoost 3.3+ requires Python ≥ 3.12.
- The deployed app therefore uses Python 3.12 with `xgboost-cpu` 3.4.1, the official CPU-only build without
  NVIDIA libraries.
- I checked the saved model under that setup: predictions are **bit-identical** to the development environment on
  all 1,273 laptops, and all 37 tests pass in both environments.
- XGBoost logs one generic "model saved by a different version" warning in the server log when the model loads.
  Users never see it.

The prediction interface can also be used directly from Python:

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
result['warnings']                 # notes about unseen categories or out-of-range values
```

`cpu_ghz` is optional and only needed if the `cpu` text contains no `...GHz`. Invalid input raises `InvalidSpecError`.
`get_input_options()` returns the values seen in training, which the website uses for its dropdowns.

To rebuild the artifacts (no hyperparameter tuning): `python -m src.train`. To refresh only the metadata for the
existing model: `python -m src.train --metadata-only`.

**Deploying on Streamlit Community Cloud:**

- Use entrypoint `app.py` on branch `main`, and select **Python 3.12** under *Advanced settings*.
- Community Cloud installs from `requirements.txt`, the only dependency file it recognises in this repository.
  `environment-dev.yml` is not picked up.
- No secrets are needed.

## Repository structure

**Runtime files** (needed by the deployed app):

| Path | Purpose |
|---|---|
| `app.py` | Streamlit website (entrypoint) |
| `.streamlit/config.toml` | Theme; usage statistics disabled |
| `src/predict.py` | Public interface `predict_laptop_price(specs)` with input validation |
| `src/features.py` | Single source of truth for feature engineering |
| `src/app_helpers.py` | UI-only helpers: INR formatting, resolution labels, listed-price comparison |
| `artifacts/laptop_price_model.joblib` | Fitted pipeline (company grouping, one-hot encoding, XGBoost) |
| `artifacts/model_metadata.json` | Features, company grouping, dropdown values, metrics, limitations, library versions |
| `requirements.txt` | Minimal runtime dependencies (pip, used by Streamlit Community Cloud) |

**Research and project-history files** (not needed at runtime):

| Path | Purpose |
|---|---|
| `01_data_inspection.ipynb` | Raw-data inspection |
| `02_preprocessing.ipynb` | Cleaning, EDA, feature engineering, linear baseline |
| `03_model_comparison.ipynb` | 13 tuned models plus voting and stacking ensembles |
| `04_feature_experiments.ipynb` | Feature-engineering experiments |
| `05_final_evaluation.ipynb` | Duplicate/leakage audit, group-aware evaluation, error analysis |
| `06_production_pipeline.ipynb` | Sanity check of the production package |
| `src/train.py` | Frozen training script |
| `environment-dev.yml` | Full conda development environment (Python 3.11, notebooks, training) |
| `tests/` | Model-pipeline and website tests |
| `laptop_data.csv`, `laptop_clean.csv` | Raw and cleaned dataset |
| `*.png` | Charts from the notebooks |

## Limitations

- The dataset is **historical** (hardware of roughly 2016–2018: Intel 6th–8th generation CPUs, GeForce GTX 10-series GPUs). Newer hardware is an extrapolation.
- It is not live retailer data. Current discounts, promotions, laptop age and condition are not modelled.
- The exact model and build quality may carry price information that is not in the dataset.
- Errors are larger for very expensive and unusual laptops, and extreme prices are pulled towards the middle.
- The estimate is a **reference price**, not a guaranteed market price.

The full list is in `known_limitations` in `artifacts/model_metadata.json`.

## Dataset and source

The data comes from the Kaggle dataset
[**Laptop Price** by Muhammet Varlı](https://www.kaggle.com/datasets/muhammetvarl/laptop-price)
(`laptop_price.csv`, 1,303 laptops, prices in euros; first published November 2020).

`laptop_data.csv` in this repository is a widely circulated derivative of that file. I checked it against a public
copy of `laptop_price.csv`:

- all 1,303 rows match in the same order across all 10 specification columns;
- `laptop_ID` and `Product` were removed;
- every price equals `Price_euros × 53.28`.

The rupee prices are therefore a fixed conversion of the original euro prices, not independently collected Indian
prices. The Kaggle page does not name an upstream source or give explicit license terms, and who first made the
rupee-converted version could not be confirmed. Credit for the data belongs to the original dataset author.

## Future improvements

- Prediction intervals (e.g. quantile models) to show a calibrated price range, not only a point estimate.
- More recent laptop data, so estimates reflect current hardware and prices.
- Linked dropdowns in the app, for example CPUs filtered by manufacturer, to avoid unrealistic combinations.
- Richer hardware features, such as CPU/GPU benchmark scores in place of name-based tiers.
