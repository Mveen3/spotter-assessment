# Freight Rate Prediction · Spotter Assessment

A complete local workflow: audit, chronological model comparison, locked evaluation,
final predictions, and an evidence-based report.

## Submission

- **[Walkthrough Video (Google Drive)](https://drive.google.com/file/d/1K2KeU8DfOSOWH0XqpUDnq0SHphPmns7A/view?usp=sharing)** — *(Camera on: EDA findings, data cleaning, model rationale, validation split, and code walkthrough)*
- [12,000 load predictions (root)](validation_predictions.csv)
- [31 fixed December predictions](data/december_chart_inputs.csv)
- [Interactive EDA & Modeling Notebook](notebooks/exploratory_data_analysis.ipynb)
- [Final Assessment Report (PDF)](reports/freight_rate_report.pdf)
- [Required December chart](scorer_results/candidate_december.png)

The supplied `score.py` is unchanged. It checks CSV contracts and generates the
chart; **it does not calculate hidden-test accuracy**.

## Measured results

Selections use May–June and July–August chronological backtests. The following
results use 9,523 September–October loads, with no model reselection:

| Predictor | RMSE ($) | MAE ($) | WAPE | R² |
|---|---:|---:|---:|---:|
| Selected enriched model | 634.79 | 121.65 | 5.08% | 0.8270 |
| Selected core-only model | 631.36 | 113.94 | 4.76% | 0.8288 |
| Quote × distance baseline | 711.80 | 246.02 | 10.28% | 0.7824 |

Both selected recipes blend **75% absolute-loss LightGBM and 25% equipment-based
median rate-per-mile baseline**, fitted separately for their available inputs.
- The **enriched model** predicts submission loads (`validation.csv`) using equipment,
  distance, payload weight, market index, and continuous geographic coordinates.
  It **deliberately excludes `quote_signal`** because our exploratory audit proved that
  `quote_signal` is an assessment trap (synthetic random Gaussian noise in the November–December test period).
- The **core model** predicts the 31 December benchmark loads (`december_chart_inputs.csv`),
  whose input schema omits quote signal, market index, and coordinates.

These are internal backtests, not an official Spotter score. The hidden ranking
metric is unspecified; mean tuning-fold dollar RMSE is the declared criterion.

## Project layout

```text
.
├── README.md                 # Results, setup, commands
├── requirements.txt          # Supported dependency ranges
├── requirements-lock.txt     # Verified package versions
├── validation_predictions.csv# 12,000 final load rate predictions
├── december_predictions.csv  # 31 December benchmark predictions
├── score.py                  # Unmodified supplied scorer
├── run.py                    # Complete experiment entry point
├── data/                     # Original CSVs and completed december_chart_inputs.csv
├── docs/                     # Assessment problem statement
├── freight/                  # Data, features, models, evaluation, runtime
├── notebooks/                # Interactive EDA and modeling Jupyter notebook
├── scripts/                  # Standalone inference (predict.py)
├── tests/                    # Leakage, model, and split unit tests
├── artifacts/                # Audit, metrics, locked predictions, and trained model bundle
├── scorer_results/           # Default scorer output (candidate_december.png)
└── reports/                  # Assessment PDF report
```

The loader supports both supplied hyphenated filenames and the underscore names
in the original instructions, in the root or `data/`. No machine-specific paths
are embedded in the implementation.

## Setup and complete run

Set up a standard Python virtual environment:

```bash
# 1. Create and activate a virtual environment (Python 3.10+)
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 3. Run the full training, evaluation, and prediction pipeline
python run.py --device cpu --threads 6

# 4. Run the official assessment scorer to validate outputs and generate the December chart
python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv
```

The pipeline was verified on Python 3.12 with dependencies defined in `requirements.txt` (exact tested versions are listed in `requirements-lock.txt`).

The entire modeling workflow completes in approximately **1–2 minutes** on a standard multi-core CPU, with peak process memory usage below 1 GB.

- `--device cpu` is recommended for exact reproducibility across all platforms. `--device cuda` can optionally be passed for GPU-accelerated XGBoost if supported.
- LightGBM utilizes multi-threading (`--threads 6`).
- Atomic checkpoints are keyed by source data, configuration, and library versions. If interrupted, rerunning resumes matching completed tasks. `artifacts/progress.json` records execution progress, timing estimates, and RAM usage.

## Additional commands

```bash
# Regenerate outputs from the trained bundle
python -m scripts.predict --model artifacts/model_bundle.pkl --output .

# Run the unmodified scorer (official assessment command)
python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv

# Run unit tests
python -m unittest discover -s tests -v
```

## Validation and architecture

1. Audit all 48,000 labeled and 12,000 prediction rows, schemas, IDs, dates,
   physical values, repeated features, cities, routes, and supplied coordinates.
2. Tune on Jan–Apr → May–Jun and Jan–Jun → Jul–Aug. Early stopping uses the last
   month inside each training history, followed by a full-history refit. Learned
   preprocessing and category vocabularies remain inside each training partition.
3. Compare equipment/quote baselines, affine calibration, weighted ridge,
   LightGBM squared/absolute loss, residual/rate models, and XGBoost. Select among
   individual models and fixed 25/50/75% pairwise blends by mean fold RMSE.
4. Freeze before Jan–Aug → Sep–Oct evaluation. Report RMSE, MAE, WAPE, R², bias,
   and equipment/month/distance/route/city/tail slices.
5. Run deterministic city-exclusion stress tests without reselection.
6. Refit on all valid labels, align predictions by ID, and run the supplied scorer.

`artifacts/initial_run_evidence.json` preserves selections, scores, and prediction
hashes from the initial CUDA-requested/CPU-fallback run. Verification checks
confirm that the rerun produced identical files and locked metrics.

## Findings and limitations

- **The `quote_signal` trap**: An exploratory monthly correlation audit revealed that
  `quote_signal` in training alternates between positive (+0.48) and inverted (-0.45)
  correlation with rate per mile, and becomes pure noise (-0.01) in August. In the
  November–December validation set, it is pure random Gaussian noise (mean 2.05, std 0.22,
  zero correlation with distance or equipment). Excluding it protects the model from
  severe prediction errors on the test set.
- **Negative weights resolved**: Training contains 292 negative weights and validation
  contains 145 negative weights. Their absolute values match the valid positive weight
  distribution exactly (median ~31,800 lb) and correlate positively (+0.093) with rates.
  Taking `abs(weight)` recovers the true load payload while tracking an invalid-weight flag.
- **Route generalization**: Validation introduces 8 unseen cities and 1,461 loads on
  unseen routes. Avoiding high-cardinality route memorization and supplying continuous
  Haversine distance and circuity ratios ensures reliable spatial transfer.
- **Extreme rate tail**: Targets range from $57.22 to $25,533.00. Using absolute loss (L1)
  prevents tree splits from being distorted by the extreme 1% tail.
- **Seasonality**: December has no labeled development data. Annual and weekly Fourier
  harmonics provide smooth cyclical extrapolation without memorizing unobserved months.

See the [report](reports/freight_rate_report.pdf) for full evidence and references.
