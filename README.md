# IntelliWaste

IntelliWaste is a small, reproducible prototype for hourly waste-bin fill
prediction. It generates synthetic history for 12 bins, trains an LSTM,
requests a citizen category when validation-based confidence is below 80%,
combines only the LSTM and citizen estimates using the specified reliability
formula, and reports an approximate time-to-full.

The optional ResNet50 interface is deliberately separate from fill estimation.
It returns `ResNet50 classifier not configured` unless both a photograph and a
fine-tuned three-class checkpoint are supplied. Photo data and ResNet50
probabilities are never used in fill-level fusion.

> **Prototype limitation:** the dataset is synthetic and the metrics
> demonstrate prototype behavior, not real-world municipal performance.
> Citizen reliability is an error-rate proxy from previously verified reports,
> not a calibrated probability.

## Run

Install dependencies with `uv`:

```bash
uv sync
```

Run the three stages:

```bash
uv run python train.py
uv run python test.py
uv run python explain.py
```

Generate the local experimental report and its traceability files:

```bash
uv run python doc/build_report.py
```

This creates `doc/IntelliWaste_Experimental_Results_Report.pdf`,
`doc/experiment_summary.json`, `doc/test_predictions.csv`,
`doc/training_history.csv`, and `doc/generated_figures/`. The complete
`doc/` directory is intentionally ignored by Git because it contains
generated report artifacts.

`train.py` generates the synthetic data and trains the LSTM. `test.py` loads
the trained model, evaluates predictions, performs citizen fusion, and saves a
dashboard plus validation, history, and confidence diagnostic images.
`explain.py` loads the trained artifacts and prints a verbose
walkthrough for `BIN001`.

The scripts are unattended by default. Change `SELECTED_BIN`,
`DEFAULT_CITIZEN_CATEGORY`, `PHOTO_PATH`, or `RESNET_CHECKPOINT` near the top
of `intelliwaste.py` when needed.

## Outputs

The project keeps generated artifacts separated:

```text
model/
└── intelliwaste_lstm.keras
data/
├── synthetic_bin_history.csv
└── synthetic_citizen_history.csv
results/
├── train/
│   ├── training_history.csv
│   ├── training_curve.png
│   ├── data_distribution.png
│   ├── bin_fill_history.png
│   └── collection_timeline.png
└── test/
    ├── bin_prediction_results.csv
    ├── lstm_evaluation_results.csv
    ├── fill_prediction_dashboard.png
    ├── validation_diagnostics.png
    ├── confidence_time_to_full.png
    ├── testing_metrics_heatmap.png
    ├── test_error_heatmap.png
    └── fill_status_confusion_matrix.png
```

Training configures TensorFlow's intra-op and inter-op CPU thread pools using
the available CPU cores. The synthetic generator deliberately includes
collection cycles, temporal demand patterns, correlated daily variation, and
small random events so bins do not all saturate at 100%. The synthetic period
is January 1 through April 30, 2026, sampled hourly. Time-to-full is stored in
hours and days; charts label the displayed unit explicitly.

`PRD/` remains documentation-only. No physical sensors, photo-based fill
estimation, overflow probability, route optimization, or extra fill source is
implemented.
