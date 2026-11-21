# IntelliWaste Agent Guide

## Project layout

- `train.py` generates the synthetic January–April 2026 dataset and trains the LSTM.
- `test.py` evaluates the model, calculates regression and fill-status classification metrics, performs citizen fusion, and creates test visualizations.
- `explain.py` prints a verbose walkthrough for the selected bin.
- `intelliwaste.py` contains shared pipeline functions and configuration.
- `data/` contains generated synthetic input CSV files.
- `model/` contains the trained Keras model.
- `results/train/` contains training metrics and data-distribution visualizations.
- `results/test/` contains prediction outputs, evaluation metrics, heatmaps, and test visualizations.
- `PRD/` contains product requirements and should remain documentation-only.

## Run the pipeline

```bash
uv sync
uv run python train.py
uv run python test.py
uv run python explain.py
```

The dataset is synthetic and the metrics demonstrate prototype behavior, not
real-world municipal performance. MAE and RMSE evaluate continuous fill
percentages. Accuracy, precision, recall, F1, and the confusion matrix apply
only to the documented five-band fill-status classification:
`Empty`, `Low`, `Half`, `High`, and `Full`.

Do not use photographs for fill estimation. ResNet50 is classification-only
and must not be included in fill-level fusion.

## Generated files and logs

Generated data, models, and visualizations belong in their existing
directories. Runtime logs belong in `.log/`; `.logger.log` is the only log
file intentionally kept at the project root. Virtual environments, caches,
bytecode, and runtime logs are ignored by Git.
