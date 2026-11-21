"""Show one verbose, step-by-step IntelliWaste calculation using the trained model."""

from __future__ import annotations

import numpy as np
import pandas as pd
import tensorflow as tf

import intelliwaste as iw

SELECTED_BIN = iw.SELECTED_BIN


def main() -> None:
    required = [iw.MODEL_PATH, iw.BIN_HISTORY_PATH, iw.CITIZEN_HISTORY_PATH, iw.PREDICTIONS_PATH, iw.EVALUATION_PATH]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Run train.py and test.py first. Missing: " + ", ".join(missing))

    history = pd.read_csv(iw.BIN_HISTORY_PATH, parse_dates=["timestamp"])
    citizen_history = pd.read_csv(iw.CITIZEN_HISTORY_PATH, parse_dates=["report_timestamp"])
    results = pd.read_csv(iw.PREDICTIONS_PATH)
    evaluation = pd.read_csv(iw.EVALUATION_PATH).iloc[0].to_dict()
    row = results[results["bin_id"] == SELECTED_BIN].iloc[0]
    model = tf.keras.models.load_model(iw.MODEL_PATH)

    prepared = iw.add_features(history.sort_values(["bin_id", "timestamp"]))
    group = prepared[prepared["bin_id"] == SELECTED_BIN].sort_values("timestamp")
    columns = iw.feature_columns()
    window = group[columns].tail(iw.LOOKBACK).to_numpy(dtype=np.float32, copy=True)[None, ...]
    window[:, :, columns.index("fill_level")] /= 100.0
    window[:, :, columns.index("filling_rate")] /= iw.training_rate_scale(history)
    model_fill = float(np.clip(model.predict(window, verbose=0).ravel()[0] * 100, 0, 100))
    citizen_error = row["citizen_error"]
    citizen_reliability = row["citizen_reliability"]
    citizen_weight = row["citizen_weight"]
    if pd.isna(citizen_error):
        citizen_error = "Not applicable"
    if pd.isna(citizen_reliability):
        citizen_reliability = "Not applicable"
    if pd.isna(citizen_weight):
        citizen_weight = 0.0

    print("INTELLIWASTE VERBOSE WALKTHROUGH")
    print("Synthetic data: generated hourly data for 12 bins from January through April 2026 (120 days).")
    print(f"Selected bin: {SELECTED_BIN}")
    print(f"Target timestamp: {row['target_timestamp']}")
    print("\n1. INPUT SEQUENCE")
    print(f"   Shape: {window.shape[1:]} = 24 hours x {window.shape[2]} features")
    print("   Features: previous fill, cyclical time, one-hot zone, filling rate")
    print("\n2. LSTM PREDICTION")
    print(f"   Trained model: {iw.MODEL_PATH}")
    print(f"   LSTM prediction: {model_fill:.2f}%")
    print(f"   Validation MAE: {evaluation['validation_mae']:.3f} percentage points")
    print(f"   Test MAE: {evaluation['test_mae']:.3f} percentage points")
    print(f"   Test RMSE: {evaluation['test_rmse']:.3f} percentage points")
    print("   Fill-status classification metrics (bands: Empty, Low, Half, High, Full):")
    print(f"   Test accuracy: {evaluation['test_accuracy']:.3f}")
    print(f"   Test precision (macro): {evaluation['test_precision_macro']:.3f}")
    print(f"   Test recall (macro): {evaluation['test_recall_macro']:.3f}")
    print(f"   Test F1 (macro): {evaluation['test_f1_macro']:.3f}")
    print("\n3. CONFIDENCE")
    print(f"   Uncertainty = min(1, bin MAE / {iw.VALIDATION_ERROR_LIMIT}) = {row['uncertainty']:.3f}")
    print(f"   Confidence = 1 - uncertainty = {row['confidence']:.1%}")
    print(f"   Decision at {iw.CONFIDENCE_THRESHOLD:.0%}: {row['decision']}")
    print("\n4. CITIZEN METHOD A")
    print(f"   Category: {row['citizen_category']}")
    print(f"   Category fill value: {row['citizen_fill']}")
    print("   The photograph is never converted into a fill percentage.")
    print("\n5. CITIZEN RELIABILITY")
    print(f"   Correct verified reports: {row['correct_reports']}")
    print(f"   Total verified reports: {row['verified_reports']}")
    print(f"   CRS = correct / verified = {row['crs']}")
    print("\n6. TWO-SOURCE RELIABILITY FUSION")
    print(f"   LSTM error = validation MAE / 100 = {row['lstm_error']}")
    print(f"   Citizen error proxy = 1 - CRS = {citizen_error}")
    print(f"   LSTM reliability: {row['lstm_reliability']}")
    print(f"   Citizen reliability: {citizen_reliability}")
    print(f"   LSTM weight: {row['lstm_weight']}")
    print(f"   Citizen weight: {citizen_weight}")
    print(f"   Final fill = {row['final_fill']:.2f}%")
    print("\n7. TIME-TO-FULL")
    print(f"   Historical rate: {row['filling_rate']:.3f} percentage points/hour")
    print(f"   Initial LSTM-only estimate: {row['initial_time_to_full']:.2f} hours ({row['initial_time_to_full_days']:.2f} days)")
    print(f"   Updated fused estimate: {row['updated_time_to_full']:.2f} hours ({row['updated_time_to_full_days']:.2f} days)")
    print("\n8. SEPARATE WASTE CLASSIFIER")
    print("   ResNet50 classifier not configured; it is not used for fill fusion.")
    print("\nPrototype note: all history is synthetic; metrics are not municipal performance claims.")


if __name__ == "__main__":
    main()
