"""Evaluate the trained model, fuse citizen reports, and create a dashboard image."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import tensorflow as tf

import intelliwaste as iw


def make_results(history: pd.DataFrame, citizen_history: pd.DataFrame, model: Any) -> tuple[pd.DataFrame, dict[str, float]]:
    train, validation, test, _ = iw.build_splits(history)
    predictions: dict[str, np.ndarray] = {}
    for name, split in (("training", train), ("validation", validation), ("test", test)):
        predictions[name] = np.clip(model.predict(split.x, verbose=0).ravel() * 100, 0, 100)
    report: dict[str, float] = {}
    for name, split in (("training", train), ("validation", validation), ("test", test)):
        mae, rmse = iw.metrics(split.y * 100, predictions[name])
        report[f"{name}_mae"] = mae
        report[f"{name}_rmse"] = rmse
        class_report = iw.classification_metrics(split.y * 100, predictions[name])
        for metric_name, metric_value in class_report.items():
            if metric_name != "confusion_matrix":
                report[f"{name}_{metric_name}"] = metric_value

    validation_errors = np.abs(validation.y * 100 - predictions["validation"])
    per_bin_mae = pd.DataFrame({
        "bin_id": validation.bins, "error": validation_errors
    }).groupby("bin_id")["error"].mean()
    overall_mae = report["validation_mae"]
    prepared = iw.add_features(history.sort_values(["bin_id", "timestamp"]))
    rows: list[dict[str, Any]] = []
    columns = iw.feature_columns()
    rate_scale = iw.training_rate_scale(history)
    for bin_id in iw.BIN_ZONES:
        group = prepared[prepared["bin_id"] == bin_id].sort_values("timestamp")
        window = group[columns].tail(iw.LOOKBACK).to_numpy(
            dtype=np.float32, copy=True
        )[None, ...]
        window[:, :, columns.index("fill_level")] /= 100.0
        window[:, :, columns.index("filling_rate")] /= rate_scale
        lstm_fill = float(np.clip(model.predict(window, verbose=0).ravel()[0] * 100, 0, 100))
        bin_mae = float(per_bin_mae.get(bin_id, overall_mae))
        uncertainty = min(1.0, bin_mae / iw.VALIDATION_ERROR_LIMIT)
        confidence = 1.0 - uncertainty
        required = confidence < iw.CONFIDENCE_THRESHOLD
        citizen_fill = float(iw.CATEGORIES[iw.DEFAULT_CITIZEN_CATEGORY]) if required else None
        correct, verified, crs = (
            iw.citizen_reliability(citizen_history, iw.DEFAULT_CITIZEN_ID)
            if required else (0, 0, None)
        )
        fused = iw.fusion(lstm_fill, citizen_fill, crs, overall_mae)
        rate = iw.recent_positive_rate(history, bin_id)
        initial = iw.time_to_full(lstm_fill, rate)
        updated = iw.time_to_full(float(fused["final_fill"]), rate)
        rows.append({
            "bin_id": bin_id, "target_timestamp": group["timestamp"].max() + pd.Timedelta(hours=1),
            "lstm_fill": lstm_fill, "bin_mae": bin_mae, "uncertainty": uncertainty,
            "confidence": confidence, "decision": "Verification Required" if required else "Not Required",
            "citizen_category": iw.DEFAULT_CITIZEN_CATEGORY if required else "Not Required",
            "citizen_fill": citizen_fill if citizen_fill is not None else "Not Required",
            "correct_reports": correct, "verified_reports": verified,
            "crs": crs if crs is not None else "Not Required", **fused,
            "final_fill": float(fused["final_fill"]), "filling_rate": rate,
            "initial_time_to_full": initial if initial is not None else "Not estimable",
            "updated_time_to_full": updated if updated is not None else "Not estimable",
            "initial_time_to_full_days": initial / 24 if initial is not None else "Not estimable",
            "updated_time_to_full_days": updated / 24 if updated is not None else "Not estimable",
        })
    return pd.DataFrame(rows), report


def save_visualizations(results: pd.DataFrame, history: pd.DataFrame, model: Any) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.ticker import PercentFormatter

    figure, axis = plt.subplots(figsize=(12, 6))
    x = np.arange(len(results))
    actual_last = history.sort_values("timestamp").groupby("bin_id").tail(1).set_index("bin_id")
    actual_values = [float(actual_last.loc[bin_id, "fill_level"]) for bin_id in results["bin_id"]]
    axis.bar(x - 0.28, actual_values, width=0.25, label="Latest actual")
    axis.bar(x, results["lstm_fill"], width=0.25, label="LSTM prediction")
    axis.bar(x + 0.28, results["final_fill"], width=0.25, label="Final estimate")
    axis.set_xticks(x, results["bin_id"], rotation=45)
    axis.set_ylim(0, 100)
    axis.set_ylabel("Fill level (%)")
    axis.set_title("IntelliWaste: actual vs predicted vs final fill")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(iw.VISUALIZATION_PATH, dpi=150)
    plt.close(figure)

    selected_bins = ["BIN001", "BIN004", "BIN007", "BIN010"]
    figure, axis = plt.subplots(figsize=(12, 6))
    for bin_id in selected_bins:
        sample = history[history["bin_id"] == bin_id].sort_values("timestamp")
        axis.plot(sample["timestamp"], sample["fill_level"], label=bin_id, linewidth=1.4)
    axis.set_title("Synthetic fill history with collection resets")
    axis.set_ylabel("Fill level (%)")
    axis.yaxis.set_major_formatter(PercentFormatter(100))
    axis.legend(ncol=4)
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(iw.HISTORY_VISUALIZATION_PATH, dpi=150)
    plt.close(figure)

    _, validation, _, _ = iw.build_splits(history)
    validation_actual = validation.y * 100
    validation_predicted = np.clip(model.predict(validation.x, verbose=0).ravel() * 100, 0, 100)
    residuals = validation_actual - validation_predicted
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(validation_actual, validation_predicted, alpha=0.25, s=10)
    axes[0].plot([0, 100], [0, 100], "k--", linewidth=1)
    axes[0].set_xlabel("Actual fill (%)")
    axes[0].set_ylabel("Predicted fill (%)")
    axes[0].set_title("Validation: actual vs predicted")
    axes[1].hist(residuals, bins=30, color="#4c78a8", edgecolor="white")
    axes[1].axvline(0, color="black", linestyle="--", linewidth=1)
    axes[1].set_xlabel("Residual (actual - predicted), percentage points")
    axes[1].set_title("Validation residual distribution")
    figure.tight_layout()
    figure.savefig(iw.DIAGNOSTICS_VISUALIZATION_PATH, dpi=150)
    plt.close(figure)

    figure, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    axes[0].bar(results["bin_id"], results["confidence"] * 100, color="#59a14f")
    axes[0].axhline(iw.CONFIDENCE_THRESHOLD * 100, color="red", linestyle="--", label="80% threshold")
    axes[0].set_ylabel("Confidence (%)")
    axes[0].set_ylim(0, 100)
    axes[0].legend()
    axes[1].bar(results["bin_id"], pd.to_numeric(results["updated_time_to_full_days"], errors="coerce"), color="#f28e2b")
    axes[1].set_ylabel("Estimated time-to-full (days)")
    axes[1].set_xlabel("Bin")
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].grid(axis="y", alpha=0.25)
    figure.suptitle("Confidence and estimated time-to-full")
    figure.tight_layout()
    figure.savefig(iw.CONFIDENCE_VISUALIZATION_PATH, dpi=150)
    plt.close(figure)

    # For a regression model, a testing matrix is more accurate than a
    # classification confusion matrix: each cell is an error metric.
    splits = {
        "Training": iw.build_splits(history)[0],
        "Validation": iw.build_splits(history)[1],
        "Test": iw.build_splits(history)[2],
    }
    metric_rows = []
    for split_name, split in splits.items():
        split_pred = np.clip(model.predict(split.x, verbose=0).ravel() * 100, 0, 100)
        for bin_id in iw.BIN_ZONES:
            mask = split.bins == bin_id
            mae, rmse = iw.metrics(split.y[mask] * 100, split_pred[mask])
            metric_rows.append({"bin_id": bin_id, "split": split_name, "MAE": mae, "RMSE": rmse})
    metric_frame = pd.DataFrame(metric_rows)
    mae_matrix = metric_frame.pivot(index="bin_id", columns="split", values="MAE")[
        ["Training", "Validation", "Test"]
    ]
    figure, axis = plt.subplots(figsize=(8, 7))
    sns.heatmap(mae_matrix, annot=True, fmt=".2f", cmap="YlOrRd", cbar_kws={"label": "MAE (percentage points)"}, ax=axis)
    axis.set_title("Testing matrix: MAE by bin and chronological split")
    axis.set_xlabel("Chronological split")
    axis.set_ylabel("Bin ID")
    figure.tight_layout()
    figure.savefig(iw.METRICS_HEATMAP_PATH, dpi=150)
    plt.close(figure)

    test_split = splits["Test"]
    test_pred = np.clip(model.predict(test_split.x, verbose=0).ravel() * 100, 0, 100)
    test_class_report = iw.classification_metrics(test_split.y * 100, test_pred)
    figure, axis = plt.subplots(figsize=(8, 7))
    sns.heatmap(
        test_class_report["confusion_matrix"],
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=iw.FILL_STATUS_LABELS,
        yticklabels=iw.FILL_STATUS_LABELS,
        cbar_kws={"label": "Number of test observations"},
        ax=axis,
    )
    axis.set_title("Test fill-status confusion matrix")
    axis.set_xlabel("Predicted fill status")
    axis.set_ylabel("Actual fill status")
    figure.tight_layout()
    figure.savefig(iw.CONFUSION_MATRIX_PATH, dpi=150)
    plt.close(figure)

    # Test error by bin and hour makes time-of-day weaknesses visible.
    target_hours = []
    for _, group in history.sort_values(["bin_id", "timestamp"]).groupby("bin_id", sort=True):
        n = len(group)
        test_start = int(n * 0.85)
        target_hours.extend(group.iloc[test_start:]["hour"].to_numpy())
    test_hours = np.asarray(target_hours[: len(test_pred)])
    error_frame = pd.DataFrame({
        "bin_id": test_split.bins,
        "hour": test_hours,
        "absolute_error": np.abs(test_split.y * 100 - test_pred),
    })
    error_matrix = error_frame.pivot_table(
        index="bin_id", columns="hour", values="absolute_error", aggfunc="mean"
    ).reindex(index=list(iw.BIN_ZONES), columns=list(range(24)))
    figure, axis = plt.subplots(figsize=(14, 7))
    sns.heatmap(error_matrix, cmap="mako", annot=False, cbar_kws={"label": "Absolute error (percentage points)"}, ax=axis)
    axis.set_title("Test error heatmap by bin and hour of day")
    axis.set_xlabel("Target hour of day")
    axis.set_ylabel("Bin ID")
    figure.tight_layout()
    figure.savefig(iw.ERROR_HEATMAP_PATH, dpi=150)
    plt.close(figure)


def main() -> None:
    if not iw.MODEL_PATH.exists():
        raise FileNotFoundError("Trained model missing. Run: uv run python train.py")
    history = pd.read_csv(iw.BIN_HISTORY_PATH, parse_dates=["timestamp"])
    citizen_history = pd.read_csv(iw.CITIZEN_HISTORY_PATH, parse_dates=["report_timestamp"])
    model = tf.keras.models.load_model(iw.MODEL_PATH)
    results, report = make_results(history, citizen_history, model)
    results.to_csv(iw.PREDICTIONS_PATH, index=False)
    pd.DataFrame([report]).to_csv(iw.EVALUATION_PATH, index=False)
    save_visualizations(results, history, model)
    print("TESTING COMPLETE")
    print(results[[
        "bin_id", "lstm_fill", "confidence", "final_fill",
        "updated_time_to_full", "updated_time_to_full_days"
    ]].to_string(index=False))
    print(f"Test accuracy (fill-status bands): {report['test_accuracy']:.3f}")
    print(f"Test precision (macro): {report['test_precision_macro']:.3f}")
    print(f"Test recall (macro): {report['test_recall_macro']:.3f}")
    print(f"Test F1 (macro): {report['test_f1_macro']:.3f}")
    print(f"Predictions: {iw.PREDICTIONS_PATH}")
    print(f"Visualization: {iw.VISUALIZATION_PATH}")
    print(
        "Additional visualizations: "
        f"{iw.HISTORY_VISUALIZATION_PATH}, {iw.DIAGNOSTICS_VISUALIZATION_PATH}, "
        f"{iw.CONFIDENCE_VISUALIZATION_PATH}, {iw.METRICS_HEATMAP_PATH}, "
        f"{iw.ERROR_HEATMAP_PATH}"
    )


if __name__ == "__main__":
    main()
