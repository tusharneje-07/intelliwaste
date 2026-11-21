"""IntelliWaste: synthetic fill prediction, citizen verification, and fusion.

This is a reproducible prototype. The generated data and model metrics are
synthetic demonstrations, not evidence of real-world municipal performance.
"""

from __future__ import annotations

import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SEED = 42
LOOKBACK = 24
HORIZON = 1
HISTORY_DAYS = 120
EPOCHS = 50
BATCH_SIZE = 32
CPU_THREADS = max(2, min(8, os.cpu_count() or 2))
INTER_OP_THREADS = max(1, CPU_THREADS // 2)
VALIDATION_ERROR_LIMIT = 10.0
CONFIDENCE_THRESHOLD = 0.80
RELIABILITY_EPSILON = 0.001
SELECTED_BIN = "BIN001"
DEFAULT_CITIZEN_CATEGORY = "Around 50–75% full"
DEFAULT_CITIZEN_ID = "citizen_demo_001"
PHOTO_PATH: str | None = None
RESNET_CHECKPOINT: str | None = None

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
TRAIN_RESULTS_DIR = RESULTS_DIR / "train"
TEST_RESULTS_DIR = RESULTS_DIR / "test"
for output_dir in (MODEL_DIR, DATA_DIR, TRAIN_RESULTS_DIR, TEST_RESULTS_DIR):
    output_dir.mkdir(exist_ok=True)
BIN_HISTORY_PATH = DATA_DIR / "synthetic_bin_history.csv"
CITIZEN_HISTORY_PATH = DATA_DIR / "synthetic_citizen_history.csv"
MODEL_PATH = MODEL_DIR / "intelliwaste_lstm.keras"
PREDICTIONS_PATH = TEST_RESULTS_DIR / "bin_prediction_results.csv"
EVALUATION_PATH = TEST_RESULTS_DIR / "lstm_evaluation_results.csv"
TRAINING_HISTORY_PATH = TRAIN_RESULTS_DIR / "training_history.csv"
VISUALIZATION_PATH = TEST_RESULTS_DIR / "fill_prediction_dashboard.png"
HISTORY_VISUALIZATION_PATH = TRAIN_RESULTS_DIR / "bin_fill_history.png"
DIAGNOSTICS_VISUALIZATION_PATH = TEST_RESULTS_DIR / "validation_diagnostics.png"
CONFIDENCE_VISUALIZATION_PATH = TEST_RESULTS_DIR / "confidence_time_to_full.png"
METRICS_HEATMAP_PATH = TEST_RESULTS_DIR / "testing_metrics_heatmap.png"
ERROR_HEATMAP_PATH = TEST_RESULTS_DIR / "test_error_heatmap.png"
CONFUSION_MATRIX_PATH = TEST_RESULTS_DIR / "fill_status_confusion_matrix.png"
DATA_DISTRIBUTION_PATH = TRAIN_RESULTS_DIR / "data_distribution.png"
COLLECTION_TIMELINE_PATH = TRAIN_RESULTS_DIR / "collection_timeline.png"
TRAINING_CURVE_PATH = TRAIN_RESULTS_DIR / "training_curve.png"

BIN_ZONES = {
    "BIN001": "Residential", "BIN002": "Residential", "BIN003": "Residential",
    "BIN004": "Commercial", "BIN005": "Commercial", "BIN006": "Commercial",
    "BIN007": "Market", "BIN008": "Market", "BIN009": "Market",
    "BIN010": "Institutional", "BIN011": "Institutional", "BIN012": "Institutional",
}
ZONE_TYPES = ["Residential", "Commercial", "Market", "Institutional"]
FILL_STATUS_LABELS = ["Empty", "Low", "Half", "High", "Full"]
FILL_STATUS_CUTS = np.array([12.5, 37.5, 62.5, 87.5])
CATEGORIES = {
    "Empty": 0.0,
    "Low": 25.0,
    "Half": 50.0,
    "High": 75.0,
    "Full": 100.0,
    "Around 50–75% full": (50.0 + 75.0) / 2.0,
}


@dataclass
class SplitData:
    x: np.ndarray
    y: np.ndarray
    bins: np.ndarray


def generate_bin_history() -> pd.DataFrame:
    """Generate dependent hourly observations with realistic non-saturated cycles."""
    rng = np.random.default_rng(SEED)
    timestamps = pd.date_range("2026-01-01", periods=HISTORY_DAYS * 24, freq="h")
    # Rates are percentage points/hour. Collection cadence and rates vary by zone
    # so bins do not all converge to the 100% clipping boundary.
    profiles = {
        "Residential": (0.18, 168),
        "Commercial": (0.31, 96),
        "Market": (0.39, 72),
        "Institutional": (0.23, 144),
    }
    rows: list[dict[str, Any]] = []
    for _, (bin_id, zone) in enumerate(BIN_ZONES.items()):
        fill = float(rng.uniform(8, 28))
        base_rate, base_period = profiles[zone]
        rate_scale = base_rate * rng.uniform(0.78, 1.24)
        collection_period = int(base_period + rng.choice([-24, 0, 24]))
        collection_day_offset = int(rng.integers(0, max(1, collection_period // 24))) * 24
        collection_hour = int(rng.choice([3, 4, 5]))
        collection_offset = collection_day_offset + collection_hour
        daily_shocks = rng.normal(1.0, 0.10, HISTORY_DAYS)
        daily_shocks = np.clip(daily_shocks, 0.72, 1.32)
        previous_delta = 0.0
        for index, timestamp in enumerate(timestamps):
            collection = bool(
                index >= collection_period
                and (index - collection_offset) % collection_period == 0
                and timestamp.hour in (3, 4, 5)
            )
            previous = fill
            if collection:
                fill = float(rng.uniform(5, 18))
                observed_rate = 0.0
            else:
                morning_rush = 1.0 + 0.30 * np.exp(-((timestamp.hour - 8) / 2.5) ** 2)
                evening_rush = 1.0 + 0.24 * np.exp(-((timestamp.hour - 19) / 3.0) ** 2)
                weekday_factor = 1.12 if timestamp.day_of_week < 5 else 0.78
                fortnight_factor = 1.0 + 0.12 * np.sin(2 * np.pi * index / (24 * 14) + collection_offset)
                event_spike = rng.uniform(0.25, 0.85) if rng.random() < 0.012 else 0.0
                noise = rng.normal(0, 0.085)
                delta = (
                    rate_scale * morning_rush * evening_rush * weekday_factor
                    * daily_shocks[index // 24] * fortnight_factor
                    + 0.12 * previous_delta + noise + event_spike
                )
                fill = float(np.clip(fill + max(-0.08, delta), 0, 100))
                observed_rate = max(0.0, fill - previous)
                previous_delta = observed_rate
            rows.append({
                "bin_id": bin_id,
                "timestamp": timestamp,
                "fill_level": round(fill, 4),
                "hour": timestamp.hour,
                "day_of_week": timestamp.day_of_week,
                "month": timestamp.month,
                "zone_type": zone,
                "filling_rate": round(observed_rate, 4),
                "collection_event": collection,
            })
    return pd.DataFrame(rows)


def feature_columns() -> list[str]:
    return [
        "fill_level", "hour_sin", "hour_cos", "weekday_sin", "weekday_cos",
        "month_sin", "month_cos", *[f"zone_{zone}" for zone in ZONE_TYPES],
        "filling_rate",
    ]


def training_rate_scale(frame: pd.DataFrame) -> float:
    rates = [
        float(group["filling_rate"].iloc[: int(len(group) * 0.70)].max())
        for _, group in frame.groupby("bin_id", sort=True)
    ]
    return max(max(rates, default=0.0), 1.0)


def add_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["hour_sin"] = np.sin(2 * np.pi * result["hour"] / 24)
    result["hour_cos"] = np.cos(2 * np.pi * result["hour"] / 24)
    result["weekday_sin"] = np.sin(2 * np.pi * result["day_of_week"] / 7)
    result["weekday_cos"] = np.cos(2 * np.pi * result["day_of_week"] / 7)
    result["month_sin"] = np.sin(2 * np.pi * (result["month"] - 1) / 12)
    result["month_cos"] = np.cos(2 * np.pi * (result["month"] - 1) / 12)
    for zone in ZONE_TYPES:
        result[f"zone_{zone}"] = (result["zone_type"] == zone).astype(float)
    return result


def build_splits(frame: pd.DataFrame) -> tuple[SplitData, SplitData, SplitData, dict[str, int]]:
    """Build chronological windows; split assignment is based on target time."""
    prepared = add_features(frame.sort_values(["bin_id", "timestamp"]))
    columns = feature_columns()
    # The only learned preprocessing values are fitted from each bin's
    # chronological training region, never from validation or test rows.
    train_rate_scale = training_rate_scale(prepared)
    train_rows: list[tuple[np.ndarray, float, str]] = []
    val_rows: list[tuple[np.ndarray, float, str]] = []
    test_rows: list[tuple[np.ndarray, float, str]] = []
    boundaries: dict[str, int] = {}
    for bin_id, group in prepared.groupby("bin_id", sort=True):
        values = group.reset_index(drop=True)
        n = len(values)
        train_end = int(n * 0.70)
        val_end = int(n * 0.85)
        boundaries[bin_id] = train_end
        matrix = values[columns].to_numpy(dtype=np.float32)
        matrix[:, columns.index("fill_level")] /= 100.0
        matrix[:, columns.index("filling_rate")] /= train_rate_scale
        target = values["fill_level"].to_numpy(dtype=np.float32) / 100.0
        for target_index in range(LOOKBACK, n):
            item = (matrix[target_index - LOOKBACK:target_index], float(target[target_index]), bin_id)
            if target_index < train_end:
                train_rows.append(item)
            elif target_index < val_end:
                val_rows.append(item)
            else:
                test_rows.append(item)

    def convert(rows: list[tuple[np.ndarray, float, str]]) -> SplitData:
        return SplitData(
            np.asarray([row[0] for row in rows], dtype=np.float32),
            np.asarray([row[1] for row in rows], dtype=np.float32),
            np.asarray([row[2] for row in rows]),
        )
    return convert(train_rows), convert(val_rows), convert(test_rows), boundaries


def build_model(input_shape: tuple[int, int]):
    import tensorflow as tf
    regularizer = tf.keras.regularizers.l2(1e-4)
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.LSTM(64, return_sequences=True, kernel_regularizer=regularizer),
        tf.keras.layers.Dropout(0.30),
        tf.keras.layers.LSTM(32, kernel_regularizer=regularizer),
        tf.keras.layers.Dense(16, activation="relu", kernel_regularizer=regularizer),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="mse")
    return model


def metrics(actual: np.ndarray, predicted: np.ndarray) -> tuple[float, float]:
    errors = actual - predicted
    return float(np.mean(np.abs(errors))), float(np.sqrt(np.mean(errors ** 2)))


def fill_status(values: np.ndarray) -> np.ndarray:
    """Map continuous percentages to explicit status bands for classification metrics."""
    return np.digitize(np.asarray(values), FILL_STATUS_CUTS, right=False)


def classification_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, Any]:
    """Calculate classification metrics on five documented fill-status bands."""
    actual_classes = fill_status(actual)
    predicted_classes = fill_status(predicted)
    size = len(FILL_STATUS_LABELS)
    confusion = np.zeros((size, size), dtype=int)
    for actual_class, predicted_class in zip(actual_classes, predicted_classes):
        confusion[int(actual_class), int(predicted_class)] += 1
    true_positive = np.diag(confusion).astype(float)
    precision = true_positive / np.maximum(confusion.sum(axis=0), 1)
    recall = true_positive / np.maximum(confusion.sum(axis=1), 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    support = confusion.sum(axis=1).astype(float)
    total = max(float(support.sum()), 1.0)
    return {
        "accuracy": float(true_positive.sum() / total),
        "precision_macro": float(precision.mean()),
        "recall_macro": float(recall.mean()),
        "f1_macro": float(f1.mean()),
        "precision_weighted": float(np.sum(precision * support) / total),
        "recall_weighted": float(np.sum(recall * support) / total),
        "f1_weighted": float(np.sum(f1 * support) / total),
        "confusion_matrix": confusion,
    }


def train_and_evaluate(train: SplitData, validation: SplitData, test: SplitData):
    import tensorflow as tf
    # TensorFlow uses these CPU worker pools for parallel matrix operations.
    tf.config.threading.set_intra_op_parallelism_threads(CPU_THREADS)
    tf.config.threading.set_inter_op_parallelism_threads(INTER_OP_THREADS)
    tf.keras.utils.set_random_seed(SEED)
    model = build_model((LOOKBACK, train.x.shape[2]))
    callback = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=6, restore_best_weights=True
    )
    fit_history = model.fit(
        train.x, train.y, validation_data=(validation.x, validation.y),
        epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[callback], verbose=0,
    )
    predictions: dict[str, np.ndarray] = {}
    for name, split in [("training", train), ("validation", validation), ("test", test)]:
        predictions[name] = np.clip(model.predict(split.x, verbose=0).ravel() * 100, 0, 100)
    actuals = {
        "training": train.y * 100, "validation": validation.y * 100, "test": test.y * 100
    }
    report = {}
    for name in actuals:
        mae, rmse = metrics(actuals[name], predictions[name])
        report[f"{name}_mae"] = mae
        report[f"{name}_rmse"] = rmse
        class_report = classification_metrics(actuals[name], predictions[name])
        for metric_name, metric_value in class_report.items():
            if metric_name != "confusion_matrix":
                report[f"{name}_{metric_name}"] = metric_value
    return model, predictions, report, fit_history.history


def generate_citizen_history(history: pd.DataFrame) -> pd.DataFrame:
    """Create 20+ previously verified reports per demo citizen from known actuals."""
    rng = np.random.default_rng(SEED + 1)
    rows: list[dict[str, Any]] = []
    candidates = history.iloc[24:-1].reset_index(drop=True)
    for citizen_number in range(1, 4):
        citizen_id = f"citizen_demo_{citizen_number:03d}"
        selected = candidates.sample(20, random_state=SEED + citizen_number)
        for _, record in selected.iterrows():
            actual = float(record["fill_level"])
            if rng.random() < 0.82:
                reported = actual + rng.normal(0, 5)
            else:
                reported = actual + rng.choice([-25, 25])
            reported = float(np.clip(reported, 0, 100))
            nearest = min(CATEGORIES, key=lambda name: abs(CATEGORIES[name] - reported))
            category_fill = CATEGORIES[nearest]
            rows.append({
                "citizen_id": citizen_id, "bin_id": record["bin_id"],
                "report_timestamp": record["timestamp"], "selected_category": nearest,
                "reported_fill": category_fill, "actual_fill": actual,
                "absolute_error": abs(category_fill - actual),
                "correct": abs(category_fill - actual) <= 10,
            })
    result = pd.DataFrame(rows)
    result.to_csv(CITIZEN_HISTORY_PATH, index=False)
    return result


def citizen_reliability(citizen_history: pd.DataFrame, citizen_id: str) -> tuple[int, int, float | None]:
    records = citizen_history[citizen_history["citizen_id"] == citizen_id]
    if records.empty:
        return 0, 0, None
    correct = int(records["correct"].sum())
    total = int(len(records))
    return correct, total, correct / total if total else None


def fusion(lstm_fill: float, citizen_fill: float | None, crs: float | None, validation_mae: float) -> dict[str, float | None]:
    if citizen_fill is None or crs is None:
        return {"final_fill": lstm_fill, "lstm_error": validation_mae / 100, "citizen_error": None,
                "lstm_reliability": None, "citizen_reliability": None, "lstm_weight": 1.0, "citizen_weight": 0.0}
    lstm_error = validation_mae / 100
    citizen_error = 1 - crs
    lstm_reliability = 1 / (lstm_error + RELIABILITY_EPSILON)
    citizen_reliability = 1 / (citizen_error + RELIABILITY_EPSILON)
    total = lstm_reliability + citizen_reliability
    lstm_weight = lstm_reliability / total
    citizen_weight = citizen_reliability / total
    return {
        "final_fill": float(np.clip(lstm_weight * lstm_fill + citizen_weight * citizen_fill, 0, 100)),
        "lstm_error": lstm_error, "citizen_error": citizen_error,
        "lstm_reliability": lstm_reliability, "citizen_reliability": citizen_reliability,
        "lstm_weight": lstm_weight, "citizen_weight": citizen_weight,
    }


def recent_positive_rate(history: pd.DataFrame, bin_id: str) -> float:
    group = history[history["bin_id"] == bin_id].sort_values("timestamp")
    observations = list(group.iloc[::-1].itertuples())
    for current, previous in zip(observations, observations[1:]):
        if not current.collection_event and not previous.collection_event:
            hours = (pd.Timestamp(current.timestamp) - pd.Timestamp(previous.timestamp)).total_seconds() / 3600
            rate = (float(current.fill_level) - float(previous.fill_level)) / hours
            if rate > 0:
                return rate
    return 0.0


def time_to_full(fill: float, rate: float) -> float | None:
    return (100 - fill) / rate if rate > 0 and fill < 100 else None


def classify_waste(photo_path: str | None, checkpoint: str | None) -> str:
    """Classify only waste type when a fine-tuned three-class model is supplied."""
    if not photo_path or not checkpoint:
        return "ResNet50 classifier not configured"
    image_file, model_file = Path(photo_path), Path(checkpoint)
    if not image_file.is_file() or not model_file.is_file():
        return "ResNet50 classifier not configured (image/checkpoint missing)"
    try:
        import tensorflow as tf
        from tensorflow.keras.utils import load_img, img_to_array
        model = tf.keras.models.load_model(model_file)
        image = load_img(image_file, target_size=(224, 224))
        values = img_to_array(image)[None, ...] / 255.0
        probabilities = model.predict(values, verbose=0)[0]
        classes = ["Organic", "Recyclable", "E-waste"]
        return classes[int(np.argmax(probabilities))]
    except Exception as exc:
        return f"ResNet50 classification unavailable: {exc}"


def display_detail(row: dict[str, Any], report: dict[str, float], classification: str) -> None:
    print(f"""
SELECTED BIN: {row['bin_id']}

STEP 1 – LSTM PREDICTION
Predicted Fill Level: {row['lstm_fill']:.2f}%
Historical MAE: {row['bin_mae']:.2f} percentage points

STEP 2 – CONFIDENCE
Uncertainty: {row['uncertainty']:.3f}
Confidence: {row['confidence']:.1%}
Threshold: {CONFIDENCE_THRESHOLD:.0%}
Decision: {row['decision']}

STEP 3 – CITIZEN PARTICIPATION
Selected Category: {row['citizen_category']}
Citizen Fill Estimate: {row['citizen_fill']}

STEP 4 – CITIZEN RELIABILITY
Correct Verified Reports: {row['correct_reports']}
Total Verified Reports: {row['verified_reports']}
CRS: {row['crs']}

STEP 5 – RELIABILITY FUSION
LSTM Error: {row['lstm_error']}
Citizen Error Proxy: {row['citizen_error']}
LSTM Reliability: {row['lstm_reliability']}
Citizen Reliability: {row['citizen_reliability']}
LSTM Weight: {row['lstm_weight']}
Citizen Weight: {row['citizen_weight']}
Final Fill Level: {row['final_fill']:.2f}%

STEP 6 – TIME-TO-FULL
Historical Filling Rate: {row['filling_rate']:.3f} percentage points/hour
Initial Time-to-Full: {row['initial_time_to_full']}
Updated Time-to-Full: {row['updated_time_to_full']}

FINAL BIN STATUS
Fill Level: {row['final_fill']:.2f}%
Time-to-Full: {row['updated_time_to_full']}
Citizen Verification: {row['decision']}
Waste Classification: {classification}
""")


def main() -> None:
    np.random.seed(SEED)
    print("INTELLIWASTE – SYNTHETIC DATA GENERATION")
    history = generate_bin_history()
    history.to_csv(BIN_HISTORY_PATH, index=False)
    print(f"\nTotal Bins: {history['bin_id'].nunique()}")
    print(f"Historical Duration: {HISTORY_DAYS} Days\nSampling Interval: 1 Hour")
    print(f"Total Records: {len(history)}")
    print("\nZones:\n" + "\n".join(ZONE_TYPES))
    print("\nSample:\n", history.head(3).to_string(index=False))

    train, validation, test, boundaries = build_splits(history)
    print("\nTraining LSTM model...")
    model, predictions, report, _ = train_and_evaluate(train, validation, test)
    model.save(MODEL_PATH)
    evaluation = pd.DataFrame([report])
    evaluation.to_csv(EVALUATION_PATH, index=False)
    print("""
LSTM MODEL TRAINING RESULTS
""")
    print("Model: LSTM\nLookback: 24 Hours\nForecast Horizon: 1 Hour")
    for split in ("training", "validation", "test"):
        print(f"{split.title()} MAE: {report[split + '_mae']:.3f} percentage points")
    for split in ("training", "validation", "test"):
        print(f"{split.title()} RMSE: {report[split + '_rmse']:.3f} percentage points")

    citizen_history = generate_citizen_history(history)
    validation_errors = np.abs(validation.y * 100 - predictions["validation"])
    overall_validation_mae = report["validation_mae"]
    per_bin_mae = pd.DataFrame({"bin_id": validation.bins, "error": validation_errors}).groupby("bin_id")["error"].mean()
    last_windows = add_features(history.sort_values(["bin_id", "timestamp"]))
    rows: list[dict[str, Any]] = []
    for bin_id in BIN_ZONES:
        group = last_windows[last_windows["bin_id"] == bin_id].sort_values("timestamp")
        window = group[feature_columns()].tail(LOOKBACK).to_numpy(
            dtype=np.float32, copy=True
        )[None, ...]
        window[:, :, feature_columns().index("fill_level")] /= 100.0
        window[:, :, feature_columns().index("filling_rate")] /= training_rate_scale(history)
        lstm_fill = float(np.clip(model.predict(window, verbose=0).ravel()[0] * 100, 0, 100))
        bin_mae = float(per_bin_mae.get(bin_id, overall_validation_mae))
        uncertainty = min(1.0, bin_mae / VALIDATION_ERROR_LIMIT)
        confidence = 1.0 - uncertainty
        required = confidence < CONFIDENCE_THRESHOLD
        citizen_fill = float(CATEGORIES[DEFAULT_CITIZEN_CATEGORY]) if required else None
        correct, verified, crs = citizen_reliability(citizen_history, DEFAULT_CITIZEN_ID) if required else (0, 0, None)
        fused = fusion(lstm_fill, citizen_fill, crs, overall_validation_mae)
        rate = recent_positive_rate(history, bin_id)
        row = {
            "bin_id": bin_id, "lstm_fill": lstm_fill, "bin_mae": bin_mae,
            "uncertainty": uncertainty, "confidence": confidence,
            "decision": "Verification Required" if required else "Not Required",
            "citizen_category": DEFAULT_CITIZEN_CATEGORY if required else "Not Required",
            "citizen_fill": citizen_fill if citizen_fill is not None else "Not Required",
            "crs": crs if crs is not None else "Not Required",
            "correct_reports": correct, "verified_reports": verified,
            **fused, "final_fill": float(fused["final_fill"]), "filling_rate": rate,
            "initial_time_to_full": time_to_full(lstm_fill, rate) if time_to_full(lstm_fill, rate) is not None else "Not estimable",
            "updated_time_to_full": time_to_full(float(fused["final_fill"]), rate) if time_to_full(float(fused["final_fill"]), rate) is not None else "Not estimable",
            "target_timestamp": group["timestamp"].max() + pd.Timedelta(hours=1),
        }
        rows.append(row)
    results = pd.DataFrame(rows)
    results.to_csv(PREDICTIONS_PATH, index=False)
    print("\nALL-BIN PREDICTION DASHBOARD")
    dashboard = results[["bin_id", "lstm_fill", "confidence", "citizen_fill", "crs", "final_fill", "updated_time_to_full"]].copy()
    dashboard.columns = ["Bin ID", "LSTM Fill", "Confidence", "Citizen Fill", "CRS", "Final Fill", "Time-to-Full"]
    print(dashboard.to_string(index=False, formatters={
        "LSTM Fill": "{:.2f}%".format, "Confidence": "{:.1%}".format, "Final Fill": "{:.2f}%".format,
    }))
    selected = next(row for row in rows if row["bin_id"] == SELECTED_BIN)
    display_detail(selected, report, classify_waste(PHOTO_PATH, RESNET_CHECKPOINT))
    print("NOTE: Historical data is synthetic; metrics demonstrate prototype behavior, not real-world municipal performance.")
    print("Fill-level fusion uses only the LSTM prediction and citizen category report. ResNet50 is classification-only.")
    print(f"\nSaved: {BIN_HISTORY_PATH.name}, {CITIZEN_HISTORY_PATH.name}, {MODEL_PATH.name}, {PREDICTIONS_PATH.name}, {EVALUATION_PATH.name}")


if __name__ == "__main__":
    main()
