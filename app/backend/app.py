"""Flask API for the IntelliWaste React dashboard."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import intelliwaste as iw  # noqa: E402

app = Flask(__name__)
CORS(app)

history = pd.read_csv(iw.BIN_HISTORY_PATH, parse_dates=["timestamp"])
citizen_history = pd.read_csv(iw.CITIZEN_HISTORY_PATH, parse_dates=["report_timestamp"])
evaluation_frame = pd.read_csv(iw.EVALUATION_PATH)
_model: Any | None = None
_cache: dict[str, dict[str, Any]] | None = None


def load_model():
    global _model
    if _model is None:
        if not iw.MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Trained model is missing at {iw.MODEL_PATH}. Run `uv run python train.py`."
            )
        import tensorflow as tf

        tf.config.threading.set_intra_op_parallelism_threads(iw.CPU_THREADS)
        tf.config.threading.set_inter_op_parallelism_threads(iw.INTER_OP_THREADS)
        _model = tf.keras.models.load_model(iw.MODEL_PATH)
    return _model


def safe_number(value: Any) -> float | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    return float(value)


def bin_history(bin_id: str) -> pd.DataFrame:
    return history[history["bin_id"] == bin_id].sort_values("timestamp")


def predict_cache() -> dict[str, dict[str, Any]]:
    global _cache
    if _cache is not None:
        return _cache
    model = load_model()
    _, validation, _, _ = iw.build_splits(history)
    validation_prediction = np.clip(model.predict(validation.x, verbose=0).ravel() * 100, 0, 100)
    validation_errors = np.abs(validation.y * 100 - validation_prediction)
    per_bin_mae = pd.DataFrame({
        "bin_id": validation.bins,
        "error": validation_errors,
    }).groupby("bin_id")["error"].mean()
    overall_mae = float(np.mean(validation_errors))
    prepared = iw.add_features(history.sort_values(["bin_id", "timestamp"]))
    columns = iw.feature_columns()
    rate_scale = iw.training_rate_scale(history)
    _cache = {}
    for bin_id in iw.BIN_ZONES:
        group = prepared[prepared["bin_id"] == bin_id].sort_values("timestamp")
        window = group[columns].tail(iw.LOOKBACK).to_numpy(dtype=np.float32, copy=True)[None, ...]
        window[:, :, columns.index("fill_level")] /= 100.0
        window[:, :, columns.index("filling_rate")] /= rate_scale
        predicted = float(np.clip(model.predict(window, verbose=0).ravel()[0] * 100, 0, 100))
        bin_mae = float(per_bin_mae.get(bin_id, overall_mae))
        uncertainty = min(1.0, bin_mae / iw.VALIDATION_ERROR_LIMIT)
        rate = iw.recent_positive_rate(history, bin_id)
        _cache[bin_id] = {
            "predicted_fill": predicted,
            "validation_mae": bin_mae,
            "uncertainty": uncertainty,
            "confidence": 1.0 - uncertainty,
            "filling_rate": rate,
            "target_timestamp": group["timestamp"].max() + pd.Timedelta(hours=1),
        }
    return _cache


def base_bin_payload(bin_id: str) -> dict[str, Any]:
    group = bin_history(bin_id)
    if group.empty:
        raise KeyError(f"Unknown bin: {bin_id}")
    latest = group.iloc[-1]
    forecast = predict_cache().get(bin_id)
    return {
        "bin_id": bin_id,
        "zone_type": str(latest["zone_type"]),
        "latest_fill": float(latest["fill_level"]),
        "average_fill": float(group["fill_level"].mean()),
        "maximum_fill": float(group["fill_level"].max()),
        "filling_rate": float(group["filling_rate"].mean()),
        "latest_observation": latest["timestamp"].isoformat(),
        "predicted_fill": forecast["predicted_fill"] if forecast else None,
        "prediction_confidence": forecast["confidence"] if forecast else None,
        "time_to_full_hours": iw.time_to_full(
            float(forecast["predicted_fill"]), float(forecast["filling_rate"])
        ) if forecast else None,
    }


def detail_payload(bin_id: str) -> dict[str, Any]:
    base = base_bin_payload(bin_id)
    group = bin_history(bin_id)
    base["history"] = [
        {
            "timestamp": row.timestamp.isoformat(),
            "fill_level": float(row.fill_level),
            "filling_rate": float(row.filling_rate),
            "collection_event": bool(row.collection_event),
        }
        for row in group.itertuples()
    ]
    base["last_24_hours"] = base["history"][-iw.LOOKBACK:]
    return base


def citizen_payload(citizen_id: str) -> dict[str, Any]:
    rows = citizen_history[citizen_history["citizen_id"] == citizen_id]
    if rows.empty:
        raise KeyError(f"Unknown citizen: {citizen_id}")
    correct = int(rows["correct"].sum())
    total = int(len(rows))
    return {
        "citizen_id": citizen_id,
        "verified_reports": total,
        "correct_reports": correct,
        "incorrect_reports": total - correct,
        "crs": correct / total if total else None,
    }


def run_prediction(bin_id: str, demo_confidence: float | None = None) -> dict[str, Any]:
    result = base_bin_payload(bin_id)
    actual_confidence = float(result["prediction_confidence"])
    confidence = actual_confidence if demo_confidence is None else float(demo_confidence)
    confidence = min(1.0, max(0.0, confidence))
    uncertainty = 1.0 - confidence
    result.update({
        "actual_confidence": actual_confidence,
        "confidence": confidence,
        "uncertainty": uncertainty,
        "confidence_threshold": iw.CONFIDENCE_THRESHOLD,
        "demo_override_active": demo_confidence is not None,
        "requires_verification": confidence < iw.CONFIDENCE_THRESHOLD,
        "decision": "Request Citizen Verification" if confidence < iw.CONFIDENCE_THRESHOLD else "Retain LSTM Prediction",
        "validation_mae": predict_cache()[bin_id]["validation_mae"],
    })
    return result


@app.get("/api/health")
def health():
    return jsonify({
        "status": "ok",
        "model_available": iw.MODEL_PATH.exists(),
        "data_available": iw.BIN_HISTORY_PATH.exists() and iw.CITIZEN_HISTORY_PATH.exists(),
    })


@app.get("/api/summary")
def summary():
    return jsonify({
        "total_bins": int(history["bin_id"].nunique()),
        "historical_records": int(len(history)),
        "average_fill": float(history["fill_level"].mean()),
        "average_filling_rate": float(history["filling_rate"].mean()),
        "zone_categories": int(history["zone_type"].nunique()),
        "date_start": history["timestamp"].min().isoformat(),
        "date_end": history["timestamp"].max().isoformat(),
        "model_available": iw.MODEL_PATH.exists(),
        "model_path": str(iw.MODEL_PATH),
    })


@app.get("/api/bins")
def bins():
    try:
        return jsonify({"bins": [base_bin_payload(bin_id) for bin_id in iw.BIN_ZONES]})
    except Exception as exc:
        return jsonify({"error": str(exc), "bins": []}), 503


@app.get("/api/bins/<bin_id>")
def bin_detail(bin_id: str):
    try:
        return jsonify(detail_payload(bin_id))
    except KeyError as exc:
        return jsonify({"error": str(exc)}), 404
    except Exception as exc:
        return jsonify({"error": str(exc)}), 503


@app.get("/api/citizens")
def citizens():
    return jsonify({"citizens": [citizen_payload(str(value)) for value in sorted(citizen_history["citizen_id"].unique())]})


@app.post("/api/predict")
def predict():
    body = request.get_json(silent=True) or {}
    bin_id = str(body.get("bin_id", ""))
    demo_confidence = body.get("demo_confidence")
    try:
        return jsonify(run_prediction(bin_id, float(demo_confidence) if demo_confidence is not None else None))
    except (KeyError, ValueError) as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"error": str(exc)}), 503


@app.post("/api/fusion")
def fusion():
    body = request.get_json(silent=True) or {}
    try:
        bin_id = str(body["bin_id"])
        citizen_id = str(body["citizen_id"])
        lower = float(body["lower_limit"])
        upper = float(body["upper_limit"])
        if lower not in {0, 25, 50, 75, 100} or upper not in {0, 25, 50, 75, 100}:
            raise ValueError("Fill limits must be 0, 25, 50, 75, or 100.")
        if lower > upper:
            raise ValueError("Lower fill limit cannot exceed upper fill limit.")
        demo_confidence = body.get("demo_confidence")
        prediction = run_prediction(bin_id, float(demo_confidence) if demo_confidence is not None else None)
        citizen = citizen_payload(citizen_id)
        citizen_fill = (lower + upper) / 2.0
        fused = iw.fusion(
            float(prediction["predicted_fill"]),
            citizen_fill,
            float(citizen["crs"]),
            float(prediction["validation_mae"]),
        )
        rate = float(prediction["filling_rate"])
        initial = iw.time_to_full(float(prediction["predicted_fill"]), rate)
        updated = iw.time_to_full(float(fused["final_fill"]), rate)
        return jsonify({
            "prediction": prediction,
            "citizen": citizen,
            "lower_limit": lower,
            "upper_limit": upper,
            "citizen_fill": citizen_fill,
            "fusion": fused,
            "initial_time_to_full_hours": initial,
            "updated_time_to_full_hours": updated,
            "initial_time_to_full_days": initial / 24 if initial is not None else None,
            "updated_time_to_full_days": updated / 24 if updated is not None else None,
            "steps": [
                {"title": "LSTM prediction", "formula": "Fv = LSTM(last 24 hourly observations)", "value": f"{prediction['predicted_fill']:.2f}%"},
                {"title": "Confidence decision", "formula": "Cv = 1 - Uv", "value": f"{prediction['confidence']:.1%} vs {iw.CONFIDENCE_THRESHOLD:.0%} threshold"},
                {"title": "Citizen fill estimate", "formula": "Fc = (Lc + Uc) / 2", "value": f"({lower:.0f} + {upper:.0f}) / 2 = {citizen_fill:.2f}%"},
                {"title": "Citizen reliability", "formula": "CRS = Ncorrect / Nverified", "value": f"{citizen['correct_reports']} / {citizen['verified_reports']} = {citizen['crs']:.3f}"},
                {"title": "Reliability calculation", "formula": "ev = MAEval / 100; ec = 1 - CRS; R = 1 / (e + 0.001)", "value": f"ev={fused['lstm_error']:.5f}, ec={fused['citizen_error']:.5f}"},
                {"title": "Reliability weights", "formula": "wv = Rv/(Rv+Rc), wc = Rc/(Rv+Rc)", "value": f"wv={fused['lstm_weight']:.3f}, wc={fused['citizen_weight']:.3f}"},
                {"title": "Final fill level", "formula": "Ffinal = wv*Fv + wc*Fc", "value": f"{fused['final_fill']:.2f}%"},
                {"title": "Time-to-full", "formula": "T = (100 - F) / Gt", "value": f"{updated:.2f} hours" if updated is not None else "Not estimable"},
            ],
        })
    except (KeyError, ValueError) as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"error": str(exc)}), 503


@app.get("/api/evaluation")
def evaluation():
    training_history_path = iw.TRAINING_HISTORY_PATH
    training_history = pd.read_csv(training_history_path).fillna(0).to_dict(orient="records") if training_history_path.exists() else []
    metrics = evaluation_frame.iloc[0].to_dict() if not evaluation_frame.empty else {}
    model = load_model()
    return jsonify({
        "metrics": metrics,
        "training_history": training_history,
        "model": {
            "input_shape": model.input_shape,
            "output_shape": model.output_shape,
            "parameters": int(model.count_params()),
            "layers": [{"name": layer.name, "type": layer.__class__.__name__, "parameters": layer.count_params()} for layer in model.layers],
        },
        "status_bands": iw.FILL_STATUS_LABELS,
        "visualizations": {
            "training_curve": "train/training_curve.png",
            "data_distribution": "train/data_distribution.png",
            "dashboard": "test/fill_prediction_dashboard.png",
            "validation_diagnostics": "test/validation_diagnostics.png",
            "metrics_heatmap": "test/testing_metrics_heatmap.png",
            "confusion_matrix": "test/fill_status_confusion_matrix.png",
            "error_heatmap": "test/test_error_heatmap.png",
        },
    })


@app.get("/api/artifacts/<path:artifact_path>")
def artifact(artifact_path: str):
    requested = (iw.RESULTS_DIR / artifact_path).resolve()
    if iw.RESULTS_DIR.resolve() not in requested.parents or not requested.is_file():
        return jsonify({"error": "Artifact not found"}), 404
    return send_file(requested)


@app.post("/api/classify")
def classify():
    return jsonify({
        "status": "not_configured",
        "message": "Classifier not configured. A fine-tuned three-class ResNet50 checkpoint is required.",
        "classes": ["Organic", "Recyclable", "E-waste"],
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
