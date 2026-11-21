"""Train the IntelliWaste LSTM and create the synthetic input data."""

import numpy as np
import pandas as pd

import intelliwaste as iw


def main() -> None:
    np.random.seed(iw.SEED)
    history = iw.generate_bin_history()
    history.to_csv(iw.BIN_HISTORY_PATH, index=False)
    train, validation, test, _ = iw.build_splits(history)
    model, _, report, training_history = iw.train_and_evaluate(train, validation, test)
    model.save(iw.MODEL_PATH)
    iw.generate_citizen_history(history)
    pd.DataFrame([report]).to_csv(iw.EVALUATION_PATH, index=False)
    pd.DataFrame(training_history).to_csv(iw.TRAINING_HISTORY_PATH, index=False)
    save_training_visualizations(training_history)
    print("TRAINING COMPLETE")
    print(f"Bins: {history['bin_id'].nunique()}")
    print(f"Records: {len(history)}")
    print(f"Model: {iw.MODEL_PATH}")
    print(f"Training MAE: {report['training_mae']:.3f} percentage points")
    print(f"Validation MAE: {report['validation_mae']:.3f} percentage points")
    print(f"Test MAE: {report['test_mae']:.3f} percentage points")
    print(f"Evaluation: {iw.EVALUATION_PATH}")
    print(f"Training metrics: accuracy/F1/precision/recall are fill-status bands; {iw.FILL_STATUS_LABELS}")


def save_training_visualizations(training_history: dict[str, list[float]], history_frame=None) -> None:
    """Save training curves and data-spread charts in results/train."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    curve = pd.DataFrame(training_history)
    figure, axis = plt.subplots(figsize=(10, 5))
    axis.plot(curve["loss"], label="Training loss")
    axis.plot(curve["val_loss"], label="Validation loss")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Mean squared error")
    axis.set_title("LSTM training and validation loss")
    axis.legend()
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(iw.TRAINING_CURVE_PATH, dpi=150)
    plt.close(figure)

    data = pd.read_csv(iw.BIN_HISTORY_PATH)
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    sns.histplot(data=data, x="fill_level", hue="zone_type", bins=30, element="step", stat="density", common_norm=False, ax=axes[0])
    axes[0].set_title("Fill-level distribution by zone")
    axes[0].set_xlabel("Fill level (%)")
    axes[0].set_ylabel("Density")
    sns.boxplot(data=data, x="zone_type", y="fill_level", ax=axes[1])
    axes[1].set_title("Fill-level spread by zone")
    axes[1].set_xlabel("Zone type")
    axes[1].set_ylabel("Fill level (%)")
    axes[1].tick_params(axis="x", rotation=25)
    figure.tight_layout()
    figure.savefig(iw.DATA_DISTRIBUTION_PATH, dpi=150)
    plt.close(figure)

    collection_counts = data.assign(date=pd.to_datetime(data["timestamp"]).dt.date)
    collection_counts = collection_counts[collection_counts["collection_event"]].groupby("date").size()
    figure, axis = plt.subplots(figsize=(12, 4))
    collection_counts.plot(kind="bar", ax=axis, color="#4c78a8")
    axis.set_title("Synthetic collection events by date")
    axis.set_xlabel("Date")
    axis.set_ylabel("Collection events")
    axis.tick_params(axis="x", rotation=65, labelsize=7)
    figure.tight_layout()
    figure.savefig(iw.COLLECTION_TIMELINE_PATH, dpi=150)
    plt.close(figure)


if __name__ == "__main__":
    main()
