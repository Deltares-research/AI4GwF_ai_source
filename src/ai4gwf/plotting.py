import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def save_epoch_plots(
    prediction_path: Path, history: list[dict], output_range, output_dir: Path,
) -> None:
    """Render stored target units; volumes use the central depth slice."""
    with np.load(prediction_path) as saved:
        targets = saved["targets"]
        predictions = saved["predictions"]
    if targets.shape != predictions.shape or targets.ndim not in (4, 5) or not len(targets):
        raise ValueError("Epoch plots require matching nonempty 2D/3D prediction arrays")
    if not np.isfinite(targets).all() or not np.isfinite(predictions).all():
        raise ValueError("Epoch plots contain non-finite targets or predictions")
    slice_label = ""
    if targets.ndim == 5:
        depth = targets.shape[2] // 2
        targets, predictions = targets[:, :, depth], predictions[:, :, depth]
        slice_label = f", depth slice {depth}"
    lower, upper = float(output_range[1]), float(output_range[2])
    for channel in range(targets.shape[1]):
        truth, predicted = targets[:, channel], predictions[:, channel]
        error = predicted - truth
        vmin = min(lower, float(truth.min()), float(predicted.min()))
        vmax = max(upper, float(truth.max()), float(predicted.max()))
        error_limit = max(float(np.abs(error).max()), np.finfo(np.float32).eps)
        fig, axes = plt.subplots(
            len(truth), 3, figsize=(13, 4 * len(truth)), squeeze=False,
            layout="constrained",
        )
        for sample in range(len(truth)):
            for column, values in enumerate((truth[sample], predicted[sample], error[sample])):
                ax = axes[sample, column]
                image = ax.imshow(
                    values, origin="upper", interpolation="nearest",
                    cmap="RdBu_r" if column == 2 else "viridis",
                    vmin=-error_limit if column == 2 else vmin,
                    vmax=error_limit if column == 2 else vmax,
                )
                title = ("Ground truth", "Prediction", "Prediction - ground truth")[column]
                ax.set_title(f"Validation example {sample + 1}: {title}")
                ax.set_xlabel("Grid column")
                ax.set_ylabel("Grid row")
                fig.colorbar(image, ax=ax, shrink=0.8, label=f"{output_range[0]}: stored units")
        fig.suptitle(
            f"Epoch {history[-1]['epoch']}, step {history[-1]['step']}, "
            f"channel {channel}{slice_label}\n"
            "Ground truth and prediction share color limits; predictions are not clipped"
        )
        fig.savefig(output_dir / f"comparison_channel_{channel:02d}.png", dpi=120)
        plt.close(fig)

    epochs = [record["epoch"] for record in history]
    fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
    ax.plot(epochs, [record["train_mse"] for record in history], marker="o", label="Training")
    ax.plot(epochs, [record["validation_mse"] for record in history], marker="o", label="Validation")
    if all(record[key] > 0 for record in history for key in ("train_mse", "validation_mse")):
        ax.set_yscale("log")
    ax.set_xlabel("Complete training epoch")
    ax.set_ylabel("Normalized MSE")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(output_dir.parent / "loss_curve.png", dpi=140)
    plt.close(fig)
    (output_dir.parent / "loss_history.json").write_text(json.dumps(history, indent=2) + "\n")
