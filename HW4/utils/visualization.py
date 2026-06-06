# =============================================================================
# src/visualize.py  —  All plotting for the stock-forecasting project
# =============================================================================
#
# Adapted from the ConvViT visualization module.
# Structural changes
#   • save_fig     — kept almost verbatim; timestamp helper retained
#   • plot_training_dashboard — GridSpec 2×2 kept; top-1/top-5 panels replaced
#                               by val_rmse and val_f1 (regression context)
#   • LR log-scale formatter  — ported directly from reference (beautiful!)
#   • New plots added: plot_per_horizon_rmse, plot_exact_vs_rolling,
#                      plot_confusion_matrix, plot_all_learning_curves
"""
utils/visualization.py
All plotting for the stock-forecasting project

Functionalities:
- save_fig
- plot_training_dashboard: GridSpec 2x2 kept; loss/LR/val_rmse/val_f1 (for regression context)
- LR log-scale formatter
- plot_per_horizon_rmse
- plot_exact_vs_rolling
- plot_confusion_matrix
- plot_all_learning_curves
"""
import datetime
import math
import os
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

os.makedirs("./assets/", exist_ok=True)


# =============================================================================
# Helpers
# =============================================================================

def _ts() -> str:
    """Return a filesystem-safe timestamp string"""
    return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def save_fig(fig_id: str, tight_layout: bool = True, fig_extension: str = "png", resolution: int = 300,
             assets_path: str = "./assets/") -> None:
    """
    Save the current matplotlib figure to <assets_path>/<fig_id>.<ext>.

    Args:
        fig_id: filename stem (no extension)
        tight_layout: call plt.tight_layout() before saving
        fig_extension: file format, default "png"
        resolution: DPI for raster formats, default 300
        assets_path: output directory (created if absent)
    """
    os.makedirs(assets_path, exist_ok=True)
    path = os.path.join(assets_path, f"{fig_id}.{fig_extension}")
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution, bbox_inches="tight")
    print(f"[plot] saved -> {path}")


def _format_log(x, pos):
    """
    Custom log-scale tick formatter — ported verbatim from reference.
    Renders 1e-3 as 10^{-3} and 4e-4 as 4 x 10^{-4}.
    """
    if x <= 0:
        return ""

    # Calculate the exponent and the base multiplier
    power = int(math.floor(math.log10(x)))
    base = x / (10 ** power)

    # If the base is 1 (e.g., 0.0001), format as 10^x
    if round(base, 2) == 1.0:
        return f"$10^{{{power}}}$"

    # For subdivisions (e.g., 0.0004), format as 4 x 10^x
    return f"${base:g} \\times 10^{{{power}}}$"


def plot_learning_curves(history: dict, model_name: str = "", save_path: str | None = None) -> None:
    """
    Two-panel figure: train/val loss | LR schedule.

    Args
    ----
    history: dict with "train_loss", "val_loss", "lr"
    model_name: used in the figure title
    save_path: full path including filename; if None the figure is discarded
    """
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    title = f"Learning Curves{' — ' + model_name if model_name else ''}"
    fig.suptitle(title, fontsize=13, fontweight="bold")

    # Panel 1: loss
    ax = axes[0]
    ax.plot(epochs, history["train_loss"], label="Training Loss", linewidth=2)
    ax.plot(epochs, history["val_loss"], label="Validation Loss", linewidth=2, linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (MSE)")
    ax.set_title("Training vs Validation Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 2: LR schedule
    ax = axes[1]
    if "lr" in history:
        ax.semilogy(epochs, history["lr"], color="darkorange", linewidth=2)
        ax.set_title("Learning Rate Schedule (log scale)")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("LR")
        ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0, subs="all"))
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(_format_log))
        ax.grid(True, alpha=0.3, which="major")
    else:
        axes[1].set_visible(False)

    if save_path:
        save_fig(os.path.splitext(os.path.basename(save_path))[0],
                 assets_path=os.path.dirname(save_path) or "./assets/", tight_layout=False)
    plt.close(fig)


def plot_training_dashboard(history: dict, model_name: str, best_epoch: int | None = None,
                            save_path: str | None = None) -> None:
    """
    Four-panel training overview: GridSpec 2x2.

    Panels
    [0,0] Loss          (Loss)
    [0,1] Val RMSE      (regression equivalent)
    [1,0] Val F1 / MAE  (F1 if turning_point, MAE otherwise)
    [1,1] LR schedule   (Learning Rate with log formatter)

    The red vertical line marks best_epoch.

    Args:
        history: dict from run_training(), must contain train_loss, val_loss, lr. Optionally: val_rmse, val_f1, val_mae.
        model_name: displayed in the figure title
        best_epoch: draws a vertical "best" marker on all panels
        save_path: full file path; uses save_fig internally
    """
    if not history or "train_loss" not in history:
        raise ValueError("history must contain at least 'train_loss' and 'val_loss'.")

    epochs = range(1, len(history["train_loss"]) + 1)

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(
        f"{model_name.upper()} — Training Dashboard "
        f"{f'(Best Epoch: {best_epoch})' if best_epoch else ''}",
        fontsize=15, fontweight="bold"
    )
    gs = gridspec.GridSpec(2, 2, hspace=0.38, wspace=0.32)

    def _vline(ax):
        """Draw best-epoch marker if provided."""
        if best_epoch is not None:
            ax.axvline(x=best_epoch, color="red", linestyle="--", alpha=0.6, label="Best epoch")

    # [0,0] Loss
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(epochs, history["train_loss"], label="Train", linewidth=2)
    ax.plot(epochs, history["val_loss"],   label="Val",   linewidth=2, linestyle="--")
    _vline(ax)
    ax.set_title("Loss (MSE)")
    ax.set_xlabel("Epoch"); ax.set_ylabel("MSE")
    ax.legend(); ax.grid(True, alpha=0.3)

    # [0,1] Val RMSE (replaces Top-1 accuracy)
    ax = fig.add_subplot(gs[0, 1])
    if "val_rmse" in history:
        ax.plot(epochs, history["val_rmse"], color="darkorange", linewidth=2, label="Val RMSE")
        _vline(ax)
        ax.set_title("Validation RMSE")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("RMSE")
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.set_visible(False)

    # [1,0] Val F1 or Val MAE (replaces Top-5 accuracy)
    ax = fig.add_subplot(gs[1, 0])
    if "val_f1" in history:                         # turning_point mode
        ax.plot(epochs, history["val_f1"], color="mediumseagreen", linewidth=2, label="Val F1")
        ax.plot(epochs, history["val_precision"], color="steelblue", linewidth=1.5, linestyle="--",
                label="Val Precision")
        ax.plot(epochs, history["val_recall"], color="tomato", linewidth=1.5, linestyle=":", label="Val Recall")
        _vline(ax)
        ax.set_title("Classification Metrics (Buy/Pass)")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Score")
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    else:
        ax.set_visible(False)

    # [1,1] LR schedule
    ax = fig.add_subplot(gs[1, 1])
    if "lr" in history:
        ax.semilogy(epochs, history["lr"], color="darkorange", linewidth=2)
        _vline(ax)
        ax.set_title("Learning Rate (log scale)")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("LR")
        ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0, subs="all"))
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(_format_log))
        ax.grid(True, alpha=0.3, which="major")
    else:
        ax.set_visible(False)

    if save_path:
        save_fig(os.path.splitext(os.path.basename(save_path))[0],
                 assets_path=os.path.dirname(save_path) or "./assets/", tight_layout=False)
    plt.close(fig)


def plot_all_learning_curves(histories: dict[str, dict], save_path: str | None = None) -> None:
    """
    One subplot per experiment run, all on a single figure.
    Useful for the report as a compact overview of all training runs.

    Args:
        histories: { run_name: history_dict }, same format as run_training() output
        save_path: Path to save the figure
    """
    n = len(histories)
    cols = min(n, 3)
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)
    flat_axes = axes.flatten()
    colors = plt.cm.tab10.colors

    for idx, (name, hist) in enumerate(histories.items()):
        ax = flat_axes[idx]
        eps = range(1, len(hist["train_loss"]) + 1)
        ax.plot(eps, hist["train_loss"], color=colors[0], linewidth=2, label="Train")
        ax.plot(eps, hist["val_loss"], color=colors[1], linewidth=2, linestyle="--", label="Val")
        ax.set_title(name, fontsize=10, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE loss")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    for idx in range(n, len(flat_axes)):
        flat_axes[idx].set_visible(False)

    fig.suptitle("All Runs — Training vs Validation Loss", fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()

    if save_path:
        save_fig(os.path.splitext(os.path.basename(save_path))[0],
                 assets_path=os.path.dirname(save_path) or "./assets/", tight_layout=False)
    plt.close(fig)


def plot_per_horizon_rmse(results: dict[str, dict], D: int, title: str = "Per-Horizon RMSE on Test Set",
                          save_path: str | None = None) -> None:
    """
    Grouped bar chart: RMSE for each forecast horizon d=1..D, one bar per model.

    Args:
        results : { "LSTM": metrics_dict, "GRU": metrics_dict,... } metrics_dict must contain "rmse" with
                                                                    shape (D,) numpy array
        D : number of forecast horizons (e.g., D=5 for d=1..5)
        title : title of the figure
        save_path : path to save the figure
    """
    horizons = [f"d={d}" for d in range(1, D + 1)]
    x = np.arange(D)
    n_models = len(results)
    width = 0.8 / n_models
    colors = plt.cm.tab10.colors

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, (name, metrics) in enumerate(results.items()):
        offset = (i - n_models / 2 + 0.5) * width
        ax.bar(x + offset, metrics["rmse"], width, label=name, color=colors[i % len(colors)], alpha=0.85,
               edgecolor="white", linewidth=0.5)

    ax.set_xlabel("Forecast horizon")
    ax.set_ylabel("RMSE")
    ax.set_title(title, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(horizons)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    if save_path:
        save_fig(os.path.splitext(os.path.basename(save_path))[0],
                 assets_path=os.path.dirname(save_path) or "./assets/")
    plt.close(fig)


def plot_exact_vs_rolling(exact_metrics: dict, rolling_metrics: dict, D: int, model_name: str = "Model",
                          save_path: str | None = None) -> None:
    """
    Side-by-side bar chart directly answering the Part (c) question:
    "is training more stable with rolling-average targets?"

    Compares per-horizon RMSE between exact (Part b) and rolling (Part c) targets for the same model
    architecture, with mean-RMSE dashed lines. D here is the number of forecast horizons  (d = 1 ... 5)
    """
    horizons = [f"d={d}" for d in range(1, D + 1)]
    x = np.arange(D)
    w = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - w/2, exact_metrics["rmse"], w, label="Exact targets (Part b)", color="steelblue", alpha=0.85,
           edgecolor="white")
    ax.bar(x + w/2, rolling_metrics["rmse"], w, label="Rolling-avg targets (Part c)", color="tomato", alpha=0.85,
           edgecolor="white")

    ax.axhline(exact_metrics["mean_rmse"], color="steelblue", linestyle=":", linewidth=2,
               label=f"Mean exact = {exact_metrics['mean_rmse']:.5f}")
    ax.axhline(rolling_metrics["mean_rmse"], color="tomato", linestyle=":", linewidth=2,
               label=f"Mean rolling = {rolling_metrics['mean_rmse']:.5f}")

    ax.set_xlabel("Forecast horizon")
    ax.set_ylabel("RMSE on test set")
    ax.set_title(f"{model_name} - Exact vs Rolling-Average Targets", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(horizons)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    if save_path:
        save_fig(os.path.splitext(os.path.basename(save_path))[0],
                 assets_path=os.path.dirname(save_path) or "./assets/")
    plt.close(fig)


def plot_confusion_matrix(cm: np.ndarray, model_name: str = "", save_path: str | None = None) -> None:
    """
    Annotated heatmap of the 2x2 binary confusion matrix.
    Rows = actual, Columns = predicted.  Cell color scales with count.

    Args:
        cm: (2, 2) int array  [[TN, FP], [FN, TP]]
        model_name: shown in the title
        save_path: Path to save the figure
    """
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")

    labels = ["Pass (0)", "Buy (1)"]
    ax.set_xticks([0, 1])
    ax.set_xticklabels(labels)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    title = "Confusion Matrix — Buy/Pass Signal"
    if model_name:
        title += f" ({model_name})"
    ax.set_title(title, fontweight="bold")

    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=14, fontweight="bold",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    if save_path:
        save_fig(os.path.splitext(os.path.basename(save_path))[0],
                 assets_path=os.path.dirname(save_path) or "./assets/")
    plt.close(fig)
