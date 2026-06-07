"""
test.py
Loads best checkpoint, evaluate on test set, log and visualize

Functionalities:
    - get_test_loader: build the right DataLoader for the chosen mode
    - run_eval: core inference loop (raw predictions + metrics)
    - run_test: orchestrate: load weights -> eval -> log -> plot

New:
- Three modes instead of one: "exact", "rolling", "turning_point"
- Regression metrics (MSE/RMSE/MAE)
- Dual-head output for turning_point (returns + buy signal)
"""
import os
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import build_dataloaders
from utils.evaluation import RegressionMetrics, BinaryMetrics, model_summary, print_results_table
from utils.visualization import plot_per_horizon_rmse, plot_confusion_matrix, plot_exact_vs_rolling
from parameters import D


def get_test_loader(params: dict, mode: str) -> DataLoader:
    """
    Return only the test DataLoader for the given experiment mode.
    Dataset-agnostic caller, one return value.

    Args:
        params : flat cfg dict from get_params()
        mode: "exact" | "rolling" | "turning_point"

    Returns:
        DataLoader
    """
    _, _, test_loader = build_dataloaders(mode=mode, batch_size=params["batch_size"], tickers=params.get("tickers"), data_dir=params.get("data_dir", "data"))
    return test_loader


@torch.no_grad()
def run_eval(model: nn.Module, test_loader: DataLoader, mode: str, device: str = "cpu", capture_sample: bool = True) -> dict:
    """
    Core evaluation loop over the full test set.

    Accumulates metrics AND raw prediction arrays so callers can do further analysis (e.g. scatter plots, per-ticker
    breakdowns) without re-running inference. Adapted for regression and dual-head output.

    Args:
        model: model already in eval mode with weights loaded
        test_loader: test DataLoader (from get_test_loader)
        mode: "exact" | "rolling" | "turning_point"
        device: device to run evaluation on
        capture_sample: if True, stores the first batch for spot-check logging

    Returns:
        dict with keys:
            reg_metrics: RegressionMetrics.compute() dict (always present)
            all_preds: (N, D) float32 numpy array of predicted return ratios
            all_targets: (N, D) float32 numpy array of true return ratios
            sample: first-batch dict {X, preds, targets} or None
            [turning_point only]
            cls_metrics: BinaryMetrics.compute() dict
            all_probs: (N, ) predicted buy probabilities
            all_labels: (N, ) true binary labels
    """
    model.eval()

    reg_m = RegressionMetrics(num_horizons=D)
    cls_m = BinaryMetrics() if mode == "turning_point" else None

    all_preds: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    all_probs: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    sample: Optional[dict] = None

    for batch_idx, batch in enumerate(test_loader):
        X = batch[0].to(device)

        if mode in ("exact", "rolling"):
            y_true = batch[1].to(device)
            y_pred = model(X)

            reg_m.update(y_pred, y_true)
            all_preds.append(y_pred.cpu().numpy())
            all_targets.append(y_true.cpu().numpy())

            # Capture first batch for spot-check
            if capture_sample and batch_idx == 0:
                sample = {"X": X.cpu(), "preds": y_pred.cpu(), "targets": y_true.cpu()}

        else:  # turning_point
            y_ret_true = batch[1].to(device)
            y_lbl_true = batch[2].to(device)

            y_ret_pred, y_lbl_pred = model(X)

            y_lbl_prob = torch.sigmoid(y_lbl_pred)

            reg_m.update(y_ret_pred, y_ret_true)
            cls_m.update(y_lbl_prob, y_lbl_true)

            all_preds.append(y_ret_pred.cpu().numpy())
            all_targets.append(y_ret_true.cpu().numpy())
            all_probs.append(y_lbl_prob.cpu().squeeze().numpy())
            all_labels.append(y_lbl_true.cpu().numpy())

            if capture_sample and batch_idx == 0:
                sample = {"X": X.cpu(), "ret_preds": y_ret_pred.cpu(), "ret_targets": y_ret_true.cpu(),
                          "buy_probs": y_lbl_prob.cpu(), "buy_labels": y_lbl_true.cpu()}

    output = {
        "reg_metrics": reg_m.compute(),
        "all_preds": np.concatenate(all_preds, axis=0),
        "all_targets": np.concatenate(all_targets, axis=0),
        "sample": sample,
    }
    if mode == "turning_point":
        output["cls_metrics"] = cls_m.compute()
        output["all_probs"] = np.concatenate(all_probs, axis=0)
        output["all_labels"] = np.concatenate(all_labels, axis=0)

    return output


# =============================================================================
# run_test  —  full test pipeline
# =============================================================================

def run_test(model: nn.Module, mode: str, params: dict, run_name: str,
             device: str = "cpu", plots_dir: str = "assets") -> dict:
    """
    Load best checkpoint weights -> test-set inference -> log -> plots.

    Args:
        model: model instance (weights overwritten by checkpoint)
        mode: "exact" | "rolling" | "turning_point"
        params: flat cfg dict from get_params(). Used for checkpoint_dir, batch_size, tickers, data_dir, D, etc.
        run_name: checkpoint subdirectory name (e.g. "lstm_exact")
        device: compute device string
        plots_dir: directory where result plots are saved

    Returns:
        dict from run_eval (metrics + raw arrays)
    """
    # Load best checkpoint
    checkpoint_dir = params.get("checkpoint_dir", "checkpoints")
    checkpoint_path = os.path.join(checkpoint_dir, run_name, f"best_{mode}.pt")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: '{checkpoint_path}'\nTrain the model first with run_name='{run_name}'."
        )

    print(f"\n{'=' * 40}")
    print(f"Testing starts: (mode={mode}), (run={run_name})")
    print(f"{'=' * 40}")
    print(f"Loading weights from '{checkpoint_path}'")

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    trained_epoch = ckpt.get("epoch", "?")
    best_val_loss = ckpt.get("best_val_loss", float("nan"))
    print(f"Loaded Checkpoint with {trained_epoch} epochs (best_val_loss={best_val_loss:.6f})")

    # Model summary
    model_summary(model, model_name=run_name)

    # Test DataLoader
    test_loader = get_test_loader(params, mode)
    print(f"Test batches: {len(test_loader)}")

    # Inference
    output = run_eval(model, test_loader, mode, device, capture_sample=True)

    # Log results
    reg = output["reg_metrics"]
    n = output["all_preds"].shape[0]

    print(f"\n=== Regression metrics (n={n} samples) ===")
    print(f"{'Horizon':<10} {'MSE':>10} {'RMSE':>10} {'MAE':>10}")
    print(f"{'-' * 42}")

    for d in range(D):
        print(f"d={d + 1:<8} {reg['mse'][d]:>10.6f} {reg['rmse'][d]:>10.6f} {reg['mae'][d]:>10.6f}")
    print(f"{'Mean':<10} {reg['mean_mse']:>10.6f} {reg['mean_rmse']:>10.6f} {reg['mean_mae']:>10.6f}")

    if mode == "turning_point":
        cls = output["cls_metrics"]
        print(f"\n=== Classification metrics (buy/pass signal) ===")
        print(f"Accuracy  : {cls['accuracy']:.4f}")
        print(f"Precision : {cls['precision']:.4f}")
        print(f"Recall    : {cls['recall']:.4f}")
        print(f"F1        : {cls['f1']:.4f}")
        print(f"Confusion matrix (rows=actual, cols=predicted):")
        cm = cls["confusion_matrix"]
        print(f"             Pass   Buy")
        print(f"Actual Pass  {cm[0, 0]:>4}  {cm[0, 1]:>4}")
        print(f"Actual Buy   {cm[1, 0]:>4}  {cm[1, 1]:>4}")

    # First-batch spot check
    s = output.get("sample")
    if s is not None:
        print(f"\n=== First-batch spot check (first 3 windows) ===")
        if mode in ("exact", "rolling"):
            for i in range(min(3, s["preds"].shape[0])):
                p = "  ".join(f"{v:+.4f}" for v in s["preds"][i].numpy())
                t = "  ".join(f"{v:+.4f}" for v in s["targets"][i].numpy())
                print(f"[{i}] pred: {p}")
                print(f"  target: {t}")
        else:
            for i in range(min(3, s["buy_probs"].shape[0])):
                prob = s["buy_probs"][i].item()
                label = int(s["buy_labels"][i].item())
                dec = "BUY ✓" if prob >= 0.5 else "PASS"
                print(f"[{i}] P(buy)={prob:.3f}, label={label} => Decision: {dec}")

    # Plots
    os.makedirs(plots_dir, exist_ok=True)

    plot_per_horizon_rmse(
        {run_name: reg},
        D=params.get("D", D),
        title=f"Per-Horizon RMSE - {run_name}",
        save_path=os.path.join(plots_dir, f"{run_name}_per_horizon_rmse.png"),
    )

    if mode == "turning_point":
        plot_confusion_matrix(
            output["cls_metrics"]["confusion_matrix"],
            model_name=run_name,
            save_path=os.path.join(plots_dir, f"{run_name}_confusion_matrix.png"),
        )

    print(f"Plots saved to '{plots_dir}/'")
    print(f"{'=' * 40}\n")
    return output

