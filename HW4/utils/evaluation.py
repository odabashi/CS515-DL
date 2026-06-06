"""
utils/evaluation.py
Metric accumulators, model summary, results table

Functionalities:
- RegressionMetrics: MSE / RMSE / MAE per forecast horizon
- BinaryMetrics: precision / recall / F1 / confusion matrix
"""
import os
import sys
import numpy as np
import torch
import torch.nn as nn


# =============================================================================
# RegressionMetrics
# =============================================================================

class RegressionMetrics:
    """
    Stateful accumulator for regression metrics, one slot per horizon d=1 ... D.

    Every caller follows the same pattern regardless of task type since they have the update/compute/reset interface.

    Metrics
    -------
    MSE  = (1/N) Σ (ŷ − y)²
    RMSE = √MSE (same units as the return ratio - more interpretable)
    MAE  = (1/N) Σ |ŷ − y| (robust to outliers)
    """
    def __init__(self, num_horizons: int) -> None:
        self.D = num_horizons
        self.reset()

    def reset(self) -> None:
        """Zero all accumulators. Call before each evaluation pass."""
        self._sse = np.zeros(self.D)   # sum of squared errors,  shape (D,)
        self._sae = np.zeros(self.D)   # sum of absolute errors, shape (D,)
        self._count = 0

    def update(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> None:
        """
        Accumulate one batch.

        Parameters
        ----------
        y_pred : (batch, D) - model predictions
        y_true : (batch, D) - ground-truth return ratios
        """
        pred = y_pred.detach().cpu().numpy()
        true = y_true.detach().cpu().numpy()
        diff = pred - true
        self._sse += (diff ** 2).sum(axis=0)
        self._sae += np.abs(diff).sum(axis=0)
        self._count += pred.shape[0]

    def compute(self) -> dict:
        """
        Compute final metrics from accumulated state.

        Returns
        -------
        dict with:
          "mse", "rmse", "mae" - each shape (D, ) numpy array
          "mean_mse", "mean_rmse", "mean_mae" - scalars averaged over horizons
          "n_samples" - total sample count
        """
        n = max(self._count, 1)
        mse = self._sse / n
        rmse = np.sqrt(mse)
        mae = self._sae / n
        return {
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "mean_mse": float(mse.mean()),
            "mean_rmse": float(rmse.mean()),
            "mean_mae": float(mae.mean()),
            "n_samples": self._count,
        }

    def __repr__(self) -> str:
        r = self.compute()
        return f"RegressionMetrics(n={r['n_samples']}, mean_mse={r['mean_mse']:.6f}, mean_rmse={r['mean_rmse']:.6f})"


# =============================================================================
# BinaryMetrics
# =============================================================================

class BinaryMetrics:
    """
    Stateful accumulator for binary classification (buy=1, pass=0).

    Follows the update/compute/reset pattern.

    Confusion matrix layout:
          Predicted 0   Predicted 1
    True 0    TN            FP
    True 1    FN            TP
    """

    def __init__(self) -> None:
        self._tp = 0
        self._fp = 0
        self._fn = 0
        self._tn = 0
        self.reset()

    def reset(self) -> None:
        self._tp = 0
        self._fp = 0
        self._fn = 0
        self._tn = 0

    def update(self, y_pred_prob: torch.Tensor, y_true: torch.Tensor, threshold: float = 0.5) -> None:
        """
        Args:
            y_pred_prob: (batch, ) or (batch, 1) - sigmoid output in [0, 1]
            y_true: (batch, ) or (batch, 1) - binary ground truth {0, 1}
            threshold: float, default=0.5 - threshold for binary classification
        """
        pred = (y_pred_prob.detach().cpu().squeeze() >= threshold).long()
        true = y_true.detach().cpu().squeeze().long()
        self._tp += int(((pred == 1) & (true == 1)).sum())
        self._fp += int(((pred == 1) & (true == 0)).sum())
        self._fn += int(((pred == 0) & (true == 1)).sum())
        self._tn += int(((pred == 0) & (true == 0)).sum())

    def compute(self) -> dict:
        """
        Returns accuracy, precision, recall, F1, and the (2x2) confusion matrix.
        eps prevents division by zero when all samples belong to one class.
        """
        eps = 1e-8
        tp, fp, fn, tn = self._tp, self._fp, self._fn, self._tn
        total = tp + fp + fn + tn
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        accuracy = (tp + tn) / max(total, 1)
        return {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "confusion_matrix": np.array([[tn, fp], [fn, tp]], dtype=int),
        }

    def __repr__(self) -> str:
        r = self.compute()
        return (f"BinaryMetrics(acc={r['accuracy']:.4f}, prec={r['precision']:.4f}, "
                f"rec={r['recall']:.4f}, f1={r['f1']:.4f})")


# =============================================================================
# model_summary
# =============================================================================

def model_summary(model: nn.Module, model_name: str = "") -> dict:
    """
    Count trainable and total parameters.

    ptflops doesn't handle RNN ops accurately, so we count parameters directly - the standard approach for 
    recurrent models.
    Always restores the model's original train/eval mode.
    """
    was_training = model.training
    model.eval()

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    model.train(was_training)

    summary = {
        "model_name": model_name or type(model).__name__,
        "total_params": total,
        "trainable_params": trainable,
        "frozen_params": total - trainable,
    }
    
    print(f"{'-'*40}")
    print(f"Model: {summary['model_name']}")
    print(f"Total: {total:,}")
    print(f"Trainable: {trainable:,}")
    print(f"Frozen: {total - trainable:,}")
    print(f"{'-'*40}")
    return summary


# =============================================================================
# print_results_table  (text output - no matplotlib)
# =============================================================================

def print_results_table(all_results: dict[str, dict]) -> None:
    """
    Formatted ASCII comparison table across all runs.
    Classification columns show '-' for regression-only runs.
    """
    header = (f"{'Run':<26} {'MSE':>10} {'RMSE':>10} {'MAE':>10} "
              f"{'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7}")
    sep = "=" * len(header)
    print(f"\n{sep}\nRESULTS TABLE\n{sep}")
    print(header)
    print("-" * len(header))
    for name, r in all_results.items():
        mse = f"{r.get('mean_mse',  float('nan')):.6f}"
        rmse = f"{r.get('mean_rmse', float('nan')):.6f}"
        mae = f"{r.get('mean_mae',  float('nan')):.6f}"
        acc = f"{r['accuracy']:.4f}" if "accuracy" in r else "   -   "
        prec = f"{r['precision']:.4f}" if "precision" in r else "   -   "
        rec = f"{r['recall']:.4f}" if "recall" in r else "   -   "
        f1 = f"{r['f1']:.4f}" if "f1" in r else "   -   "
        print(f"{name:<26} {mse:>10} {rmse:>10} {mae:>10} "
              f"{acc:>7} {prec:>7} {rec:>7} {f1:>7}")
    print(f"{sep}\n")
