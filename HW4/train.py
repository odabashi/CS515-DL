"""
train.py
Training loop, validation, checkpointing for all three experiment modes.

Tunable hyperparameters (lr, weight_decay, num_epochs, patience, ...) arrive
as function arguments so callers can pass them straight from get_params().
Structural constants (T, D) are imported directly from parameters.py.
"""
import copy
import os
from math import inf
from pathlib import Path
from typing import Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils.evaluation import BinaryMetrics
import sys
sys.path.insert(0, os.path.dirname(__file__))


# =============================================================================
# EarlyStopping
# =============================================================================

class EarlyStopping:
    """
    Halt training when validation loss stops improving.

    How it works:
    ------------
    After each epoch we call .step(val_loss).
    - If val_loss drops by at least min_delta  	-> reset the patience counter.
    - Otherwise 								-> increment counter.
    - When counter >= patience                 	-> set self.stop = True.

    The training loop checks `if early_stopping.stop: break` after every epoch.

    Attributes:
        stop: set True when patience is exhausted
        counter: consecutive epochs without improvement
    """
    def __init__(self, patience: int = 10, min_delta: float = 1e-5) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss: float = inf
        self.counter: int = 0
        self.stop: bool = False

    def step(self, val_loss: float) -> None:
        if self.patience == 0:
            return
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            print(f"[EarlyStopping] no improvement {self.counter}/{self.patience} (best={self.best_loss:.6f})")
            if self.counter >= self.patience:
                self.stop = True
                print("[EarlyStopping] triggered.")

    def state_dict(self) -> dict:
        return {"counter": self.counter, "best_loss": self.best_loss, "stop": self.stop}

    def load_state_dict(self, state: dict) -> None:
        self.counter = state["counter"]
        self.best_loss = state["best_loss"]
        self.stop = state["stop"]

    def __repr__(self) -> str:
        return f"EarlyStopping(patience={self.patience}, counter={self.counter}, best={self.best_loss:.6f})"


# =============================================================================
# AverageMeter
# =============================================================================

class AverageMeter:
    """Running mean of any scalar (loss, RMSE, ...)."""
    def __init__(self, name: str = "") -> None:
        self.name = name
        self.val: float = 0.0
        self.avg: float = 0.0
        self.sum: float = 0.0
        self.count: int = 0
        self.reset()

    def reset(self) -> None:
        self.val: float = 0.0
        self.avg: float = 0.0
        self.sum: float = 0.0
        self.count: int = 0

    def update(self, val: float, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __repr__(self) -> str:
        return f"AverageMeter(name={self.name!r}, avg={self.avg:.6f}, n={self.count})"


# =============================================================================
# Optimizer & Scheduler
# =============================================================================


def get_optimizer(model: nn.Module, lr: float = 1e-3, weight_decay: float = 1e-4) -> torch.optim.Optimizer:
    """AdamW on trainable parameters only."""
    trainable = filter(lambda p: p.requires_grad, model.parameters())

    # AdamW applies weight decay (L2 regularization) DIRECTLY to the weights rather than folding it into the gradient
    # update. This is mathematically cleaner and empirically works better for most deep learning tasks.
    return torch.optim.AdamW(trainable, lr=lr, weight_decay=weight_decay)


def get_scheduler(optimizer: torch.optim.Optimizer, num_epochs: int = 50, warmup_epochs: int = 5,
                  min_lr: float = 1e-6) -> torch.optim.lr_scheduler.LRScheduler:
    """
    Linear warm-up -> cosine annealing.

    Why a scheduler?
    ----------------
    A fixed learning rate is often sub-optimal:
    - Warm-up: start with a small LR to stabilize early training when random weights produce large, noisy gradients.
    - Cosine decay: gradually reduce LR so the model fine-tunes smoothly towards a min rather than bouncing around it.

    SequentialLR chains:
      epochs 1 to warmup_epochs: LinearLR (ramps from min_lr to lr)
      epochs warmup_epochs+1 to end: CosineAnnealingLR (decays to min_lr)
    """
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=min_lr / max(optimizer.param_groups[0]["lr"], 1e-12),
        end_factor=1.0,
        total_iters=warmup_epochs,
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(num_epochs - warmup_epochs, 1),
        eta_min=min_lr,
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[warmup_epochs],
    )


# =============================================================================
# Checkpoint helpers
# =============================================================================

def save_checkpoint(state: dict, save_dir: str, filename: str) -> None:
    """Persist a checkpoint dict to disk."""
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    path = os.path.join(save_dir, filename)
    torch.save(state, path)
    print(f"[ckpt] saved -> '{path}'")


def load_checkpoint(path: str, model: nn.Module, optimizer: Optional[torch.optim.Optimizer] = None, scheduler=None,
                    early_stopping: Optional[EarlyStopping] = None, device: str = "cpu") -> tuple[int, float, dict]:
    """Restore weights (and optionally optimizer/scheduler/early-stopping) from disk."""
    print(f"[ckpt] loading from '{path}'")
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    if optimizer and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler and "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    if early_stopping and "early_stopping_state" in ckpt:
        early_stopping.load_state_dict(ckpt["early_stopping_state"])

    epoch = ckpt.get("epoch", 0)
    best_val_loss = ckpt.get("best_val_loss", inf)
    history = ckpt.get("history", None)
    print(f"[ckpt] resumed from epoch {epoch}, best_val_loss={best_val_loss:.6f}")
    return epoch, best_val_loss, history


# =============================================================================
# train_one_epoch
# =============================================================================

def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, device: str, mode: str,
                    bce_lambda: float = 0.5, grad_clip: float = 1.0) -> dict:
    """
    One full pass over the training DataLoader (a single training epoch).

    Standard PyTorch training step (per batch):
      1. optimizer.zero_grad()	: clear previous gradients
      2. out = model(X)			: forward pass: compute predictions
      3. loss = criterion(out,y): compute loss: MSE or MSE + λ·BCE for turning_point
      4. loss.backward()		: backpropagate gradients
      5. clip_grad_norm_()		: prevent exploding gradients in RNNs
      6. optimizer.step()		: update weights

    Args:
        model: Model instance used for training
        loader: DataLoader instance
        optimizer: Optimizer instance
        device: Device to run on (e.g., "cpu" or "cuda")
        mode: "exact" | "rolling" | "turning_point"
        bce_lambda: weight of BCE in the combined turning-point loss
        grad_clip: global gradient norm clip value (0 = disabled)

    Returns:
        dict with "loss" (and "bce_loss" / "mse_loss" for turning_point mode)
    """
    model.train()
    mse_criterion = nn.MSELoss()
    # Heavily penalize the model for missing a "Buy" signal
    pos_weight = torch.tensor([10.0], device=device)
    bce_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)     # only used in turning_point mode

    loss_meter = AverageMeter("train_loss")
    mse_sub_meter = AverageMeter("train_mse")
    bce_sub_meter = AverageMeter("train_bce")

    for batch in loader:
        X = batch[0].to(device)  # (batch, T, F)
        B = X.size(0)
        optimizer.zero_grad(set_to_none=True)
        # set_to_none=True is slightly faster than zeroing tensors:
        # it frees the gradient memory rather than filling it with zeros.

        # Forward + loss
        if mode in ("exact", "rolling"):
            y_true = batch[1].to(device)  # (batch, D)
            y_pred = model(X)  # (batch, D)
            loss = mse_criterion(y_pred, y_true)
            loss_meter.update(loss.item(), n=B)

        else:  # turning_point
            y_ret_true = batch[1].to(device)  # (batch, D)
            y_lbl_true = batch[2].to(device).unsqueeze(1)  # (batch, 1)
            y_ret_pred, y_lbl_pred = model(X)  # (batch,D), (batch,1)

            mse_loss = mse_criterion(y_ret_pred, y_ret_true)
            bce_loss = bce_criterion(y_lbl_pred, y_lbl_true)
            loss = mse_loss + bce_lambda * bce_loss  # combined loss

            loss_meter.update(loss.item(), n=B)
            mse_sub_meter.update(mse_loss.item(), n=B)
            bce_sub_meter.update(bce_loss.item(), n=B)

        # Backward + step
        loss.backward()

        # Gradient clipping: caps the global gradient norm at 1.0.
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()

    result = {"loss": loss_meter.avg}
    if mode == "turning_point":
        result["mse_loss"] = mse_sub_meter.avg
        result["bce_loss"] = bce_sub_meter.avg
    return result


# =============================================================================
# validate
# =============================================================================

@torch.no_grad()
def validate(model: nn.Module, loader: DataLoader, device: str, mode: str, bce_lambda: float = 0.5) -> dict:
    """
    Validation pass: No gradients are computed.

    Returns a dict with "loss", "rmse", and (for turning_point) "mse_loss", "bce_loss", "accuracy", "precision",
    "recall", "f1".
    """
    model.eval()  # disables dropout; uses running stats in batch-norm

    mse_criterion = nn.MSELoss()
    pos_weight = torch.tensor([10.0], device=device)
    bce_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    loss_meter = AverageMeter("val_loss")
    mse_meter = AverageMeter("val_mse")
    bce_meter = AverageMeter("val_bce")

    # Initialize the stateful tracker if we are in turning_point mode
    binary_tracker = BinaryMetrics() if mode == "turning_point" else None

    for batch in loader:
        X = batch[0].to(device)
        B = X.size(0)

        if mode in ("exact", "rolling"):
            y_true = batch[1].to(device)
            y_pred = model(X)
            loss = mse_criterion(y_pred, y_true)
            loss_meter.update(loss.item(), n=B)
            mse_meter.update(loss.item(), n=B)

        else:  # turning_point
            y_ret_true = batch[1].to(device)
            y_lbl_true = batch[2].to(device).unsqueeze(1)

            y_ret_pred, y_lbl_pred = model(X)

            mse_loss = mse_criterion(y_ret_pred, y_ret_true)
            bce_loss = bce_criterion(y_lbl_pred, y_lbl_true)
            loss = mse_loss + bce_lambda * bce_loss

            loss_meter.update(loss.item(), n=B)
            mse_meter.update(mse_loss.item(), n=B)
            bce_meter.update(bce_loss.item(), n=B)
            binary_tracker.update(torch.sigmoid(y_lbl_pred), y_lbl_true)

    result = {
        "loss": loss_meter.avg,
        "rmse": mse_meter.avg ** 0.5  # RMSE is in the same units as the return ratio
    }

    if mode == "turning_point":
        result["mse_loss"] = mse_meter.avg
        result["bce_loss"] = bce_meter.avg

        # Compute binary classification metrics
        metrics = binary_tracker.compute()
        result.update({k: metrics[k] for k in ["accuracy", "precision", "recall", "f1"]})

    return result


# =============================================================================
# run_training  — full training loop
# =============================================================================
def run_training(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, mode: str, run_name: str = "run",
                 num_epochs: int = 50, lr: float = 1e-3, weight_decay: float = 1e-4, device: str = "cpu",
                 bce_lambda: float = 0.5, patience: int = 10, checkpoint_dir: str = "checkpoints",
                 grad_clip: float = 1.0, warmup_epochs: int = 5, min_lr: float = 1e-6,
                 resume: Optional[str] = None) -> tuple[nn.Module, dict]:
    """
    Full training loop.

    Args:
        model: uninitialized model (moved to device here)
        train_loader: DataLoader for the training split
        val_loader: DataLoader for the validation split
        mode: "exact" | "rolling" | "turning_point"
        run_name: checkpoint subdirectory name
        num_epochs: Number of training epochs
        device: device to run on
        lr: learning rate
        weight_decay: weight decay (L2 regularization)
        bce_lambda: weight of BCE in the turning-point combined loss
        patience: early-stopping patience (0 = disabled)
        checkpoint_dir: root directory for saved checkpoints
        grad_clip: gradient norm clip value
        warmup_epochs: linear warm-up length before cosine annealing
        min_lr: LR floor for cosine scheduler
        resume: path to a .pt checkpoint to resume from (or None)

    Returns:
        (best_model, history_dict): best_model has the weights from the epoch with the lowest val loss.
                                    history_dict contains lists of per-epoch metrics for plotting.
    """
    model = model.to(device)
    optimizer = get_optimizer(model, lr=lr, weight_decay=weight_decay)
    scheduler = get_scheduler(optimizer, num_epochs=num_epochs, warmup_epochs=warmup_epochs, min_lr=min_lr)
    early_stop = EarlyStopping(patience=patience)
    save_dir = os.path.join(checkpoint_dir, run_name)

    history: dict[str, list] = {
        "train_loss": [], "val_loss": [], "val_rmse": [], "lr": []
    }
    if mode == "turning_point":
        history.update({"val_accuracy": [], "val_precision": [], "val_recall": [], "val_f1": []})

    best_val_loss = inf
    best_weights = None
    start_epoch = 0

    # Resume from checkpoint
    if resume and os.path.exists(resume):
        start_epoch, best_val_loss, saved_history = load_checkpoint(
            resume, model, optimizer, scheduler, early_stop, device
        )
        if saved_history:
            history = saved_history

    for epoch in range(start_epoch + 1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")

        # Train
        train_metrics = train_one_epoch(model, train_loader, optimizer, device, mode, bce_lambda, grad_clip)

        # Validate
        val_metrics = validate(model, val_loader, device, mode, bce_lambda)

        # LR step
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        # Log
        print(f"train_loss={train_metrics['loss']:.6f}, val_loss={val_metrics['loss']:.6f}, "
              f"val_rmse={val_metrics['rmse']:.6f}, lr={current_lr:.2e}")
        if mode == "turning_point":
            print(f"Validation: Accuracy={val_metrics.get('accuracy', 0):.4f}, "
                  f"Precision={val_metrics.get('precision', 0):.4f}, "
                  f"Recall={val_metrics.get('recall', 0):.4f}, "
                  f"F1={val_metrics.get('f1', 0):.4f}")

        # Record history
        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_rmse"].append(val_metrics["rmse"])
        history["lr"].append(current_lr)

        if mode == "turning_point":
            history["val_accuracy"].append(val_metrics.get("accuracy", 0))
            history["val_precision"].append(val_metrics.get("precision", 0))
            history["val_recall"].append(val_metrics.get("recall", 0))
            history["val_f1"].append(val_metrics.get("f1", 0))

        # Save best checkpoint
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_weights = copy.deepcopy(model.state_dict())
            save_checkpoint(
                state={
                    "epoch": epoch,
                    "model_state_dict": best_weights,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "early_stopping_state": early_stop.state_dict(),
                    "best_val_loss": best_val_loss,
                    "history": history,
                    "mode": mode,
                },
                save_dir=save_dir,
                filename=f"best_{mode}.pt",
            )
            print(f"Saved best model (validation_loss={best_val_loss:.6f})")

        # Early stopping
        if patience > 0:
            early_stop.step(val_metrics["loss"])
            if early_stop.stop:
                print(f"Early stopping triggered. Epoch {epoch - patience} had the lowest validation loss "
                      f"({best_val_loss:.6f}).")
                break

    # Restore best weights before returning
    if best_weights is not None:
        model.load_state_dict(best_weights)
        print(f"\nTraining complete. Best val_loss={best_val_loss:.6f}")

    return model, history
