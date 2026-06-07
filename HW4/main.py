"""
main.py  -  Entry point for the stock-forecasting deep-learning assignment.

Functionalities:
    - set_seed()
    - _select_device()
    - main(): calls get_params() => flat cfg dict, then dispatches to run_part_b / run_part_c / run_part_d

Assignment parts
----------------
    b   StockLSTM and StockGRU - exact d-day return ratios
    c   Rolling-average targets - stability comparison with Part b
    d   Bidirectional LSTM/GRU - turning-point / buy-signal detection
    all Run b -> c -> d in sequence (default)

Run modes
---------
    train  Train models and save checkpoints; skip test evaluation.
    test   Load best checkpoints and test on the test set; skip training.
    both   Train then immediately test (default).

Usage examples
--------------
    python main.py                              # run all parts, train + test
    python main.py --part b                     # Part b only, train + test
    python main.py --part d --mode train        # Part d training only
    python main.py --part all --mode test       # evaluate all saved checkpoints
    python main.py --mode test                  # evaluate all checkpoints
    python main.py --epochs 30 --lr 5e-4        # override hyperparameters
    python main.py --tickers AAPL TSLA NVDA     # custom tickers
    python main.py --hidden_size 128 --num_layers 3
    python main.py --no_ma_features             # disable auxiliary features
"""
import os
import random
import sys
import numpy as np
import torch
from parameters import get_params
from models import build_model
from dataset import build_dataloaders
from train import run_training
from utils.evaluation import model_summary, print_results_table
from utils.visualization import (plot_training_dashboard, plot_all_learning_curves,
                                 plot_per_horizon_rmse, plot_exact_vs_rolling)
from test import run_test


sys.path.append(os.path.dirname(__file__))


# =============================================================================
# Reproducibility 
# =============================================================================

def set_seed(seed: int) -> None:
    """
    Fix all random seeds for reproducibility.
    Applies the same seed to Python's random, NumPy, PyTorch CPU, and all CUDA devices. Also sets cuDNN to
    deterministic mode (may slow training slightly - disable if speed is critical).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =============================================================================
# Part-specific helpers
# =============================================================================

def _train_and_test(arch: str, mode_str: str, cfg: dict) -> dict:
    """
    Build one model, train it (if cfg["mode"] in train/both), test it (if cfg["mode"] in test/both), 
    return the output dict.

    Args:
        arch: "lstm" | "gru" | "bilstm" | "bigru"
        mode_str: dataset target mode - "exact" | "rolling" | "turning_point"
        cfg: flat config dict from get_params()
    """
    run_name = f"{arch}_{mode_str}"
    plots_dir = os.path.join(cfg["plots_dir"], f"part_{mode_str[:1].upper()}")

    print(f"\n{'='*40}")
    print(f"{run_name.upper()}")
    print(f"{'='*40}")

    # Model - pass only the architectural keys it cares about
    model = build_model(
        arch,
        hidden_size=cfg["hidden_size"],
        num_layers=cfg["num_layers"],
        dropout=cfg["dropout"],
        use_ma_features=cfg["use_ma_features"],
    )
    model_summary(model, model_name=run_name)

    history = None

    # Train
    if cfg["mode"] in ("train", "both"):
        train_loader, val_loader, _ = build_dataloaders(mode=mode_str, batch_size=cfg["batch_size"])
        model, history = run_training(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            mode=mode_str,
            run_name=run_name,
            num_epochs=cfg["epochs"],
            lr=cfg["lr"],
            weight_decay=cfg["weight_decay"],
            device=cfg["device"],
            bce_lambda=cfg["bce_lambda"],
            patience=cfg["patience"],
            min_lr=cfg["min_lr"],
            warmup_epochs=cfg["warmup_epochs"],
        )
        if history:
            best_ep = int(np.argmin(history["val_loss"])) + 1
            plot_training_dashboard(
                history,
                model_name=run_name,
                best_epoch=best_ep,
                save_path=os.path.join(plots_dir, f"{run_name}_dashboard.png"),
            )

    # Test
    output = {}
    if cfg["mode"] in ("test", "both"):
        output = run_test(
            model=build_model(arch, hidden_size=cfg["hidden_size"], num_layers=cfg["num_layers"], 
                              dropout=cfg["dropout"], use_ma_features=cfg["use_ma_features"]),
            mode=mode_str,
            params=cfg,
            run_name=run_name,
            device=cfg["device"],
            plots_dir=plots_dir,
        )
        output["history"] = history

    return output


# =============================================================================
# Part b: exact d-day return ratios
# =============================================================================

def run_part_b(cfg: dict) -> dict[str, dict]:
    """StockLSTM and StockGRU on exact d-day return targets. Returns { run_name: test_output }."""
    print("\n" + "="*55)
    print("PART (b) - Exact return ratio forecasting")
    print("="*55)

    results = {}
    for arch in ("lstm", "gru"):
        results[f"{arch}_exact"] = _train_and_test(arch, "exact", cfg)

    # Grouped per-horizon RMSE comparison (LSTM vs GRU)
    if cfg["mode"] in ("test", "both"):
        reg_results = {k: v["reg_metrics"] for k, v in results.items() if "reg_metrics" in v}
        if reg_results:
            plot_per_horizon_rmse(
                reg_results,
                D=cfg["D"],
                title="Part (b) - Per-Horizon RMSE: LSTM vs GRU",
                save_path=os.path.join(cfg["plots_dir"], "part_B", "part_b_per_horizon_rmse.png"),
            )
            print_results_table(reg_results)

    return results


# =============================================================================
# Part c: rolling-average targets
# =============================================================================

def run_part_c(cfg: dict, part_b_results: dict | None = None) -> dict[str, dict]:
    """
    Train and evaluate StockLSTM and StockGRU on rolling-average return targets.
    Compares with Part b results if provided
    Returns { run_name: test_output }.
    """
    print("\n" + "="*55)
    print("PART (c) - Rolling-average return forecasting")
    print("="*55)

    results = {}
    for arch in ("lstm", "gru"):
        results[f"{arch}_rolling"] = _train_and_test(arch, "rolling", cfg)

    if cfg["mode"] in ("test", "both"):
        reg_results = {k: v["reg_metrics"] for k, v in results.items() if "reg_metrics" in v}
        if reg_results:
            # Rolling-only per-horizon comparison
            plot_per_horizon_rmse(
                reg_results,
                D=cfg["D"],
                title="Part (c) - Per-Horizon RMSE: Rolling Targets",
                save_path=os.path.join(cfg["plots_dir"], "part_C", "part_c_per_horizon_rmse.png"),
            )

        # Exact-vs-rolling comparison (uses LSTM as the representative model)
        if part_b_results:
            b_m = (part_b_results.get("lstm_exact") or {}).get("reg_metrics")
            c_m = (results.get("lstm_rolling") or {}).get("reg_metrics")
            if b_m and c_m:
                plot_exact_vs_rolling(
                    exact_metrics=b_m,
                    rolling_metrics=c_m,
                    D=cfg["D"],
                    model_name="StockLSTM",
                    save_path=os.path.join(cfg["plots_dir"], "part_C", "part_c_exact_vs_rolling.png"),
                )

        if reg_results:
            print_results_table(reg_results)

    return results


# =============================================================================
# Part d: turning-point / buy-signal detection
# =============================================================================

def run_part_d(cfg: dict) -> dict[str, dict]:
    """BiDirLSTM and BiDirGRU on the turning-point / buy-signal task."""
    print("\n" + "="*55)
    print("PART (d) - Turning-point detection (buy/pass signal)")
    print("="*55)

    results = {}
    for arch in ("bilstm", "bigru"):
        results[f"{arch}_turning"] = _train_and_test(arch, "turning_point", cfg)

    if cfg["mode"] in ("test", "both"):
        reg_results = {k: v["reg_metrics"] for k, v in results.items() if "reg_metrics" in v}
        if reg_results:
            # Per-horizon RMSE (max-price return predictions)
            plot_per_horizon_rmse(
                reg_results,
                D=cfg["D"],
                title="Part (d) - Per-Horizon RMSE: BiDir Models",
                save_path=os.path.join(cfg["plots_dir"], "part_D", "part_d_per_horizon_rmse.png"),
            )

        # Full table: regression + classification metrics together
        flat = {}
        for k, v in results.items():
            m = dict(v.get("reg_metrics", {}))
            m.update(v.get("cls_metrics", {}))
            if m:
                flat[k] = m
        if flat:
            print_results_table(flat)

    return results


def main() -> None:
    cfg = get_params()

    print("=" * 40)
    print("=" * 40)
    print(f"Part : {cfg['part']}")
    print(f"Mode : {cfg['mode']}")
    print(f"Device : {cfg['device']}")
    print(f"Seed : {cfg['seed']}")
    print(f"Tickers: {' '.join(cfg['tickers'])}")
    print(f"Epochs: {cfg['epochs']}  |  LR: {cfg['lr']:.0e}  |  Batch: {cfg['batch_size']}")
    print(f"Hidden size: {cfg['hidden_size']}  |  Layers: {cfg['num_layers']}  |  Dropout: {cfg['dropout']}")
    print(f"MA features: {cfg['use_ma_features']}  |  BCE λ: {cfg['bce_lambda']}")
    print(f"Patience : {cfg['patience']}  |  Grad clip: {cfg['grad_clip']}")
    print(f"Checkpoint dir: {cfg['checkpoint_dir']}")
    print(f"Plots dir: {cfg['plots_dir']}")
    print("=" * 40)

    # Reproducibility
    set_seed(cfg["seed"])

    # Dispatch
    all_results: dict[str, dict] = {}
    part_b_res: dict[str, dict] = {}
    parts = ["b", "c", "d"] if cfg["part"] == "all" else [cfg["part"]]

    for part in parts:
        if part == "b":
            part_b_res = run_part_b(cfg)
            all_results.update(part_b_res)

        elif part == "c":
            # Pass part_b_res for the exact-vs-rolling comparison plot.
            # Running --part c alone skips that comparison gracefully.
            res = run_part_c(cfg, part_b_results=part_b_res)
            all_results.update(res)

        elif part == "d":
            res = run_part_d(cfg)
            all_results.update(res)

    # Final summary
    if all_results and cfg["mode"] in ("test", "both"):
        print("\n" + "=" * 40)
        print("FINAL SUMMARY - ALL RUNS")
        print("=" * 40)

        # Flatten to a single table: reg metrics + cls metrics where available
        flat = {}
        for run_name, out in all_results.items():
            m = dict(out.get("reg_metrics", {}))
            m.update(out.get("cls_metrics", {}))
            if m:
                flat[run_name] = m
        if flat:
            print_results_table(flat)

        # All-curves overview for the report
        histories = {k: v["history"] for k, v in all_results.items() if v.get("history")}
        if histories:
            plot_all_learning_curves(histories, save_path=os.path.join(cfg["plots_dir"], "all_learning_curves.png"))


if __name__ == "__main__":
    main()
