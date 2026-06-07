"""
Structural constants + get_params() CLI configuration for the stock forecasting assignment.
Two-layer design:
Layer 1 -   Module-level constants (structural / architectural). 
            These are imported directly by dataset.py, evaluation.py, etc. and do NOT change per run.

Layer 2 -   get_params() with argparse.
            Returns a flat dict consumed exclusively by main.py. Tunable hyperparameters (lr, epochs, ...) live here 
            so they can be overridden from the CLI without editing source files.
"""
import argparse
import torch


# =============================================================================
# Device selection
# =============================================================================

def _select_device() -> str:
    """
    Best available compute device.  Priority: CUDA -> MPS (Apple Silicon) -> CPU.
    """
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# =============================================================================
# Layer 1 - Structural constants
# =============================================================================

# Data
FEATURES = ["Open", "High", "Low", "Close"]   # F = 4
START_DATE = "2020-01-01"
END_DATE = "2025-12-31"
TRAIN_END = "2024-07-31"    # train:   Jan 2020 – Jul 2024
VAL_END = "2024-12-31"      # val:     Aug 2024 – Dec 2024         # test:    Jan 2025 – Dec 2025
INPUT_SIZE = len(FEATURES)

# Sliding window / task
T = 20    # look-back window length (trading days)
D = 5     # number of forecast horizons  (d = 1 ... 5)

# Part (c) rolling average
ROLLING_WINDOW = 3                                            # l = 3
ROLLING_WEIGHTS = torch.tensor([1/3, 1/3, 1/3], dtype=torch.float32)

# Part (d) turning-point threshold 
GAMMA = 1.1


# =============================================================================
# Layer 2 - get_params()
# =============================================================================

def get_params() -> dict:
    """
    Parse CLI arguments and return a flat configuration dictionary.
    All structural constants (T, D, FEATURES, ...) are injected into the dict at the end so callers have a 
    single cfg object with everything.

    Returns:
        dict: flat configuration; keys map 1-to-1 to argparse dest names
    """
    parser = argparse.ArgumentParser(
        description="Stock Forecasting - LSTM / GRU deep-learning pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Run control
    parser.add_argument("--mode", choices=["train", "test", "both"], default="both", 
                        help="train only | test only (needs saved checkpoints) | both")
    parser.add_argument("--part", choices=["b", "c", "d", "all"], default="all", 
                        help="Which assignment part to run")
    parser.add_argument("--device", type=str, default=None, help="Compute device (cuda / mps / cpu).")
    parser.add_argument("--seed", type=int, default=42, help="Global random seed for reproducibility")

    # Data
    parser.add_argument("--tickers", type=str, nargs="+", default=["AAPL", "MSFT", "GOOGL"], 
                        help="S&P 500 ticker symbols to download")
    parser.add_argument("--data_dir", type=str, default="data", help="Directory for cached CSV files")

    # Model architecture
    parser.add_argument("--hidden_size", type=int, default=64, help="Hidden units per LSTM / GRU layer")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of stacked recurrent layers")
    parser.add_argument("--dropout", type=float, default=0.2, 
                        help="Dropout probability (applied between and after recurrent layers)")
    parser.add_argument("--no_ma_features", action="store_true", default=False, 
                        help="Disable the 1-D conv moving-average auxiliary features")

    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3, help="AdamW learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="AdamW L2 weight decay")
    parser.add_argument("--warmup_epochs", type=int, default=5,
                        help="Linear warm-up epochs before cosine decay")
    parser.add_argument("--min_lr", type=float, default=1e-6, 
                        help="Minimum LR at the end of cosine annealing")

    # Regularisation / stopping
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience (0 = disabled)")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Global gradient norm clip value")

    # Part (d) specific
    parser.add_argument("--bce_lambda", type=float, default=0.5,
                        help="Weight of the BCE term in the turning-point combined loss "
                             "(total = MSE + bce_lambda * BCE)")

    # Paths
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints",
                        help="Root directory for saved model checkpoints")
    parser.add_argument("--plots_dir", type=str, default="assets",
                        help="Root directory for saved figures")

    args = parser.parse_args()

    # input_size grows if MA features are enabled
    input_size = INPUT_SIZE + (0 if args.no_ma_features else 4)

    # Build and return flat cfg dict (mirrors reference return block) ----
    return {
        # Run control
        "mode": args.mode,
        "part": args.part,
        "device": args.device or _select_device(),
        "seed": args.seed,

        # Data
        "tickers": args.tickers,
        "data_dir": args.data_dir,
        "features": FEATURES,        # structural constant injected
        "start_date": START_DATE,
        "end_date": END_DATE,
        "train_end": TRAIN_END,
        "val_end": VAL_END,

        # Sliding window / task (structural constants injected)
        "T": T,
        "D": D,
        "rolling_window": ROLLING_WINDOW,
        "gamma": GAMMA,

        # Model architecture
        "input_size": input_size,
        "hidden_size": args.hidden_size,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "use_ma_features": not args.no_ma_features,

        # Training
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "warmup_epochs": args.warmup_epochs,
        "min_lr": args.min_lr,

        # Regularisation
        "patience": args.patience,
        "grad_clip": args.grad_clip,

        # Part (d)
        "bce_lambda": args.bce_lambda,

        # Paths
        "checkpoint_dir": args.checkpoint_dir,
        "plots_dir": args.plots_dir,
    }
