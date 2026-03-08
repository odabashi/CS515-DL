import argparse


def get_params():
    parser = argparse.ArgumentParser(description="MLP on MNIST")

    parser.add_argument("--mode",      choices=["train", "test", "both"], default="both")
    parser.add_argument("--dataset",   choices=["mnist"], default="mnist")
    parser.add_argument("--model",     choices=["mlp"], default="mlp")
    parser.add_argument("--epochs",    type=int,   default=10)
    parser.add_argument("--lr",        type=float, default=1e-3)
    parser.add_argument("--device",    choices=["cpu", "cuda"], type=str,   default="cpu")
    parser.add_argument("--batch_size", type=int,   default=64)

    args = parser.parse_args()

    # Dataset-dependent settings
    if args.dataset == "mnist":
        input_size = 784          # 1 × 28 × 28
        mean, std = (0.1307,), (0.3081,)
    else:
        input_size, mean, std = None, None, None

    return {
        # Data
        "dataset":      args.dataset,
        "data_dir":     "./data",
        "num_workers":  2,
        "mean":         mean,
        "std":          std,

        # Model
        "model":        args.model,
        "input_size":   input_size,
        "hidden_sizes": [512, 256, 128],
        "num_classes":  10,
        "dropout":      0.3,

        # Training
        "epochs":        args.epochs,
        "batch_size":    args.batch_size,
        "learning_rate": args.lr,
        "weight_decay":  1e-4,

        # Misc
        "seed":         42,
        "device":       args.device,
        "save_path":    "best_model.pth",
        "log_interval": 100,                # print every N batches

        # CLI
        "mode":         args.mode,
    }
