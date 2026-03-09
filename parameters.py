import argparse


def get_params():
    parser = argparse.ArgumentParser(description="MLP on MNIST")

    parser.add_argument("--mode",       choices=["train", "test", "both"], default="both")
    parser.add_argument("--device",     choices=["cpu", "cuda"], type=str,   default="cuda")
    parser.add_argument("--dataset",    choices=["mnist"], default="mnist")
    parser.add_argument("--model",      choices=["mlp"], default="mlp")

    parser.add_argument("--epochs",     type=int,   default=30)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int,   default=64)
    parser.add_argument("--hidden_sizes", type=int, nargs="*", default=[512, 256, 128])
    parser.add_argument("--hidden_activation", type=str, choices=["relu", "gelu", "leaky_relu", "elu", "tanh"],
                        default="relu")

    # --no-enable_dropout, # --enable_dropout
    parser.add_argument('--enable_dropout', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--dropout", type=float, default=0.3)

    # --no-enable_early_stopping, # --enable_early_stopping
    parser.add_argument('--enable_early_stopping', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--patience",   type=int, default=5)

    # --no-enable_batch_norm, # --enable_batch_norm
    parser.add_argument('--enable_batch_norm', action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument("--optimizer", type=str, choices=["sgd", "adam", "adamw", "nadam", "rmsprop"], default="adam")
    parser.add_argument("--l1_lambda", type=float, default=0.0)

    args = parser.parse_args()

    # Dataset-dependent settings
    if args.dataset == "mnist":
        input_size = 784          # 1 × 28 × 28
        mean, std = (0.1307,), (0.3081,)
        num_classes = 10
    else:
        input_size, mean, std, num_classes = None, None, None, None

    return {
        # Data
        "dataset":                  args.dataset,
        "data_dir":                 "./data",
        "num_workers":              2,
        "mean":                     mean,
        "std":                      std,

        # Model
        "model":                    args.model,
        "input_size":               input_size,
        "hidden_sizes":             args.hidden_sizes,
        "num_classes":              num_classes,
        "hidden_activation":        args.hidden_activation,
        "enable_dropout":           args.enable_dropout,
        "dropout":                  args.dropout,
        "enable_batch_norm":        args.enable_batch_norm,

        # Training
        "epochs":                   args.epochs,
        "batch_size":               args.batch_size,
        "learning_rate":            args.lr,
        "enable_early_stopping":    args.enable_early_stopping,
        "patience":                 args.patience,
        "optimizer":                args.optimizer,
        "l1_lambda":                args.l1_lambda,
        "weight_decay":             1e-4,

        # Misc
        "seed":                     42,
        "device":                   args.device,
        "save_path":                "best_model.pth",
        "log_interval":             100,                # print every N batches

        # CLI
        "mode":                     args.mode,
    }
