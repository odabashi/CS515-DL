import argparse


def get_params():
    parser = argparse.ArgumentParser(description="Deep Learning on MNIST / CIFAR-10")

    parser.add_argument("--mode",       choices=["train", "test", "both"], default="both")
    parser.add_argument("--device",     choices=["cpu", "cuda"], type=str,   default="cuda")
    parser.add_argument("--dataset",    choices=["mnist", "cifar10"], default="cifar10")
    parser.add_argument("--model",      choices=["mlp", "vgg", "resnet", "mobilenet"], default="resnet")
    parser.add_argument("--teacher_model", choices=["mlp", "vgg", "resnet", "mobilenet"], default="resnet")

    parser.add_argument("--epochs",     type=int,   default=30)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int,   default=64)
    parser.add_argument("--hidden_sizes", type=int, nargs="*", default=[512, 256, 128])
    parser.add_argument("--hidden_activation", type=str, choices=["relu", "gelu", "leaky_relu", "elu", "tanh"],
                        default="relu")

    # --no-enable_early_stopping, # --enable_early_stopping
    parser.add_argument('--enable_early_stopping', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--patience",       type=int, default=5)

    # --no-enable_dropout, # --enable_dropout
    parser.add_argument('--enable_dropout', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dropout",        type=float, default=0.2)

    # --no-enable_batch_norm, # --enable_batch_norm
    parser.add_argument('--enable_batch_norm', action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--optimizer", type=str, choices=["sgd", "adam", "adamw"], default="adamw")
    parser.add_argument("--l1_lambda", type=float, default=0.0)
    parser.add_argument("--weight_decay", type=float, default=0.0)

    # VGG-specific
    parser.add_argument("--vgg_depth", choices=["11", "13", "16", "19"], default="16")

    # ResNet-specific: map a simple int to a block config
    parser.add_argument("--resnet_layers", type=int, nargs=4, default=[2, 2, 2, 2],
                        metavar=("L1", "L2", "L3", "L4"),
                        help="Number of blocks per ResNet layer (default: 2 2 2 2 = ResNet-18)")

    parser.add_argument('--plot_tsne', action=argparse.BooleanOptionalAction, default=True)

    # Transfer Learning
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--freeze_features", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--resize_input", action=argparse.BooleanOptionalAction, default=False)

    # Knowledge Distillation & Label Smoothing
    parser.add_argument("--label_smoothing", type=float, default=0.0,
                        help="Epsilon value for label smoothing (0.0 = disabled)")
    parser.add_argument("--enable_kd", action=argparse.BooleanOptionalAction, default=False,
                        help="Enable knowledge distillation")
    parser.add_argument("--teacher_model_path", type=str, default="teacher.pth",
                        help="Path to pretrained teacher model")
    parser.add_argument("--kd_temperature", type=float, default=4.0,
                        help="Temperature for KD softening")
    parser.add_argument("--kd_alpha", type=float, default=0.5,
                        help="Weight between CE loss and KD loss")
    parser.add_argument("--kd_mode", choices=["standard", "custom"], default="standard",
                        help="standard: normal KD | custom: teacher-guided label smoothing")

    args = parser.parse_args()

    # Dataset-dependent settings
    if args.dataset == "mnist":
        input_size = 784          # 1 × 28 × 28
        mean, std = (0.1307,), (0.3081,)
        num_classes = 10
    elif args.dataset == "cifar10":
        input_size = 3072         # 3 × 32 × 32
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
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
        "vgg_depth":                args.vgg_depth,
        "resnet_layers":            args.resnet_layers,

        # Transfer Learning
        "pretrained":               args.pretrained,
        "freeze_features":          args.freeze_features,
        "resize_input":             args.resize_input,

        # Knowledge distillation & Label smoothing
        "teacher_model": args.teacher_model,
        "label_smoothing": args.label_smoothing,
        "enable_kd": args.enable_kd,
        "teacher_model_path": args.teacher_model_path,
        "kd_temperature": args.kd_temperature,
        "kd_alpha": args.kd_alpha,
        "kd_mode": args.kd_mode,

        # Training
        "epochs":                   args.epochs,
        "batch_size":               args.batch_size,
        "learning_rate":            args.lr,
        "enable_early_stopping":    args.enable_early_stopping,
        "patience":                 args.patience,
        "optimizer":                args.optimizer,
        "l1_lambda":                args.l1_lambda,
        "weight_decay":             args.weight_decay,

        # Misc
        "seed":                     42,
        "device":                   args.device,
        "save_path":                "best_model.pth",
        "log_interval":             100,                # print every N batches
        "plot_tsne":                args.plot_tsne,

        # CLI
        "mode":                     args.mode,
    }
