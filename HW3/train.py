import copy
from math import inf
from typing import Tuple
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from tqdm import tqdm
import logging
from utils import EarlyStopping, plot_learning_curves, ClassificationMetrics, measure_runtime
from data_loaders import AugMixDataset, build_augmix_transforms


logger = logging.getLogger("HW3")


def kd_loss(student_logits, teacher_logits, temperature):
    """
    Standard Knowledge Distillation loss using KL Divergence.
    The student is trained to match the soft probability distribution produced by the teacher at temperature T.
    Scales by T^2 to keep gradient magnitude consistent with the CE term as temperature increases

    Args:
        student_logits (torch.Tensor): Raw logits from the student.
        teacher_logits (torch.Tensor): Raw logits from the teacher.
        temperature (float): Softening temperature T > 1 spreads the teacher distribution, making it easier for the
                             student to learn from.

    Returns:
        KD loss (torch.Tensor): Scalar KD loss value.
    """
    student_soft = F.log_softmax(student_logits / temperature, dim=1)
    teacher_soft = F.softmax(teacher_logits / temperature, dim=1)

    return F.kl_div(student_soft, teacher_soft, reduction='batchmean') * (temperature * temperature)


def custom_kd_targets(teacher_logits, labels, temperature):
    """
    Custom KD mode.
    Teacher-guided label smoothing: keep teacher confidence on the true class, spread remaining probability
    uniformly across all other classes.

    The teacher's confidence on the true class is kept as-is, the remaining probability mass is distributed uniformly
    across all other classes.

    Args:
        teacher_logits (torch.Tensor): Teacher output logits (before softmax)
        labels (torch.Tensor): Ground-truth integer labels
        temperature (float): Temperature for softening teacher probs.

    Returns:
        torch.Tensor: Soft target probability distribution sums to 1 along dim 1.
    """
    probs = F.softmax(teacher_logits / temperature, dim=1)

    batch_size, num_classes = probs.shape
    new_targets = torch.zeros_like(probs)

    # Get teacher confidence for true class
    true_class_probs = probs[torch.arange(batch_size), labels]

    # Assign true class probability
    new_targets[torch.arange(batch_size), labels] = true_class_probs

    # Distribute remaining probability uniformly
    remaining = (1.0 - true_class_probs) / (num_classes - 1)

    for i in range(batch_size):
        for j in range(num_classes):
            if j != labels[i]:
                new_targets[i, j] = remaining[i]

    return new_targets


def custom_kd_loss(student_logits, teacher_logits, labels, temperature):
    """
    Custom KD loss using teacher-guided label smoothing targets.

    Args:
        student_logits (torch.Tensor): Student logits.
        teacher_logits (torch.Tensor): Teacher logits.
        labels (torch.Tensor): Ground-truth labels.
        temperature (float): Softening temperature.

    Returns:
        torch.Tensor: Scalar loss value.
    """
    student_log_probs = F.log_softmax(student_logits / temperature, dim=1)
    target_probs = custom_kd_targets(teacher_logits, labels, temperature)
    return F.kl_div(student_log_probs, target_probs, reduction='batchmean') * (temperature * temperature)


def jsd_loss(p_clean, p_aug1, p_aug2) -> torch.Tensor:
    """
    Three-way (Across three views) Jensen-Shannon Divergence consistency loss.

    Used in the full AugMix objective to encourage the model to produce consistent predictions for a clean image
    and two independently augmented versions of the same image.
    Measures prediction inconsistency across a clean image and two AugMix views.

    Defined in the AugMix paper as:
        M   = (p_clean + p_aug1 + p_aug2) / 3
        JSD = (1/3) * [KL(p_clean||M) + KL(p_aug1||M) + KL(p_aug2||M)]

    --------------------------------
    F.kl_div(input, target) computes KL(target || input), i.e. it expects
    input=log(Q) and target=P and returns sum(P * (log P - log Q)).

    To compute KL(p || M) we therefore pass input=log(M), target=p:
        F.kl_div(M.log(), p, reduction="batchmean")
        -> sum(p * (log p - log M))  =  KL(p || M)  (correct)

    Args:
        p_clean (torch.Tensor): Softmax probabilities for clean images.
        p_aug1  (torch.Tensor): Softmax probabilities for AugMix view 1.
        p_aug2  (torch.Tensor): Softmax probabilities for AugMix view 2.

    Returns:
        torch.Tensor: Scalar mean JSD value in [0, log(3)].
    """
    M = (p_clean + p_aug1 + p_aug2) / 3.0
    log_M = torch.log(M + 1e-8)

    kl = lambda p: F.kl_div(log_M, p, reduction="batchmean")
    return (kl(p_clean) + kl(p_aug1) + kl(p_aug2)) / 3.0


def get_optimizer(model, params) -> torch.optim.Optimizer:
    """
    Instantiate and return the optimizer specified by params["optimizer"].

    Only parameters with requires_grad=True are passed to the optimizer, so frozen layers (e.g. in transfer-learning
    or KD setups) are excluded.

    Args:
        model (nn.Module): Model whose trainable parameters are optimized.
        params (dict): Configuration dictionary.

    Returns:
        torch.optim.Optimizer: Configured optimizer instance.

    Raises:
        ValueError: If params["optimizer"] is not one of {"adam", "sgd", "adamw", "nadam"}.
    """
    trainable = filter(lambda p: p.requires_grad, model.parameters())

    if params["optimizer"] == "adam":
        optimizer = torch.optim.Adam(
            trainable,
            lr=params["learning_rate"],
            weight_decay=params["weight_decay"]
        )

    elif params["optimizer"] == "sgd":
        optimizer = torch.optim.SGD(
            trainable,
            lr=params["learning_rate"],
            momentum=0.9,
            weight_decay=params["weight_decay"]
        )

    elif params["optimizer"] == "adamw":
        optimizer = torch.optim.AdamW(
            trainable,
            lr=params["learning_rate"],
            weight_decay=params["weight_decay"]
        )

    elif params["optimizer"] == "nadam":
        optimizer = torch.optim.NAdam(
            trainable,
            lr=params["learning_rate"],
            weight_decay=params["weight_decay"]
        )

    else:
        raise ValueError("Unknown optimizer!")

    return optimizer


def get_transforms(params, train=True) -> transforms.Compose:
    """
    Returns torchvision transform pipeline for training or testing data.

    When AugMix is enabled, transforms.AugMix is inserted before ToTensor (it expects a PIL image).

    For validation / test only ToTensor + Normalize are applied (no stochastic augmentation).

    Args:
        params (dict): Configuration parameters.
        train (bool): If True, include training-time augmentations. If False, return deterministic transforms only.

    Returns:
        transforms.Compose: Composed transformations.
    """
    mean, std = params["mean"], params["std"]

    # Check if we need resizing (for pretrained models like VGG)
    resize = params.get("resize_input", False)

    transform_list = []

    # Resize if required (for pretrained models on ImageNet)
    if resize:
        transform_list.append(transforms.Resize((224, 224)))

    if params["dataset"] == "mnist":
        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    elif params["dataset"] == "cifar10":
        if train:
            transform_list.extend([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
            ])

            # AugMix expects a PIL image, so it goes before ToTensor
            if params["enable_augmix"]:
                transform_list.append(
                    transforms.AugMix(
                        severity=params.get("augmix_severity", 3),
                        mixture_width=params.get("augmix_mixture_width", 3),
                        chain_depth=-1
                    )
                )
                logger.info(f"AugMix enabled where severity={params['augmix_severity']} and "
                            f"mixture_width={params['augmix_mixture_width']}")

        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:
        raise ValueError(f"Unsupported dataset: {params['dataset']}")

    return transforms.Compose(transform_list)


def get_loaders(params):
    """
    Build and return train and validation DataLoaders by splitting the training set into train and validation.
    Importantly, separate transform pipelines are applied: the train split gets augmentation while
    the val split gets only normalization, avoiding data-leakage in validation metrics.

    Args:
        params (dict): Configuration dictionary.

    Returns:
        Tuple[DataLoader, DataLoader]: (train_loader, val_loader).

    Raises:
        ValueError: If params["dataset"] is not supported.
    """

    def make_generator():
        """Generator seed ensures train/val indices are identical across the two dataset objects (raw and
        val-transformed)"""
        return torch.Generator().manual_seed(42)

    TRAIN_RATIO = 0.83

    if params["dataset"] == "mnist":
        tf = get_transforms(params)
        # # IMPORTANT: This is wrong approach to use the test set as a validation set. This would lead to data leakage.
        # train_ds = datasets.MNIST(params["data_dir"], train=True,  download=True, transform=tf)
        # val_ds   = datasets.MNIST(params["data_dir"], train=False, download=True, transform=tf)

        full_train = datasets.MNIST(params["data_dir"], train=True, download=True, transform=tf)    # 60000 Data point

        train_size = int(TRAIN_RATIO * len(full_train))    # Approx. 50000 (MNIST) Data point
        val_size = len(full_train) - train_size     # Approx. 10000 (MNIST) Data point

        # random_split randomly divides the dataset while preserving  the dataset object.
        train_ds, val_ds = random_split(full_train, [train_size, val_size])
    elif params["dataset"] == "cifar10":
        full_size = len(datasets.CIFAR10(params["data_dir"], train=True, download=True))
        train_size = int(TRAIN_RATIO * full_size)    # Approx. 41500 (CIFAR-10) Data point
        val_size = full_size - train_size     # Approx. 8500  (CIFAR-10) Data point

        if params["enable_augmix_jsd"]:
            # Full AugMix path (Computing JSD): AugMix requires to load raw PIL images with no transform
            base_raw = datasets.CIFAR10(
                params["data_dir"], train=True, download=True, transform=None
            )
            train_subset_raw, _ = random_split(base_raw, [train_size, val_size], generator=make_generator())

            preprocess, augmix_preprocess = build_augmix_transforms(
                mean=params["mean"],
                std=params["std"],
                severity=params.get("augmix_severity", 3),
                mixture_width=params.get("augmix_mixture_width", 3),
            )
            train_ds = AugMixDataset(train_subset_raw, preprocess, augmix_preprocess)
            logger.info(f"AugMix enabled where severity={params['augmix_severity']} and "
                        f"mixture_width={params['augmix_mixture_width']}")
        else:
            # Standard path: transformed dataset, same split indices
            train_tf = get_transforms(params, train=True)
            full_train = datasets.CIFAR10(params["data_dir"], train=True, download=True, transform=train_tf)
            train_ds, _ = random_split(full_train, [train_size, val_size], generator=make_generator())

        # Validation always uses clean transforms
        val_tf = get_transforms(params, train=False)
        full_val = datasets.CIFAR10(params["data_dir"], train=True, download=True, transform=val_tf)
        _, val_ds = random_split(full_val, [train_size, val_size], generator=make_generator())
    else:
        raise ValueError(f"Unsupported dataset: {params['dataset']}")

    train_loader = DataLoader(train_ds, batch_size=params["batch_size"], shuffle=True,
                              num_workers=params["num_workers"])
    val_loader = DataLoader(val_ds, batch_size=params["batch_size"], shuffle=False, num_workers=params["num_workers"])
    return train_loader, val_loader


def train_one_epoch(model, loader, optimizer, criterion, device, params, teacher_model=None):
    """
    Run one full training epoch.

    Batch format detection
    ----------------------
    The function inspects len(batch) at runtime:

        len(batch) == 2  ->  standard (imgs, labels)
        len(batch) == 4  ->  Full AugMix with JSD (clean, aug1, aug2, labels)

    This avoids extra flags and keeps the DataLoader as the single source of truth about which mode is active.

    Loss composition
    ----------------
        loss = CE_or_CutMix
             + (optional) KD term (if teacher_model is provided and Knowledge distillation is enabled)
             + (optional) AugMix JSD term (if Full AugMix with JSD is enabled)
             + (optional) L1 regularization

    The JSD term reuses the clean forward pass output (detached) to avoid an extra forward pass.

    Args:
        model (nn.Module): Model in training mode.
        loader (DataLoader): Train loader (2-tuple or 4-tuple batches).
        optimizer (Optimizer): Configured optimizer.
        criterion (nn.Module): Base Cross-Entropy loss.
        device (torch.device): Compute device.
        params (dict): Configuration dictionary.
        teacher_model (optional): Frozen teacher for knowledge distillation.

    Returns:
        Tuple[float, float]: (mean_loss, accuracy) over the epoch.
    """
    model.train()
    total_loss, correct, n = 0.0, 0, 0

    use_kd = params.get("enable_kd") and teacher_model is not None

    for batch_idx, batch in enumerate(tqdm(loader, desc="Training")):
        # 4-tuple for AugMix+JSD, 2-tuple otherwise
        if len(batch) == 4:
            imgs_clean, imgs_aug1, imgs_aug2, labels = batch
            imgs_clean = imgs_clean.to(device)
            imgs_aug1 = imgs_aug1.to(device)
            imgs_aug2 = imgs_aug2.to(device)
            labels = labels.to(device)
            use_jsd = True
        else:
            imgs_clean, labels = batch
            imgs_clean = imgs_clean.to(device)
            labels = labels.to(device)
            imgs_aug1 = imgs_aug2 = None
            use_jsd = False

        optimizer.zero_grad()

        # Forward pass
        out = model(imgs_clean)

        # Standard CE loss
        ce_loss = criterion(out, labels)

        if use_kd:
            with torch.no_grad():
                teacher_out = teacher_model(imgs_clean)

            if params["kd_mode"] == "standard":
                kd = kd_loss(out, teacher_out, params["kd_temperature"])
            elif params["kd_mode"] == "custom":
                kd = custom_kd_loss(out, teacher_out, labels, params["kd_temperature"])
            else:
                raise ValueError(f"Unknown KD mode: {params['kd_mode']}")

            loss = params["kd_alpha"] * ce_loss + (1 - params["kd_alpha"]) * kd
        else:
            loss = ce_loss

        # AugMix JSD Consistency
        if use_jsd:
            with torch.no_grad():
                p_aug1 = F.softmax(model(imgs_aug1), dim=1)
                p_aug2 = F.softmax(model(imgs_aug2), dim=1)
            p_clean = F.softmax(out.detach(), dim=1)
            jsd_term = jsd_loss(p_clean, p_aug1, p_aug2)
            loss = loss + params["jsd_lambda"] * jsd_term

        # L1 Regularization
        if params["l1_lambda"] > 0:
            l1_norm = sum(p.abs().sum() for p in model.parameters())
            loss = loss + params["l1_lambda"] * l1_norm

        loss.backward()
        optimizer.step()

        total_loss += loss.detach().item() * imgs_clean.size(0)
        correct += out.argmax(1).eq(labels).sum().item()
        n += imgs_clean.size(0)

        # if (batch_idx + 1) % params["log_interval"] == 0:
        #     print(f"  [{batch_idx+1}/{len(loader)}] "
        #           f"Training Loss: {total_loss/n:.4f} - Training Accuracy: {correct/n:.4f}")

    return total_loss / n, correct / n


def validate(model, loader, criterion, device, params) -> Tuple[float, float]:
    """
    Evaluate the model on the validation set.

    No augmentation is applied; only CE loss and accuracy are reported.
    Macro precision, recall, and F1 are logged via ClassificationMetrics.

    Args:
        model (nn.Module): Model to evaluate.
        loader (DataLoader): Validation DataLoader.
        criterion (nn.Module): Loss function (same as training for consistency).
        device (torch.device): Compute device.
        params (dict): Configuration dictionary.

    Returns:
        Tuple[float, float]: (mean_val_loss, val_accuracy).
    """
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    val_metrics = ClassificationMetrics(params["num_classes"], device)
    with torch.no_grad():
        for batch in loader:
            imgs, labels = batch[0], batch[-1]
            imgs, labels = imgs.to(device), labels.to(device)
            out = model(imgs)

            loss = criterion(out, labels)
            total_loss += loss.detach().item() * imgs.size(0)
            correct += out.argmax(1).eq(labels).sum().item()
            n += imgs.size(0)
            preds = torch.argmax(out, dim=1)
            val_metrics.update(preds, labels)

    results = val_metrics.compute()

    logger.info("========= Validation Metrics =========")
    logger.info(f"=> Accuracy:  {(correct / n):.4f} ({correct}/{n}), Precision: {results['precision']:.4f}, "
                f"Recall: {results['recall']:.4f}, F1 Score: {results['f1']:.4f}")
    logger.info("======================================")
    return total_loss / n, correct / n


@measure_runtime
def run_training(model, params, device, teacher_model=None):
    train_loader, val_loader = get_loaders(params)

    criterion = nn.CrossEntropyLoss(label_smoothing=params["label_smoothing"])
    optimizer = get_optimizer(model, params)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    early_stopping = EarlyStopping(patience=params["patience"]) if params["enable_early_stopping"] else None

    best_loss = inf
    best_acc = 0.0
    best_weights = None

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": []
    }

    for epoch in range(1, params["epochs"] + 1):
        logger.info(f"\nEpoch {epoch}/{params['epochs']}")
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer,
                                          criterion, device, params, teacher_model)
        logger.info(f"=> Training loss:   {tr_loss:.4f} - Training Accuracy:   {tr_acc:.4f}")
        val_loss, val_acc = validate(model, val_loader, criterion, device, params)
        logger.info(f"=> Validation loss: {val_loss:.4f} - Validation Accuracy: {val_acc:.4f}")
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(val_acc)

        scheduler.step()

        if val_loss < best_loss:
            best_loss = val_loss
            best_acc = val_acc
            best_weights = copy.deepcopy(model.state_dict())    # snapshot in memory
            torch.save(best_weights, params["save_path"])       # persist to disk
            logger.info(f"Saved best model (validation_loss={best_loss:.4f})")

        if params["enable_early_stopping"]:
            early_stopping.step(val_loss)

            if early_stopping.stop:
                logger.warning(f"Early stopping triggered. Epoch {epoch - params["patience"]} had the lowest "
                               f"validation loss ({best_loss}).")
                break

    # Restore best weights into the model before returning
    model.load_state_dict(best_weights)
    logger.info(f"\nTraining done. Best validation accuracy: {best_acc:.4f}")

    plot_learning_curves(history)
