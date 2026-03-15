import copy
from math import inf
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from tqdm import tqdm
import logging
from utils import EarlyStopping, plot_learning_curves, ClassificationMetrics


logger = logging.getLogger("HW2")


def get_optimizer(model, params) -> torch.optim.Optimizer:
    if params["optimizer"] == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=params["learning_rate"],
            weight_decay=params["weight_decay"]
        )

    elif params["optimizer"] == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=params["learning_rate"],
            momentum=0.9,
            weight_decay=params["weight_decay"]
        )

    elif params["optimizer"] == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=params["learning_rate"],
            weight_decay=params["weight_decay"]
        )

    elif params["optimizer"] == "nadam":
        optimizer = torch.optim.NAdam(
            model.parameters(),
            lr=params["learning_rate"],
            weight_decay=params["weight_decay"]
        )

    else:
        raise ValueError("Unknown optimizer!")

    return optimizer


def get_transforms(params):
    mean, std = params["mean"], params["std"]

    if params["dataset"] == "mnist":
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:
        return None


def get_loaders(params):
    tf = get_transforms(params)

    if params["dataset"] == "mnist":
        # # IMPORTANT: This is wrong approach to use the test set as a validation set. This would lead to data leakage.
        # train_ds = datasets.MNIST(params["data_dir"], train=True,  download=True, transform=tf)
        # val_ds   = datasets.MNIST(params["data_dir"], train=False, download=True, transform=tf)

        full_train = datasets.MNIST(params["data_dir"], train=True, download=True, transform=tf)    # 60000 Data point

        train_size = int(0.83 * len(full_train))    # Approx. 50000 Data point
        val_size = len(full_train) - train_size     # Approx. 10000 Data point

        # random_split randomly divides the dataset while preserving the dataset object.
        train_ds, val_ds = random_split(full_train, [train_size, val_size])

    else:
        train_ds, val_ds = None, None

    train_loader = DataLoader(train_ds, batch_size=params["batch_size"], shuffle=True,
                              num_workers=params["num_workers"])
    val_loader = DataLoader(val_ds, batch_size=params["batch_size"], shuffle=True, num_workers=params["num_workers"])
    return train_loader, val_loader


def train_one_epoch(model, loader, optimizer, criterion, device, params):
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    for batch_idx, (imgs, labels) in enumerate(tqdm(loader, desc="Training")):
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        out = model(imgs)
        loss = criterion(out, labels)

        # L1 Regularization
        l1_lambda = params["l1_lambda"]

        if l1_lambda > 0:
            l1_norm = sum(p.abs().sum() for p in model.parameters())
            loss = loss + l1_lambda * l1_norm

        loss.backward()
        optimizer.step()

        total_loss += loss.detach().item() * imgs.size(0)
        correct += out.argmax(1).eq(labels).sum().item()
        n += imgs.size(0)

        # if (batch_idx + 1) % params["log_interval"] == 0:
        #     print(f"  [{batch_idx+1}/{len(loader)}] "
        #           f"Training Loss: {total_loss/n:.4f} - Training Accuracy: {correct/n:.4f}")

    return total_loss / n, correct / n


def validate(model, loader, criterion, device, params):
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    val_metrics = ClassificationMetrics(params["num_classes"], device)
    with torch.no_grad():
        for imgs, labels in loader:
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


def run_training(model, params, device):
    train_loader, val_loader = get_loaders(params)
    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(model, params)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    early_stopping = None
    if params["enable_early_stopping"]:
        early_stopping = EarlyStopping(patience=params["patience"])

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
                                          criterion, device, params)
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
