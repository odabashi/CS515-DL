import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import os
from torchviz import make_dot
import torch
import logging


logger = logging.getLogger("HW1")
os.makedirs("./assets/", exist_ok=True)


def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300, assets_path="./assets/"):
    os.makedirs(assets_path, exist_ok=True)
    path = os.path.join(assets_path, f"{fig_id}.{fig_extension}")
    logger.info(f"Saving figure {fig_id}")
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)


def plot_confusion_matrix(cm, save_path=f"confusion_matrix_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"):
    plt.figure(figsize=(8, 6))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues"
    )

    plt.xlabel("Prediction")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")

    save_fig(save_path)
    plt.close()


def plot_learning_curves(history, save_path=f"loss_curves_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"):
    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss curve
    plt.figure(figsize=(12, 6))
    plt.plot(epochs, history["train_loss"], label="Training Loss")
    plt.plot(epochs, history["val_loss"], label="Validation Loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()

    save_fig(save_path)
    plt.close()


def visualize_model(model, save_path=f"./assets/model_graph_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"):
    x = torch.randn(4, 1, 28, 28).to(next(model.parameters()).device)
    y = model(x)
    dot = make_dot(y, params=dict(model.named_parameters()))
    dot.graph_attr.update(dpi="300", size="12,14")
    dot.render(save_path, format="png", cleanup=True)
