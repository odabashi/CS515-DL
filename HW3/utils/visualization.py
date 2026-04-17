import datetime
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
import seaborn as sns
import numpy as np
import torch.nn as nn
import os
from torchviz import make_dot
import torch
import logging
from sklearn.manifold import TSNE
from utils.gradcam import GradCAM


logger = logging.getLogger("HW3")
os.makedirs("./assets/", exist_ok=True)


def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300, assets_path="./assets/"):
    """
    Save the current figure to "assets_path/<fig_id>.<ext>".

    Args:
        fig_id (str): Filename stem (no extension).
        tight_layout (bool): Call `plt.tight_layout()` before saving.
        fig_extension (str): File format (default "png").
        resolution (int): DPI for raster formats (default 300).
        assets_path (str): Output directory (created if absent).
    """
    os.makedirs(assets_path, exist_ok=True)
    path = os.path.join(assets_path, f"{fig_id}.{fig_extension}")
    logger.info(f"Saving figure {fig_id}")
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)


def plot_confusion_matrix(cm, save_path=f"confusion_matrix_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"):
    """
    Plot and save a confusion matrix as a seaborn annotated heatmap.

    Args:
        cm: Confusion matrix tensor or array.
        save_path (str): Filename stem for `save_fig`.
    """
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
    """
    Plot training vs. validation loss curves and save to disk.

    Args:
        history (dict): Must contain keys "train_loss" and "val_loss" (lists of floats, one per epoch).
        save_path (str): Filename stem.
    """
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


def visualize_model(model, params, save_path=f"./assets/model_graph_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"):
    """
    Render and save the model computation graph using torchviz.

    Args:
        model (nn.Module): Model to visualize.
        params (dict): Configuration dict (needs "dataset" and "resize_input").
        save_path (str): Output path stem (no extension; .png appended).
    """
    if params["dataset"] == "mnist":
        x = torch.randn(4, 1, 28, 28).to(next(model.parameters()).device)
    elif params["dataset"] == "cifar10":
        if params.get("resize_input", False):
            x = torch.randn(4, 3, 224, 224).to(next(model.parameters()).device)
        else:
            x = torch.randn(4, 3, 32, 32).to(next(model.parameters()).device)
    else:
        raise ValueError("Unsupported dataset")
    y = model(x)
    dot = make_dot(y, params=dict(model.named_parameters()))
    dot.graph_attr.update(dpi="300", size="12,14")
    dot.render(save_path, format="png", cleanup=True)


def plot_tsne(logits, labels, save_path=f"tsne_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"):
    """
    Visualizes the decision structure of the trained model using t-SNE on logits.
    Each point represents one test sample colored by its true class label.

    Args:
        logits (np.ndarray): Model output logits.
        labels (np.ndarray): True integer labels.
        save_path (str): Filename stem.
    """
    tsne = TSNE(
        n_components=2,
        perplexity=30,
        learning_rate="auto",
        init="pca",
        random_state=42
    )

    embeddings = tsne.fit_transform(logits)

    plt.figure(figsize=(8, 8))

    scatter = plt.scatter(
        embeddings[:, 0],
        embeddings[:, 1],
        c=labels,
        cmap="tab10",
        s=6,
        alpha=0.8
    )

    plt.colorbar(scatter)
    plt.title("t-SNE Decision Space (Model Logits)")

    save_fig(save_path)

    plt.close()


def plot_tsne_adversarial(clean_logits, clean_labels, adv_logits, adv_labels, norm_label, save_path=None):
    """
    t-SNE visualisation that overlays adversarial samples onto clean samples.

    Both sets are embedded jointly (single TSNE fit) so they share the same 2-D coordinate space, making the
    displacement of adversarial points visible relative to the clean class clusters.

    Visual encoding
    ---------------
    - Clean samples : filled circles (o), colored by true class.
    - Adversarial : crosses (X), colored by true class (same color scheme). If the attack succeeds the cross
                    will appear near the cluster of the *wrong* class, far from its true-class circle cluster.

    Args:
        clean_logits (np.ndarray): Clean logits.
        clean_labels (np.ndarray): True labels for clean samples.
        adv_logits (np.ndarray): Adversarial logits.
        adv_labels (np.ndarray): True labels for adversarial samples. Pass the true labels (not predicted) so the
                                 color still reflects ground truth, making displacement obvious.
        norm_label (str): Threat model string for the plot title / filename.
        save_path (str, optional): Filename stem. Auto-generated if None.

    Notes:
        - For large test sets (N=10 000) t-SNE can be slow. Consider passing a random subsample if speed is a concern.
        - Logits from `run_pgd_test` are already collected for you in `pgd_results["adv_logits_linf"]`.
    """
    if save_path is None:
        ts = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        save_path = f"tsne_adversarial_{norm_label}_{ts}"

    n_clean = len(clean_logits)
    n_adv = len(adv_logits)

    # Joint embedding: stack clean + adversarial, fit once
    combined = np.vstack([clean_logits, adv_logits])  # (N+M, C)

    logger.info(f"Fitting t-SNE on {n_clean} clean + {n_adv} adversarial logits ...")
    tsne = TSNE(
        n_components=2,
        perplexity=30,
        learning_rate="auto",
        init="pca",
        random_state=42,
    )
    embeddings = tsne.fit_transform(combined)  # (N+M, 2)

    emb_clean = embeddings[:n_clean]
    emb_adv = embeddings[n_clean:]

    fig, ax = plt.subplots(figsize=(9, 9))

    # Clean samples — small filled circles
    sc_clean = ax.scatter(
        emb_clean[:, 0], emb_clean[:, 1],
        c=clean_labels, cmap="tab10",
        s=8, alpha=0.5, marker="o",
        label="Clean",
        vmin=0, vmax=9,
    )

    # Adversarial samples — larger X markers, same colour = true class
    ax.scatter(
        emb_adv[:, 0], emb_adv[:, 1],
        c=adv_labels, cmap="tab10",
        s=40, alpha=0.9, marker="X",
        edgecolors="black", linewidths=0.3,
        label="Adversarial",
        vmin=0, vmax=9,
    )

    plt.colorbar(sc_clean, ax=ax, label="True class")

    # Legend for marker shape
    legend_handles = [
        mpatches.Patch(label="o  Clean sample"),
        mpatches.Patch(label="x  Adversarial sample"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=9)

    ax.set_title(
        f"t-SNE (Logit Space) - Clean vs Adversarial [{norm_label.upper()} PGD]\n"
        f"(colour = true class;  x displaced from its cluster -> fooled)",
        fontsize=10,
    )

    save_fig(save_path)
    plt.close()
    logger.info(f"Adversarial t-SNE saved => ./assets/{save_path}.png")


def plot_gradcam(misclassified_clean, misclassified_adv, misclassified_labels, misclassified_adv_preds, model,
                 mean, stddev, device, norm_label, num_samples=5, save_path=None):
    def overlay_heatmap(image, heatmap, alpha=0.4):
        heatmap_color = cm.jet(heatmap)[..., :3]
        overlay = (1 - alpha) * image + alpha * heatmap_color
        return np.clip(overlay, 0, 1)

    def tensor_to_image(tensor, mean, stddev, device):
        tensor = tensor.to(device)
        mean = torch.tensor(mean).view(1, -1, 1, 1).to(device)
        stddev = torch.tensor(stddev).view(1, -1, 1, 1).to(device)

        img = tensor * stddev + mean
        img = img.clamp(0, 1)

        img = img.squeeze().permute(1, 2, 0).cpu().numpy()
        return img

    def get_last_conv_layer(model):
        """Automatically find the last Conv2d layer in the model."""
        last_conv = None
        for module in model.modules():
            if isinstance(module, nn.Conv2d):
                last_conv = module
        if last_conv is None:
            raise ValueError("No Conv2d layer found in the model.")
        return last_conv

    if save_path is None:
        ts = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        save_path = f"tsne_adversarial_{norm_label}_{ts}"

    model.eval()
    target_layer = get_last_conv_layer(model)
    gradcam = GradCAM(model, target_layer)

    collected = 0
    fig, axes = plt.subplots(num_samples, 4, figsize=(12, 4 * num_samples))

    CIFAR_10_CLASSES = [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ]

    for idx in range(len(misclassified_clean)):
        clean_cam = gradcam.generate(misclassified_clean[idx], misclassified_labels[idx])
        adv_cam = gradcam.generate(misclassified_adv[idx], misclassified_adv_preds[idx])

        row = axes[collected]

        # Clean image
        row[0].imshow(tensor_to_image(misclassified_clean[idx], mean, stddev, device))
        row[0].set_title(f"Clean\nLabel={CIFAR_10_CLASSES[misclassified_labels[idx].item()]}")

        # Clean CAM
        row[1].imshow(
            overlay_heatmap(
                tensor_to_image(misclassified_clean[idx], mean, stddev, device),
                clean_cam
            )
        )
        row[1].set_title(f"Clean GradCAM\nPred={CIFAR_10_CLASSES[misclassified_labels[idx].item()]}")

        # Adv image
        row[2].imshow(tensor_to_image(misclassified_adv[idx], mean, stddev, device))
        row[2].set_title("Adversarial")

        # Adv CAM
        row[3].imshow(
            overlay_heatmap(
                tensor_to_image(misclassified_adv[idx], mean, stddev, device),
                adv_cam
            )
        )
        row[3].set_title(f"Adversarial GradCAM\nPred={CIFAR_10_CLASSES[misclassified_adv_preds[idx].item()]}")

        for ax in row:
            ax.axis("off")

        collected += 1

    gradcam.remove_hooks()
    save_fig(save_path)
    plt.close()
    logger.info(f"Grad-CAM saved => ./assets/{save_path}.png")
