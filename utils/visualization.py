import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import os


def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300, assets_path="./assets/"):
    os.makedirs(assets_path, exist_ok=True)
    path = os.path.join(assets_path, f"{fig_id}.{fig_extension}")
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)


def plot_confusion_matrix(cm, save_path=f"confusion_matrix_{str(datetime.datetime.now()).replace(':', '.')}"):
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
