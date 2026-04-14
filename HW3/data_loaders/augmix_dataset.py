"""
Dataset wrapper that returns three views of every image for the AugMix
Jensen-Shannon Divergence (JSD) consistency loss.

Why a dataset wrapper?

A dataset wrapper solves several problems: a single `__getitem__` call fetches the raw PIL image once and applies 
three deterministically different transforms to it, guaranteeing that clean, aug1, and aug2 always originate from the
same image. 

Transform design
----------------
preprocess: applied to the clean view:
    RandomCrop(32, padding=4) -> RandomHorizontalFlip -> ToTensor -> Normalize

augmix_preprocess: applied to both augmented views:
    RandomCrop(32, padding=4) -> RandomHorizontalFlip -> AugMix -> ToTensor -> Normalize

AugMix is placed after the geometric augmentations and before ToTensor because it operates on PIL images and
returns PIL images. ToTensor and Normalize are always the final two steps.

Reference
---------
D. Hendrycks, N. Mu, E. D. Cubuk, B. Zoph, J. Gilmer, B. Lakshminarayanan,
"AugMix: A Simple Method to Improve Robustness and Uncertainty under Data
Shift," ICLR 2020.  https://arxiv.org/abs/1912.02781
"""

from typing import Tuple, Callable
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class AugMixDataset(Dataset):
    """
    Wraps an existing dataset and returns three views per sample.

    The underlying dataset must be constructed without a transform
    (transform=None) so that `__getitem__` returns a raw PIL image.
    The three transforms are then applied here, inside this wrapper.

    Args:
        base_dataset (Dataset):
            Source dataset returning (PIL.Image, int) tuples. Pass `transform=None` when constructing it.
        preprocess (callable):
            Transform applied to produce the clean view. Typically: RandomCrop -> RandomHFlip -> ToTensor -> Normalize.
        augmix_preprocess (callable):
            Transform applied independently twice to produce the two augmented views (aug1, aug2).
            Typically: RandomCrop -> RandomHFlip -> AugMix -> ToTensor -> Normalize.

    Returns (per sample):
        Tuple[Tensor, Tensor, Tensor, int]: (clean, aug1, aug2, label)
            - clean - standard augmented view
            - aug1 - AugMix view 1
            - aug2 - AugMix view 2  (independent draw)
            - label - class index
    """
    def __init__(
        self,
        base_dataset: Dataset,
        preprocess: Callable,
        augmix_preprocess: Callable,
    ) -> None:
        self.base_dataset = base_dataset
        self.preprocess = preprocess
        self.augmix_preprocess = augmix_preprocess

    def __len__(self) -> int:
        """Return the number of samples (identical to the base dataset)."""
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """
        Fetch sample `idx` and return three independently transformed views.

        The two AugMix views are produced by calling `augmix_preprocess` twice on the same PIL image.
        Because AugMix samples its augmentation chain stochastically, each call yields a different
        realization, giving us two independent views as required by the JSD loss.

        Args:
            idx (int): Sample index.

        Returns:
            Tuple[Tensor, Tensor, Tensor, int]: (clean, aug1, aug2, label)
        """
        img, label = self.base_dataset[idx]    # PIL.Image, int

        clean = self.preprocess(img)           # standard augmented view
        aug1 = self.augmix_preprocess(img)     # AugMix view 1
        aug2 = self.augmix_preprocess(img)     # AugMix view 2 (independent)

        return clean, aug1, aug2, label


def build_augmix_transforms(mean, std, severity=3, mixture_width=3) -> Tuple[Callable, Callable]:
    """
    Build the `preprocess` and `augmix_preprocess` transform pipelines.

    Separating this into a helper keeps `get_loaders` readable and makes it easy to swap augmentation parameters
    from `params` without duplicating the transform construction logic.

    Args:
        mean (tuple):          Per-channel normalization mean.
        std (tuple):           Per-channel normalization std.
        severity (int):        AugMix operation severity (default 3).
        mixture_width (int):   Number of augmentation chains to mix (default 3).

    Returns:
        Tuple[Callable, Callable]: (preprocess, augmix_preprocess)
            - `preprocess` - clean view pipeline (no AugMix)
            - `augmix_preprocess` - augmented view pipeline (with AugMix)
    """
    preprocess = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    augmix_preprocess = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.AugMix(severity=severity, mixture_width=mixture_width),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    return preprocess, augmix_preprocess