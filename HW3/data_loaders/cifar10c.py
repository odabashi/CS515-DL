"""
cifar10c_dataset.py
-------------------
PyTorch Dataset wrapper for the CIFAR-10-C benchmark downloaded from https://zenodo.org/records/2535967.

CIFAR-10-C (Hendrycks & Dietterich, 2019) is the standard CIFAR-10 test set
corrupted with 19 types of common image corruptions (noise, blur, weather (fog, frost, snow),
digital), each at 5 severity levels.

Directory layout expected on disk
----------------------------------
<cifar10c_dir>/
    labels.npy              # shape (50 000,)  int64  — same for every corruption
    gaussian_noise.npy      # shape (50 000, 32, 32, 3) uint8
    shot_noise.npy
    ...                     # one file per corruption type
    saturate.npy

Each corruption file stacks all five severity levels back-to-back:
    indices  0      –  9 999     -> severity 1
    indices 10 000  – 19 999     -> severity 2
    ...
    indices 40 000  – 49 999     -> severity 5

Reference
---------
D. Hendrycks and T. Dietterich, "Benchmarking neural network robustness to
common corruptions and perturbations," ICLR 2019.
https://github.com/hendrycks/robustness
"""

import os
from typing import Optional, Callable, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image


# All 19 corruption types in the benchmark
CIFAR_10_C_CORRUPTIONS = [
    # NOISE
    "gaussian_noise",
    "shot_noise",
    "impulse_noise",
    "speckle_noise",
    # BLUR
    "defocus_blur",
    "glass_blur",
    "motion_blur",
    "zoom_blur",
    "gaussian_blur",
    # WEATHER
    "snow",
    "frost",
    "fog",
    # DIGITAL
    "brightness",
    "contrast",
    "elastic_transform",
    "pixelate",
    "jpeg_compression",
    "spatter",
    "saturate",
]

# Number of test samples per severity level
SAMPLES_PER_SEVERITY = 10000


class Cifar10cDataset(Dataset):
    """
    PyTorch Dataset for a single corruption type at a single severity level from the CIFAR-10-C benchmark.

    Args:
        data_dir (str): Path to the directory containing the CIFAR-10-C `.npy` files.
        corruption (str): Name of the corruption type (e.g. "gaussian_noise"). Must be one of the 19 names listed
                          in `CIFAR_10_C_CORRUPTIONS`.
        severity (int): Severity level in the range [1, 5] or -1 for all severities.
        transform (callable, optional): Transform applied to each PIL image before returning. Should include at
                                        least `ToTensor()` and `Normalize()`.

    Raises:
        FileNotFoundError: If the `.npy` file for the requested corruption or the `labels.npy` file cannot be found.
        ValueError: If `severity` is not in [1, 5] or `corruption` is not a recognised CIFAR-10-C corruption name.
    """

    def __init__(
        self,
        data_dir: str,
        corruption: str,
        severity: int,
        transform: Optional[Callable] = None,
    ) -> None:
        if corruption not in CIFAR_10_C_CORRUPTIONS:
            raise ValueError(f"Unknown corruption '{corruption}'. Valid options: {CIFAR_10_C_CORRUPTIONS}")
        if severity not in range(1, 6) and severity != -1:
            raise ValueError(f"Severity must be 1–5 or -1, got {severity}.")

        data_path = os.path.join(data_dir, f"{corruption}.npy")
        labels_path = os.path.join(data_dir, "labels.npy")

        if not os.path.isfile(data_path):
            raise FileNotFoundError(f"Corruption file not found: {data_path}")
        if not os.path.isfile(labels_path):
            raise FileNotFoundError(f"Labels file not found: {labels_path}")

        all_data = np.load(data_path)           # (50000, 32, 32, 3) uint8
        all_labels = np.load(labels_path)       # (50000,)

        if severity != -1:
            # Each .npy file holds all 5 severities stacked -> slice the right 10000
            start = (severity - 1) * SAMPLES_PER_SEVERITY
            end = severity * SAMPLES_PER_SEVERITY
            self.data = all_data[start:end]         # (10000, 32, 32, 3)
            self.labels = all_labels[start:end]     # (10000,)
            self.transform = transform
        else:
            # Stack all 5 severities -> (50000, 32, 32, 3)
            self.data = all_data
            self.labels = all_labels
            self.transform = transform

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        """Return the number of samples (always 10 000 for one severity)."""
        return len(self.data)

    # ------------------------------------------------------------------
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Return the sample at position `idx`.

        Args:
            idx (int): Index in [0, len(self))

        Returns:
            Tuple[torch.Tensor, int]: (image, label) tuple where `image` has shape (3, 32, 32) after the transform
                                      is applied.
        """
        img = Image.fromarray(self.data[idx])   # HWC uint8 -> PIL
        label = int(self.labels[idx])

        if self.transform is not None:
            img = self.transform(img)

        return img, label
