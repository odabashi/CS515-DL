"""
attacks.py
----------
Adversarial attack implementations for robustness evaluation.

Currently implemented
---------------------
    PGDAttack — Projected Gradient Descent (Madry et al., 2018) supports L-infinity and L2 threat models

How PGD works
------------------------------------------
Instead of optimizing model weights to minimize loss (training), PGD optimizes the input pixels to maximize loss
i.e. it finds the worst-case perturbation that fools the model, subject to a budget constraint (epsilon).

The algorithm:
    1. Start from X_0 = x + uniform noise clipped to the eps-ball
       (random start makes the attack stronger than FGSM)
    2. For t = 1 … T:
           a. Forward pass: compute loss L(f(x̃), y)
           b. Backward pass: compute ∇_{x̃} L
           c. Step in gradient direction (sign for L-infinity, normalised for L2)
           d. Project x̃ back onto the eps-ball centred at x
           e. Clip to valid image range
    3. Return x̃_T as the adversarial example

L-infinity threat model
    All pixel perturbations bounded by eps in absolute value.
    Step: x̃ ← x̃ + alpha · sign(∇L)
    Project: x̃ ← clip(x̃, x − eps, x + eps)

L2 threat model
    Euclidean distance between x̃ and x bounded by eps.
    Step: x̃ ← x̃ + alpha · ∇L / ‖∇L‖₂
    Project: if ‖x̃ − x‖₂ > eps  →  x̃ ← x + eps · (x̃ − x) / ‖x̃ − x‖₂

Working in normalised pixel space
-----------------------------------
The model receives *normalised* inputs (mean-subtracted, std-divided).  We run
the attack entirely in this normalised space.  The valid pixel range [0, 1]
maps to channel-wise intervals [(0 − mean_c) / std_c, (1 − mean_c) / std_c],
so we precompute ``x_min`` and ``x_max`` tensors and use them for clipping.

The eps and alpha values are specified in raw (un-normalised) pixel scale as per the
assignment: eps_linf = 4/255 ≈ 0.0157, eps_l2 = 0.25.  Because CIFAR-10 std
values are close to 0.2, the effective normalised eps_linf is roughly 4/255/0.2
≈ 0.08 per channel.  We handle this conversion internally so callers pass the
raw values.

Reference
---------
A. Madry, A. Makelov, L. Schmidt, D. Tsipras, A. Vladu,
"Towards Deep Learning Models Resistant to Adversarial Attacks," ICLR 2018.
https://arxiv.org/abs/1706.06083
"""

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


def denormalize(x, mean, std):
    mean = torch.tensor(mean, device=x.device, dtype=x.dtype).view(1, -1, 1, 1)
    std = torch.tensor(std, device=x.device, dtype=x.dtype).view(1, -1, 1, 1)
    return x * std + mean


def normalize(x, mean, std):
    mean = torch.tensor(mean, device=x.device, dtype=x.dtype).view(1, -1, 1, 1)
    std = torch.tensor(std, device=x.device, dtype=x.dtype).view(1, -1, 1, 1)
    return (x - mean) / std


def pgd_linf_attack(model, images, labels, mean, std, eps=4/255, alpha=1/255, steps=20, random_start=True):
    """
    PGD L-infinity attack (done in pixel space, respecting [0,1])
    Core idea:
        - All perturbation logic (eps-ball, clipping to [0,1]) is done in PIXEL space.
        - Only the forward pass uses normalized images.
        - This sidesteps all per-channel normalization scaling issues.

    Returns normalized adversarial images.

    Args:
        model: neural network
        images: normalized input
        labels: labels
        mean: mean of the dataset
        std: standard deviation of the dataset
        eps: max perturbation
        alpha: step size
        steps: iterations
        random_start: whether to start from a random point
    """
    model.eval()

    # Move to pixel space [0, 1]; detach from any graph
    x = denormalize(images.detach(), mean, std).clamp(0.0, 1.0)  # (N, C, H, W) in [0, 1]
    y = labels.clone().detach().to(x.device)

    # Random start: uniform noise in the pixel-space L-inf eps-ball
    # Initialize perturbation
    if random_start:
        # uniform in [-eps, +eps], then project to [0,1]
        x_adv = x + torch.empty_like(x).uniform_(-eps, eps)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    else:
        x_adv = x.clone()

    # PGD iterations
    for _ in range(steps):
        x_adv = x_adv.detach().requires_grad_(True)

        logits = model(normalize(x_adv, mean, std))
        loss = F.cross_entropy(logits, y)

        grad = torch.autograd.grad(loss, x_adv, only_inputs=True)[0]

        with torch.no_grad():
            # Steepest ascent step in L-inf (sign of gradient)
            x_adv = x_adv + alpha * grad.sign()

            # project onto L-inf eps-ball around ORIGINAL x (not x_adv)
            delta = torch.clamp(x_adv - x, min=-eps, max=eps)
            x_adv = x + delta

            # clip to valid pixel range [0, 1]
            x_adv = x_adv.clamp(0.0, 1.0)

        x_adv = x_adv.detach()  # prevent graph growth across iterations

    # Return in normalized space
    return normalize(x_adv, mean, std).detach()


def _l2_normalize_per_sample(t: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    n = t.flatten(1).norm(p=2, dim=1).clamp(min=eps).view(-1, 1, 1, 1)
    return t / n


def pgd_l2_attack(model, images, labels, mean, std, eps=0.25, alpha=0.05, steps=20, random_start=True):
    """
    PGD L2 attack
    """
    model.eval()

    # Move to pixel space [0, 1]; detach from any graph
    x = denormalize(images.detach(), mean, std).clamp(0.0, 1.0)  # (N, C, H, W) in [0, 1]
    y = labels.clone().detach().to(x.device)

    # Random start: uniform noise in the pixel-space L2 eps-ball
    # Initialize perturbation
    if random_start:
        # Gaussian noise -> normalize to unit sphere -> scale by eps
        # Then clamp to [0,1] and recompute delta so projection is clean
        noise = torch.randn_like(x)
        noise = _l2_normalize_per_sample(noise)
        x_adv = (x + noise * eps).clamp(0.0, 1.0)
    else:
        x_adv = x.clone()

    # PGD iterations
    for _ in range(steps):
        x_adv = x_adv.detach().requires_grad_(True)

        logits = model(normalize(x_adv, mean, std))
        loss = F.cross_entropy(logits, y)

        grad = torch.autograd.grad(loss, x_adv, only_inputs=True)[0]

        with torch.no_grad():
            # ascent step in normalized gradient direction
            # Normalize gradient to unit L2 norm (per sample)
            # so the step size alpha is meaningful regardless of gradient magnitude
            grad_normalized = _l2_normalize_per_sample(grad)
            x_adv = x_adv + alpha * grad_normalized

            # Project onto L2 ball of radius eps around original x (pixel space)
            delta = x_adv - x
            delta_norm = delta.flatten(1).norm(p=2, dim=1).clamp_min(1e-12).view(-1, 1, 1, 1)
            factor = torch.clamp(eps / delta_norm, max=1.0)
            x_adv = x + delta * factor

            # clip to valid pixel range [0, 1]
            x_adv = x_adv.clamp(0.0, 1.0)

        x_adv = x_adv.detach()  # prevent graph growth across iterations

    return normalize(x_adv, mean, std).detach()
