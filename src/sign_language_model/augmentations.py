"""
Torchvision-style data augmentation for ASL features.
Keypoints and feature embeddings are handled separately.
"""

import torch
import torch.nn as nn
from torchvision import transforms


# -------------------------------------------------
# Keypoint augmentations (C, T, V)
# -------------------------------------------------


class TemporalJitter(nn.Module):
    """Randomly shift frames in time."""

    def __init__(self, max_shift: int = 2):
        super().__init__()
        self.max_shift = max_shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.max_shift <= 0:
            return x

        C, T, V = x.shape
        shift = torch.randint(
            -self.max_shift, self.max_shift + 1, (1,), device=x.device
        ).item()

        return torch.roll(x, shifts=shift, dims=1)


class AddGaussianNoise(nn.Module):
    """Add Gaussian noise to keypoints."""

    def __init__(self, std: float = 0.02):
        super().__init__()
        self.std = std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.std <= 0:
            return x
        return x + torch.randn_like(x) * self.std


# -------------------------------------------------
# Feature-space augmentations (RGB / Flow)
# -------------------------------------------------


class FeatureNoise(nn.Module):
    """Add Gaussian noise to feature vectors."""

    def __init__(self, std: float = 0.05):
        super().__init__()
        self.std = std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.std <= 0:
            return x
        return x + torch.randn_like(x) * self.std


class FeatureDropout(nn.Module):
    """Randomly zero feature dimensions."""

    def __init__(self, p: float = 0.1):
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.p <= 0:
            return x
        mask = torch.rand_like(x) > self.p
        return x * mask


# -------------------------------------------------
# Factory functions
# -------------------------------------------------


def get_keypoint_augmentation(
    temporal_jitter: bool = True,
    add_noise: bool = True,
    max_shift: int = 2,
    noise_std: float = 0.02,
) -> nn.Module:
    """
    Build keypoint augmentation pipeline.
    """
    ops = []

    if temporal_jitter:
        ops.append(TemporalJitter(max_shift=max_shift))
    if add_noise:
        ops.append(AddGaussianNoise(std=noise_std))

    return transforms.Compose(ops) if ops else nn.Identity()


def get_feature_augmentation(
    add_noise: bool = True,
    feature_dropout: bool = True,
    noise_std: float = 0.05,
    dropout_prob: float = 0.1,
) -> nn.Module:
    """
    Build feature-space augmentation pipeline.
    """
    ops = []

    if add_noise:
        ops.append(FeatureNoise(std=noise_std))
    if feature_dropout:
        ops.append(FeatureDropout(p=dropout_prob))

    return transforms.Compose(ops) if ops else nn.Identity()
