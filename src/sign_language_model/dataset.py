"""
Dataset for loading pre-extracted ASL features with optional augmentation.
Torchvision-style design.
"""

import json
from typing import Any

import numpy as np
import torch

from .augmentations import (
    get_feature_augmentation,
    get_keypoint_augmentation,
)


class WLASLDataset(torch.utils.data.Dataset):
    """
    Dataset for loading pre-extracted features (.npz) with optional augmentation.

    Expected keys in .npz:
        - X_kps  : (N, 3, T, 75)   [optional]
        - X_rgb  : (N, 1024)       [optional]
        - X_flow : (N, 1024)       [optional]
        - y      : (N,)
    """

    def __init__(
        self,
        npz_path: str,
        gloss_map_path: str,
        augment_kps: bool = False,
        augment_features: bool = False,
        kps_aug_config: dict[str, Any] | None = None,
        feature_aug_config: dict[str, Any] | None = None,
    ):
        data = np.load(npz_path)

        self.X_rgb = (
            torch.from_numpy(data["X_rgb"]).float() if "X_rgb" in data else None
        )
        self.X_flow = (
            torch.from_numpy(data["X_flow"]).float() if "X_flow" in data else None
        )
        self.X_kps = (
            torch.from_numpy(data["X_kps"]).float() if "X_kps" in data else None
        )
        self.y = torch.from_numpy(data["y"]).long()

        # Setup augmentations
        if augment_kps and self.X_kps is not None:
            self.kps_transform = get_keypoint_augmentation(**(kps_aug_config or {}))
        else:
            self.kps_transform = None

        if augment_features:
            self.feature_transform = get_feature_augmentation(
                **(feature_aug_config or {})
            )
        else:
            self.feature_transform = None

        # Gloss mapping
        with open(gloss_map_path, "r") as f:
            gloss_map = json.load(f)
        self.index_to_label = {i: gloss for i, gloss in enumerate(gloss_map)}
        self.label_to_index = {gloss: i for i, gloss in enumerate(gloss_map)}

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        sample: dict[str, torch.Tensor] = {}

        if self.X_rgb is not None:
            rgb = self.X_rgb[idx]
            if self.feature_transform is not None:
                rgb = self.feature_transform(rgb)
            sample["rgb"] = rgb

        if self.X_flow is not None:
            flow = self.X_flow[idx]
            if self.feature_transform is not None:
                flow = self.feature_transform(flow)
            sample["flow"] = flow

        if self.X_kps is not None:
            kps = self.X_kps[idx]
            if self.kps_transform is not None:
                kps = self.kps_transform(kps)
            sample["kps"] = kps

        label = self.y[idx]
        return sample, label
