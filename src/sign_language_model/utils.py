import random

import numpy as np
import torch


class EarlyStopping:
    def __init__(self, patience=10, min_delta=1e-4):
        self.best_loss = float("inf")
        self.counter = 0
        self.patience = patience
        self.min_delta = min_delta

    def step(self, loss):
        if loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


def set_seed(seed: int):
    """Set seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
