import torch
import torch.nn as nn
import constants

class MockModel(nn.Module):
    def __init__(self, num_classes=constants.NUM_CLASSES, hold_frames=30):
        super().__init__()
        self.num_classes = num_classes
        self.hold_frames = hold_frames

        self.counter = 0
        self.current_idx = 0

        self.dummy_layer = nn.Linear(1, 1)

    def forward(self, x):
        self.counter += 1

        # Switch gloss after 'hold_frames' frames
        if self.counter >= self.hold_frames:
            self.counter = 0
            self.current_idx = (self.current_idx + 1) % self.num_classes

        batch_size = x.shape[0]
        device = x.device

        # Low prob for all glosses
        logits = torch.ones(batch_size, int(self.num_classes), device=device) * -5.0

        # Set high prob for target gloss only
        logits[:, self.current_idx] = 10.0

        return logits