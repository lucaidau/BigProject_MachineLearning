# src/model.py
"""Simple CNN model for binary COVID‑19 detection.

The original dataset contains four classes, but the user asked for a *binary* classification
(COVID‑19 vs Normal).  The model therefore maps the four original labels to two by treating
`label == 1` (COVID) as positive and everything else as negative.  The training script performs
this conversion on‑the‑fly.

The architecture is deliberately lightweight – suitable for the modest image size (128×128) and
the limited amount of training data.  It can be extended later if higher accuracy is required.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CovidCNN(nn.Module):
    """Convolutional neural network.

    Input shape: ``(N, 1, 128, 128)`` (single‑channel grayscale images).
    Output: logits of shape ``(N, 2)`` for the two classes.
    """

    def __init__(self):
        super().__init__()
        # Conv block 1
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        # Conv block 2
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        # Conv block 3
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        # Fully‑connected layers
        self.fc1 = nn.Linear(64 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, 2)
        self.pool = nn.MaxPool2d(2, 2)  # halves spatial dimensions

    def forward(self, x):
        # Block 1
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # 64 → 32
        # Block 2
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # 32 → 16
        # Block 3
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  # 16 → 8
        # Flatten
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        logits = self.fc2(x)
        return logits


def get_model(device: torch.device | None = None) -> nn.Module:
    """Create a ``CovidCNN`` instance and move it to *device* if provided.

    Parameters
    ----------
    device: ``torch.device`` or ``None``
        Target device (e.g. ``torch.device('cuda')``). If ``None`` the model stays on CPU.
    """
    model = CovidCNN()
    if device is not None:
        model = model.to(device)
    return model


