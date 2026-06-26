"""CNN model definition for CIFAR-10 image classification."""

import torch
from torch import nn


class CIFAR10CNN(nn.Module):
    """A small convolutional neural network for 32x32 RGB images."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()

        self.features = nn.Sequential(
            # Input: (batch, 3, 32, 32)
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # (batch, 32, 16, 16)
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # (batch, 64, 8, 8)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(p=0.25),
            nn.Linear(256, num_classes),  # Raw class scores (logits)
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return one logit vector of length ``num_classes`` per image."""
        features = self.features(images)
        return self.classifier(features)
