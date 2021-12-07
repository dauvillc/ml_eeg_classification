"""
Implements the CNN models for the EEG fo / gi data classification
"""
import torch
import torch.nn as nn


class CNNeeg(nn.Module):
    """
    Implements the CNNeeg1-1 architecture as described in Sarmiento et. al 2021.
    """
    def __init__(self, in_channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Dropout2d(0.25),
            nn.Conv2d(in_channels=in_channels, out_channels=50, kernel_size=5),
            nn.BatchNorm2d(50),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=50, out_channels=60, kernel_size=11),
            nn.BatchNorm2d(60),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(60)
        )

    def forward(self, x):
        return self.layers(x)
