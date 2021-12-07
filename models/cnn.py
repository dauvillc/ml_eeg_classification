"""
Implements the CNN models for the EEG fo / gi data classification
"""
import torch
import torch.nn as nn
import torchvision.models as models


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


class ConvBlock(nn.Module):
    """
    A block of successive conv layers of total stride 2
    """
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride=2)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.conv2(x)
        return x


class LargeCNN(nn.Module):
    """
    A larger CNN module than CNNeeg1-1 to treat larger images.
    """
    def __init__(self):
        super().__init__()
        self.block1 = ConvBlock(1, 16, 7)
        self.block2 = ConvBlock(16, 16, 5)
        self.block3 = ConvBlock(16, 32, 5)
        self.block4 = ConvBlock(32, 32, 5)
        self.block5 = ConvBlock(32, 32, 5)
        self.block6 = ConvBlock(32, 64, 5)
        self.block7 = ConvBlock(64, 96, 5)

    def forward(self, x):
        x = torch.relu(self.block1(x))
        x = torch.relu(self.block2(x))
        x = torch.relu(self.block3(x))
        x = torch.relu(self.block4(x))
        x = torch.relu(self.block5(x))
        x = torch.relu(self.block6(x))
        x = self.block7(x)
        return x
