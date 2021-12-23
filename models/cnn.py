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


class ConvBlock(nn.Module):
    """
    A block of successive conv layers of total stride 2
    """
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride=2, padding=1)

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
        # Encoder
        self.block1 = ConvBlock(1, 16, 7)
        self.block2 = ConvBlock(16, 32, 5)
        self.bn1 = nn.BatchNorm2d(32)
        # self.dropout1 = nn.Dropout2d(0.3)
        self.block3 = ConvBlock(32, 32, 3)
        self.block4 = ConvBlock(32, 64, 3)
        self.bn2 = nn.BatchNorm2d(64)
        # self.dropout2 = nn.Dropout2d(0.3)
        self.block5 = ConvBlock(64, 96, 3)
        self.block6 = ConvBlock(96, 96, 3)
        self.bn3 = nn.BatchNorm2d(96)
        # self.dropout3 = nn.Dropout2d(0.3)
        self.block7 = ConvBlock(96, 96, 3)

        # Classification head
        self.fc1 = nn.Linear(15 * 10* 96, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.block1(x))
        x = torch.relu(self.block2(x))
        x = self.bn1(x)
        # x = self.dropout1(x)
        x = torch.relu(self.block3(x))
        x = torch.relu(self.block4(x))
        x = self.bn2(x)
        # x = self.dropout2(x)
        x = torch.relu(self.block5(x))
        x = torch.relu(self.block6(x))
        x = self.bn3(x)
        # x = self.dropout3(x)
        x = torch.relu(self.block7(x))

        x = torch.flatten(x, start_dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x


class HighGammaTemporalCNN(nn.Module):
    """
    Defines a CNN model adapted to classify images
    obtained through:
    - selecting the right and left temporal brain areas
    - selecting the High gamma frequency band only
    - using the FFT difference to create the images
    """
    def __init__(self):
        super().__init__()
        self.block1 = ConvBlock(1, 16, 7)
        # self.dropout1 = nn.Dropout2d(0.3)
        self.block2 = ConvBlock(16, 32, 5)
        self.bn1 = nn.BatchNorm2d(32)
        # self.dropout2 = nn.Dropout2d(0.3)
        self.block3 = ConvBlock(32, 32, 3)
        # self.dropout3 = nn.Dropout2d(0.3)
        self.block4 = ConvBlock(32, 64, 3)
        # self.dropout4 = nn.Dropout2d(0.3)
        self.bn2 = nn.BatchNorm2d(64)

        # Classification head
        self.fc1 = nn.Linear(64 * 14 * 9, 64)
        self.fc11 = nn.Linear(64, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.block1(x))
        # x = self.dropout1(x)
        x = torch.relu(self.block2(x))
        x = self.bn1(x)
        # x = self.dropout2(x)
        x = torch.relu(self.block3(x))
        # x = self.dropout3(x)
        x = torch.relu(self.block4(x))
        x = self.bn2(x)
        # x = self.dropout4(x)

        x = torch.flatten(x, start_dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc11(x))
        x = torch.sigmoid(self.fc2(x))
        return x



class STFCnn(nn.Module):
    """
    A CNN model made to be used with spectrograms.
    """
    def __init__(self, in_channels):
        super().__init__()
        self.block1 = ConvBlock(in_channels, 64, 5)
        self.bn1 = nn.BatchNorm2d(64)
        self.dropout1 = nn.Dropout2d(0.3)
        self.block2 = ConvBlock(64, 64, 3)

        # Classification head
        self.fc1 = nn.Linear(64 * 5 * 8, 128)
        self.fc11 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.block1(x))
        x = self.bn1(x)
        x = self.dropout1(x)
        x = torch.relu(self.block2(x))

        x = torch.flatten(x, start_dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc11(x))
        x = torch.sigmoid(self.fc2(x))
        return x
