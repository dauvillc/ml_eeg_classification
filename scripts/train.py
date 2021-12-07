"""
Trains a model on data from a specific subject and day.
"""
import os
import sys
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from preprocessing import to_fft_electrode_difference, group_frequencies
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from models import LargeCNN
from preprocessing import plot_channels, emd_filtering

_SUBJECT_ = '01'
_DAY_ = '1'
_DATA_DIR_ = 'ready_data'


if __name__ == "__main__":
    # ======================== DATA LOADING =================================#
    epochs_path = os.path.join(_DATA_DIR_, "data_Sub" + _SUBJECT_ + "_day" + _DAY_ + ".np")
    labels_path = os.path.join(_DATA_DIR_, "labels_Sub" + _SUBJECT_ + "_day" + _DAY_ + ".np")
    epochs, labels = None, None
    with open(epochs_path, "rb") as epochs_file:
        epochs = np.load(epochs_file)
    with open(labels_path, "rb") as labels_file:
        labels = np.load(labels_file)
    # Remove bad epochs
    epochs = np.delete(epochs, [46, 53, 60], 0)
    labels = np.delete(labels, [46, 53, 60])

    # Keep only a few epochs for faster debugging
    epochs = epochs[:30]
    labels = labels[:30]

    # ========================= PREPROCESSING ================================#
    # EMD filtering
    epochs = emd_filtering(epochs)

    # Converting to the FFT of cross-channels-difference matrix
    img_epochs = to_fft_electrode_difference(epochs)
    # img_epochs = group_frequencies(img_epochs, freq_groups=100)
    print(f"Obtained {img_epochs.shape[0]} images of shape {(img_epochs.shape[1], img_epochs.shape[2])}")
    # Rescales the images between 0 and 1
    img_epochs = (img_epochs - img_epochs.min()) / max(img_epochs.max() - img_epochs.min(), 0)

    # Reshapes the images to shape (batch_size, 1, H, W) as pytorch expects a channel axis
    img_epochs = img_epochs[:, np.newaxis]

    # ======================== Train-test split ==============================#
    x_train, x_test, y_train, y_test = train_test_split(img_epochs, labels, train_size=0.8, random_state=42)

    # ======================== Data Loader ===================================#
    # Loads the data as pytorch tensors to allow for the network training
    train_dataset = TensorDataset(torch.Tensor(x_train), torch.Tensor(y_train))
    train_loader = DataLoader(train_dataset, batch_size=1)

    # ======================== Training ======================================#
    model = LargeCNN()
    for x, y in train_loader:
        opt = model(x)
        print(f"Output size: {opt.shape}")


