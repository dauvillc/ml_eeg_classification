"""
Trains a model on data from a specific subject and day.
"""
import os
import numpy as np
import pandas as pd
import torch
from datetime import datetime as dt
from torch.utils.data import TensorDataset, DataLoader
from preprocessing import to_fft_electrode_difference, group_frequencies
from models import LargeCNN, STFCnn
from preprocessing import plot_channels, emd_filtering
from data_loading import load_data
from preprocessing.stf import to_spectrograms
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression


# PARAMETERS
_SUBJECT_ = '01'
# Possibles values: '1' to '5', or 'all'
_DAY_ = 'all'
_DATA_DIR_ = 'ready_data'
_RESULTS_SAVE_DIR_ = 'results'
_USE_CLEAN_DATA_ = True
_CROSS_VALIDATION_SPLITS_ = 4

if __name__ == "__main__":
    # ======================== DATA LOADING =================================#
    # Load the epochs and labels into ndarrays. Either loads the raw fif files
    # or the data cleaned by experts.
    epochs, labels = load_data(_DATA_DIR_, _SUBJECT_, _DAY_, _USE_CLEAN_DATA_)
    # Remove bad epochs
    # epochs = np.delete(epochs, [46, 53, 60], 0)
    # labels = np.delete(labels, [46, 53, 60])

    # Keep only a few epochs for faster debugging
    # epochs = epochs[:30]
    # labels = labels[:30]

    # ========================= PREPROCESSING ================================#
    # EMD filtering
    # epochs = emd_filtering(epochs)

    # Converting to the FFT of cross-channels-difference matrix
    # img_epochs = to_fft_electrode_difference(epochs, save_images=False, output_dir="ready_data/new_fft_images")
    img_epochs = to_spectrograms(epochs, 512, save_images=False, window_size=64)
    # img_epochs = group_frequencies(img_epochs, freq_groups=100)
    print(f"Obtained {img_epochs.shape[0]} images of shape {(img_epochs.shape[1], img_epochs.shape[2])}")
    # Rescales the images between 0 and 1
    img_epochs = (img_epochs - img_epochs.min()) / max(img_epochs.max() - img_epochs.min(), 0)

    # Reshapes the images to shape (batch_size, 1, H, W) as pytorch expects a channel axis
    # img_epochs = img_epochs[:, np.newaxis]

    # ======================== Computation device ============================#
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # ======================== Cross Validation ==============================#
    folds = KFold(n_splits=_CROSS_VALIDATION_SPLITS_)
    test_losses, test_accs = [], []
    for fold_indx, (train_index, test_index) in enumerate(folds.split(img_epochs, labels)):
        print(f"Cross-validating on subset {fold_indx}...")
        x_train, y_train = img_epochs[train_index], labels[train_index]
        x_test, y_test = img_epochs[test_index], labels[test_index]

        x_train = x_train.reshape(x_train.shape[0], -1)
        x_test = x_test.reshape(x_test.shape[0], -1)
        lr = LogisticRegression(C=100.0, random_state=42)
        lr.fit(x_train, y_train)
        print(f'LR train score: {lr.score(x_train, y_train)}, test score: {lr.score(x_test, y_test)}')
        continue

        # ======================== Data Loader ===================================#
        # Loads the data as pytorch tensors to allow for the network training
        train_dataset = TensorDataset(torch.Tensor(x_train), torch.Tensor(y_train))
        train_loader = DataLoader(train_dataset, batch_size=2)

        # ======================== Training ======================================#
        model = STFCnn(img_epochs.shape[1]).to(device)
        optim = torch.optim.Adam(model.parameters(), lr=0.0001)
        # Binary crossentropy loss for binary classification
        loss_fn = torch.nn.BCELoss()

        for epoch in range(40):
            losses = []
            for it, (x, y) in enumerate(train_loader):
                x, y = x.to(device), y.to(device)
                optim.zero_grad()
                pred = torch.flatten(model(x))
                loss = loss_fn(pred, y)
                loss.backward()
                optim.step()
                losses.append(loss.item())

            print(f'Epoch {epoch} - Avg Loss = {np.mean(losses)}')

        # ======================= Test ===========================================#
        with torch.no_grad():
            pred = model.cpu()(torch.Tensor(x_test))
            pred = torch.flatten(pred)
            loss = torch.mean(loss_fn(pred, torch.Tensor(y_test))).item()
            pred = np.array(pred)
            pred[pred > 0.5] = 1
            pred[pred < 1] = 0
            acc = np.mean(pred == y_test)
            print(f'Test loss: {loss}')
            print(f'Test acc: {acc}')
            test_losses.append(loss)
            test_accs.append(acc)

    # ========================= SAVING THE RESULTS ===========================#
    results = pd.DataFrame({"subset": np.arange(_CROSS_VALIDATION_SPLITS_),
                            "loss": test_losses,
                            "accuracy": test_accs})
    results.to_csv(os.path.join(_RESULTS_SAVE_DIR_,
                                'cross_valid_' + dt.now().strftime("%m%d%H") +
                                "_sub" + _SUBJECT_ + "_day" + _DAY_ + ".csv"),
                   index=False)
