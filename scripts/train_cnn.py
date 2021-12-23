"""
Trains a model on data from a specific subject and day.
usage:
python scripts/train_cnn.py [--spectrogram]
"""
import sys

sys.path.append('.')

import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from datetime import datetime as dt
from torch.utils.data import TensorDataset, DataLoader
from preprocessing import to_fft_electrode_difference
from models.cnn import LargeCNN, STFCnn, HighGammaTemporalCNN
from data_loading import load_data, load_channels_names
from preprocessing.stf import to_spectrograms
from sklearn.model_selection import KFold
from preprocessing.filtering import select_frequency_bands
from preprocessing.channels import select_electrodes_groups


# PARAMETERS
_SUBJECT_ = '01'
# Possibles values: '1' to '5', or 'all'
_DAY_ = '5'
_DATA_DIR_ = 'ready_data'
_RESULTS_SAVE_DIR_ = 'results'
_USE_CLEAN_DATA_ = True

if __name__ == "__main__":
    # ======================== DATA LOADING =================================#
    # Load the epochs and labels into ndarrays. Either loads the raw fif files
    # or the data cleaned by experts.
    epochs, labels = load_data(_DATA_DIR_, _SUBJECT_, _DAY_, _USE_CLEAN_DATA_)
    channels_names = load_channels_names(_DATA_DIR_)

    # ========================= PREPROCESSING ================================#
    use_spectrogram = '--spectrogram' in sys.argv
    use_baseline = '--baseline' in sys.argv

    # Converting to the FFT of cross-channels-difference matrix
    # or to a spectrogram as requested by the user
    if not use_baseline:
        epochs = select_electrodes_groups(epochs, channels_names, ['right temporal', 'left temporal'])
        # Uses the High Gamma frequency band by applying a bandpass filter,
        # and then keeping only the frequencies between 25 and 70 Hz in the spectrum
        epochs = select_frequency_bands(epochs, 512, 'hgamma')
        img_epochs = to_fft_electrode_difference(epochs, keep_freqs=[25, 70])
        # Reshapes the images to shape (batch_size, 1, H, W) as pytorch expects a channel axis
        img_epochs = img_epochs[:, np.newaxis]
    else:
        if use_spectrogram:
            img_epochs = to_spectrograms(epochs, 512, save_images=False, window_size=64)
        else:
            img_epochs = to_fft_electrode_difference(epochs, save_images=False, output_dir="ready_data/new_fft_images")
            # Reshapes the images to shape (batch_size, 1, H, W) as pytorch expects a channel axis
            img_epochs = img_epochs[:, np.newaxis]
    print(f"Obtained images of shape {img_epochs.shape}")

    # Rescales the images between 0 and 1
    img_epochs = (img_epochs - img_epochs.mean()) / img_epochs.std()

    # ======================== Computation device ============================#
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # ======================== Cross Validation ==============================#
    # We use sklearn's KFold to perform automatized cross-validation
    n_folds = 4
    folds = KFold(n_splits=n_folds, shuffle=False)

    test_losses, test_accs = [], []
    train_losses, train_accs = [], []
    for fold_indx, (train_index, test_index) in enumerate(folds.split(img_epochs, labels)):
        print(f"Cross-validating on subset {fold_indx}...")
        x_train, y_train = img_epochs[train_index], labels[train_index]
        x_test, y_test = img_epochs[test_index], labels[test_index]

        # ======================== Data Loader ===================================#
        # Loads the data as pytorch tensors to allow for the network training
        train_dataset = TensorDataset(torch.Tensor(x_train), torch.Tensor(y_train))
        train_loader = DataLoader(train_dataset, batch_size=2)

        # ======================== Training ======================================#
        # Loads the CNN model
        if not use_baseline:
            model = HighGammaTemporalCNN().to(device)
        elif use_spectrogram:
            model = STFCnn(x_train.shape[1]).to(device)
        else:
            model = LargeCNN().to(device)
        optim = torch.optim.Adam(model.parameters(), lr=2e-5)

        # Binary crossentropy loss for binary classification
        loss_fn = torch.nn.BCELoss()

        n_epochs = 20
        # Training loop
        for epoch in range(n_epochs):
            losses = []
            for it, (x, y) in enumerate(train_loader):
                x, y = x.to(device), y.to(device)
                optim.zero_grad()
                pred = torch.flatten(model(x))
                loss = loss_fn(pred, y)
                loss.backward()
                optim.step()
                losses.append(loss.item())

            train_loss = np.mean(losses)
            print(f'Epoch {epoch} - Avg Loss = {train_loss}')

        # Saves the training loss of the last epoch
        train_losses.append(train_loss)

        # ======================= Test ===========================================#
        with torch.no_grad():
            # Sets the model in evaluation mode, which is necessary
            # for some layers such as Dropout or BatchNorm to infer
            # properly
            model.eval()
            pred = model.cpu()(torch.Tensor(x_test))

            pred = torch.flatten(pred)
            loss = torch.mean(loss_fn(pred, torch.Tensor(y_test))).item()

            # Converts the probabilities to predictions by thresholding
            pred = np.array(pred)
            pred[pred > 0.5] = 1
            pred[pred < 1] = 0
            acc = np.mean(pred == y_test)
            print(f'Test loss: {loss}')
            print(f'Test acc: {acc}')
            test_losses.append(loss)
            test_accs.append(acc)

    mean_test_loss, std_test_loss = np.mean(test_losses), np.std(test_losses)
    mean_train_loss, std_train_loss = np.mean(train_losses), np.std(train_losses)
    mean_test_acc, std_test_acc = np.mean(test_accs), np.std(test_accs)
    print(f"Mean train loss={mean_train_loss}, std={std_train_loss}")
    print(f"Mean test loss={mean_test_loss}, std={std_test_loss}")
    print(f"Mean test acc={mean_test_acc}, std={std_test_acc}")
    # ========================= SAVING THE RESULTS ===========================#
    results = pd.DataFrame({"subset": np.arange(n_folds),
                            "loss": test_losses,
                            "accuracy": test_accs})
    results.to_csv(os.path.join(_RESULTS_SAVE_DIR_,
                                'cross_valid_' + dt.now().strftime("%m%d%H") +
                                "_sub" + _SUBJECT_ + "_day" + _DAY_ + ".csv"),
                   index=False)
