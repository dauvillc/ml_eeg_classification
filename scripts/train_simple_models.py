"""
Trains "simple" classifiers such as logistic regression or random
forests on the same data as the CNN.
"""
import os
import numpy as np
import torch
from preprocessing import to_fft_electrode_difference, group_frequencies
from preprocessing import plot_channels, emd_filtering
from preprocessing.stf import to_spectrograms
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from collections import defaultdict
from data_loading import load_data


# PARAMETERS
_SUBJECT_ = '01'
# Possibles values: '1' to '5', or 'all'
_DAY_ = '1'
_DATA_DIR_ = 'ready_data'
_RESULTS_SAVE_DIR_ = 'results'
_USE_CLEAN_DATA_ = True
_CROSS_VALIDATION_SPLITS_ = 10

_MODELS_ = ['logistic_regression', 'random_forest']


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
    img_epochs = to_spectrograms(epochs, 512, save_images=False, window_size=8)
    # img_epochs = group_frequencies(img_epochs, freq_groups=100)

    print(f"Obtained {img_epochs.shape[0]} images of shape {(img_epochs.shape[1], img_epochs.shape[2])}")

    # ======================== Cross Validation ==============================#
    folds = KFold(n_splits=_CROSS_VALIDATION_SPLITS_)
    accs = defaultdict(list)
    for fold_indx, (train_index, test_index) in enumerate(folds.split(img_epochs, labels)):
        print(f"Cross-validating on subset {fold_indx}...")
        x_train, y_train = img_epochs[train_index], labels[train_index]
        x_test, y_test = img_epochs[test_index], labels[test_index]

        # Flatten the features to be coherent with sklearn's methods
        x_train = x_train.reshape(x_train.shape[0], -1)
        x_test = x_test.reshape(x_test.shape[0], -1)

        # Logistic regression
        if 'logistic_regression' in _MODELS_:
            lr = LogisticRegression(C=0.1, random_state=42)
            lr.fit(x_train, y_train)
            accs["lr"].append(lr.score(x_test, y_test))

        # Random forests
        if 'random_forest' in _MODELS_:
            rfc = RandomForestClassifier(n_estimators=100, max_depth=3)
            rfc.fit(x_train, y_train)
            accs["rfc"].append(rfc.score(x_test, y_test))

        if 'gradient_boosting' in _MODELS_:
            gbc = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1)
            gbc.fit(x_train, y_train)
            accs["rfc"].append(gbc.score(x_test, y_test))

    for model, accuracies in accs.items():
        print(f"{model}: mean acc={np.mean(accuracies)}, std={np.std(accuracies)}")
