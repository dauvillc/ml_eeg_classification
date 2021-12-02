"""
Trains a model on data from a specific subject and day.
"""
import os
import sys
import numpy as np
from preprocessing import to_fft_electrode_difference, group_frequencies
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

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

    # ========================= PREPROCESSING ================================#
    # Converting to the FFT of cross-channels-difference matrix
    img_epochs = to_fft_electrode_difference(epochs)
    print(f"Obtained {img_epochs.shape[0]} images of shape {(img_epochs.shape[1], img_epochs.shape[2])}")
    # Rescales the images between 0 and 1
    img_epochs = (img_epochs - img_epochs.min()) / max(img_epochs.max() - img_epochs.min(), 0)

    # ======================== Train-test split ==============================#
    x_train, x_test, y_train, y_test = train_test_split(img_epochs, labels, train_size=0.8, random_state=42)

    # ======================== Training ======================================#
    rfc = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42)
    rfc.fit(x_train.reshape((x_train.shape[0], -1)), y_train)

    print(f'Classifier train score: {rfc.score(x_train.reshape((x_train.shape[0], -1)), y_train)}')
    print(f'Classifier test score: {rfc.score(x_test.reshape((x_test.shape[0], -1)), y_test)}')

