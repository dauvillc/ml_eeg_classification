"""
Trains "simple" classifiers such as logistic regression or random
forests on the same data as the CNN.
"""
import os
import numpy as np
from preprocessing import to_fft_electrode_difference, group_frequencies
from preprocessing import plot_channels
from preprocessing import select_frequency_bands
from preprocessing import select_electrodes_groups
from preprocessing.filtering import select_time_window
from preprocessing.scaling import rescale
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from scipy.fft import rfft, rfftfreq
from collections import defaultdict
from data_loading import load_data, load_channels_names

# PARAMETERS
_SUBJECT_ = '01'
# Possibles values: '1' to '5', or 'all'
_DAY_ = '4'
_DATA_DIR_ = 'ready_data'
_RESULTS_SAVE_DIR_ = 'results'
_USE_CLEAN_DATA_ = True
_CROSS_VALIDATION_SPLITS_ = 10

_MODELS_ = ['random_forest']

if __name__ == "__main__":
    # ======================== DATA LOADING =================================#
    # Load the epochs and labels into ndarrays. Either loads the raw fif files
    # or the data cleaned by experts.
    epochs, labels = load_data(_DATA_DIR_, _SUBJECT_, _DAY_, _USE_CLEAN_DATA_)
    # Loads the name of each electrode and selects a specific group
    electrodes = load_channels_names(_DATA_DIR_)
    epochs = select_electrodes_groups(epochs, electrodes, ['right temporal'])
    epochs = select_time_window(epochs, 0, 5, 512)
    epochs = select_frequency_bands(epochs, 512, 'all')

    # ========================= PREPROCESSING ================================#
    # Frequency filtering

    # EMD filtering
    # epochs = emd_filtering(epochs)

    # Converting to the FFT of cross-channels-difference matrix
    # img_epochs = to_fft_electrode_difference(epochs, save_images=False, output_dir="ready_data/new_fft_images")
    img_epochs = np.abs(rfft(epochs))
    freqs = rfftfreq(epochs.shape[-1], 1 / 512)
    img_epochs = img_epochs[:, :, freqs <= 100]
    plot_channels(img_epochs[0], x=freqs[freqs <= 100], to_file="figures/img_epochs.png")
    # img_epochs = to_spectrograms(epochs, 512, save_images=False, window_size=8)
    # img_epochs = group_frequencies(img_epochs, freq_groups=100)

    print(f"Obtained the features of shape {img_epochs.shape}")
    # Rescales the images
    img_epochs = rescale(img_epochs, 'normalization')

    # ======================== Cross Validation ==============================#
    folds = KFold(n_splits=_CROSS_VALIDATION_SPLITS_)
    accs = defaultdict(list)
    train_accs = defaultdict(list)
    for fold_indx, (train_index, test_index) in enumerate(folds.split(img_epochs, labels)):
        print(f"Cross-validating on subset {fold_indx}...")
        x_train, y_train = img_epochs[train_index], labels[train_index]
        x_test, y_test = img_epochs[test_index], labels[test_index]

        # Flatten the features to be coherent with sklearn's methods
        x_train = x_train.reshape(x_train.shape[0], -1)
        x_test = x_test.reshape(x_test.shape[0], -1)

        # Logistic regression
        if 'logistic_regression' in _MODELS_:
            lr = LogisticRegression(penalty='l2', C=1, random_state=42, solver='lbfgs', max_iter=200)
            lr.fit(x_train, y_train)
            accs["lr"].append(lr.score(x_test, y_test))
            train_accs['lr'].append(lr.score(x_train, y_train))

        # Random forests
        if 'random_forest' in _MODELS_:
            rfc = RandomForestClassifier(n_estimators=300, max_depth=2, random_state=42)
            rfc.fit(x_train, y_train)
            accs["rfc"].append(rfc.score(x_test, y_test))
            train_accs['rfc'].append(rfc.score(x_train, y_train))

        if 'gradient_boosting' in _MODELS_:
            gbc = GradientBoostingClassifier(n_estimators=50, learning_rate=5, random_state=42)
            gbc.fit(x_train, y_train)
            accs["gbc"].append(gbc.score(x_test, y_test))
            train_accs['gbc'].append(gbc.score(x_train, y_train))

        if 'lda' in _MODELS_:
            lda = LinearDiscriminantAnalysis()
            lda.fit(x_train, y_train)
            accs["lda"].append(lda.score(x_test, y_test))
            train_accs['ldc'].append(lda.score(x_train, y_train))

        if 'svm' in _MODELS_:
            svc = SVC(C=1.0)
            svc.fit(x_train, y_train)
            accs["svm"].append(svc.score(x_test, y_test))
            train_accs['svm'].append(svc.score(x_train, y_train))
    for (model, accuracies), tr_accs in zip(accs.items(), train_accs.values()):
        print(
            f"{model}: Training acc = {np.mean(tr_accs):1.4f}, std={np.std(tr_accs):1.4f} - "
            + f"Test mean acc={np.mean(accuracies):1.4f}, std={np.std(accuracies):1.4f}")
