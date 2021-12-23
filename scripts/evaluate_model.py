"""
Evaluates a model with specific conditions.
"""
import sys

sys.path.append('.')

import numpy as np
from argparse import ArgumentParser
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from scipy.fft import rfft, rfftfreq
from models.evaluation import cross_validate
from preprocessing import select_frequency_bands
from preprocessing import select_electrodes_groups
from preprocessing.scaling import rescale
from data_loading import load_data, load_channels_names
from preprocessing import to_fft_electrode_difference
from preprocessing.stf import to_spectrograms

# PARAMETERS
_SUBJECT_ = '01'
_DATA_DIR_ = 'ready_data'


def evaluate(model_type, day=4, areas='all', freq_bands='all', max_depth=3, C=1,
             cross_valid_folds=5, use_clean_data=True,
             use_spectrogram=False, use_fft_differences=False):
    """
    Evaluates a specific model over specific features.
    :param model_type: 'random_forest' or 'logistic_regression'.
    :param day: int, experiment day from 1 to 5
    :param areas: str or list of str, which brain areas to use
    :param freq_bands: str or list of str, which frequencies to use
    :param max_depth: maximum depth of the trees for Random forest models
    :param C: L2 regularization penalty for the Logistic regressions. The lower,
        the higher the penalty (same as in sklearn's LogisticRegression).
    :param cross_valid_folds: int, number of cross-validation folds.
    :param use_clean_data: whether to use the cleaned data furnished by the lab.
    :param use_spectrogram: Boolean. If True, the spectrogram of each electrode is
        used as features. Serves as a baseline model.
    :param use_fft_differences: Boolean. If True, the FFT of the difference between each pair
        of electrodes is used as features. No more preprocessing is applied, as this serves
        as a baseline model.
    :return: (train_acc, train_std), (test_acc, test_std)
    """
    # ======================= DATA LOADING ======================== #
    # Loads the epochs and labels into ndarrays. Either loads the raw fif files
    # or the data cleaned by experts.
    epochs, labels = load_data(_DATA_DIR_, _SUBJECT_, day, use_clean_data)
    # Loads the name of each electrode and selects a specific group
    electrodes = load_channels_names(_DATA_DIR_)

    # ==================== TEST =================================== #

    # Builds the features
    if use_spectrogram:
        x = to_spectrograms(epochs, 512, window_size=64)
    elif use_fft_differences:
        x = to_fft_electrode_difference(epochs)
    else:
        x = select_electrodes_groups(epochs, electrodes, areas)
        x = select_frequency_bands(x, 512, freq_bands)
        x = np.abs(rfft(x))
        # Cuts the spectrum at 100 Hz, since it null from that point on
        freqs = rfftfreq(epochs.shape[-1], 1 / 512)
        x = x[:, :, freqs <= 100]

    # If the model is a logistic regression, we perform
    # min-max scaling
    if model_type == 'logistic_regression':
        x = rescale(x, 'minmax')

    # Reshape to (N_samples, N_features)
    x = x.reshape((x.shape[0], -1))
    # Builds the model with the current parameter and evaluates it through
    # cross-validation
    if model_type == 'random_forest':
        model = RandomForestClassifier(n_estimators=300, max_depth=max_depth, random_state=42)
    else:
        model = LogisticRegression(C=C, max_iter=200, random_state=42)

    # Performs the cross-validation
    return cross_validate(model, x, labels, cross_valid_folds)


if __name__ == "__main__":
    # ======================= ARGUMENTS =========================== #
    arg_parser = ArgumentParser(description='Evaluates and optimizes a model on all combinations of features')
    arg_parser.add_argument('model', type=str, help='logistic_regression or random_forest')
    arg_parser.add_argument('--freqs', type=str, nargs='?', help='Frequency band to use.')
    arg_parser.add_argument('--brain_areas', type=str, nargs='?', help='Which electrodes to use')
    arg_parser.add_argument('--day', type=int, help='experiment day', default=4)
    arg_parser.add_argument('--penalty', type=float, help='Regularization parameter')
    arg_parser.add_argument('--max_depth', type=int, help='Maximum trees depth')
    args = arg_parser.parse_args()
    day = args.day
    freq_bands = args.freqs
    electrode_groups = args.brain_areas
    model_type = args.model
    l2_penalty = args.penalty
    max_depth = args.max_depth

    (train_acc, train_std), (acc, std) = evaluate(day, electrode_groups, freq_bands, model_type)

    print(f'Training: acc={train_acc}, std={train_std}')
    print(f'Test: acc={acc}, std={std}')
