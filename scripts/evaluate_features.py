"""
Evaluates a certain model (Logistic regression or Random Forest)
on a set of features combinations.
"""
import os
import sys

sys.path.append('.')

import numpy as np
import pandas as pd
from argparse import ArgumentParser
from itertools import product
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from scipy.fft import rfft, rfftfreq
from models.evaluation import cross_validate
from preprocessing import select_frequency_bands
from preprocessing import select_electrodes_groups
from preprocessing.filtering import select_time_window
from preprocessing.scaling import rescale
from data_loading import load_data, load_channels_names

# PARAMETERS
_SUBJECT_ = '01'
_DATA_DIR_ = 'ready_data'
_RESULTS_SAVE_DIR_ = 'results'
_USE_CLEAN_DATA_ = True
_CROSS_VALIDATION_FOLDS_ = 5
_OUT_DIR_ = "evaluations"

if __name__ == "__main__":
    # ======================= ARGUMENTS =========================== #
    arg_parser = ArgumentParser(description='Evaluates and optimizes a model on all combinations of features')
    arg_parser.add_argument('model', type=str, help='logistic_regression or random_forest')
    arg_parser.add_argument('--day', type=int, help='experiment day', default=4)
    args = arg_parser.parse_args()
    day = args.day
    model_type = args.model

    # ======================= DATA LOADING ======================== #
    # Loads the epochs and labels into ndarrays. Either loads the raw fif files
    # or the data cleaned by experts.
    epochs, labels = load_data(_DATA_DIR_, _SUBJECT_, day, _USE_CLEAN_DATA_)
    # Loads the name of each electrode and selects a specific group
    electrodes = load_channels_names(_DATA_DIR_)

    # ====================== FEATURES PARAMETERS ================== #
    # Write here all combinations of features to be tested
    freq_bands = ['all', 'hgamma', 'lgamma', 'alpha', 'beta', 'theta',
                  'delta', ['hgamma', 'lgamma']]
    electrode_groups = ['right temporal', 'left temporal', 'parietal', 'occipital',
                        'frontal', 'central', ['right temporal', 'left temporal']]

    # ==================== MODELS PARAMETERS ====================== #
    # Regularization parameter for the logistic regression (C parameters of
    # sklearn's LogisticRegression model).
    lr_C_values = [10 ** i for i in range(-3, 3)]
    # Random forest maximum depth.
    max_depths = [1, 2, 3]

    # ==================== TEST =================================== #
    # Dictionnary containing the information about each set of conditions.
    # This dictionnary will be assembled into a pandas dataframe in the end.
    results = {"frequencies": [], "electrodes": [],
               "train_acc": [], "train_std": [],
               "acc": [], "std": [],
               "param": [],
               "n_features": []}

    # Total number of features parameters to be tested
    nb_combinations = len(freq_bands) * len(electrode_groups)
    for comb_indx, (band, elec_group) in enumerate(product(freq_bands, electrode_groups)):
        print(f"Testing parameters Freq band={band}, Electrodes={elec_group} "
              + f"(Combination {comb_indx + 1} / {nb_combinations}")

        # Builds the features
        x = select_electrodes_groups(epochs, electrodes, elec_group)
        x = select_frequency_bands(x, 512, band)
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

        # Decide on which parameter to optimize according to which
        # model is being tested
        if model_type == 'random_forest':
            parameters = max_depths
        else:
            parameters = lr_C_values

        for param in parameters:
            print(f'Parameter = {param}')
            # Builds the model with the current parameter and evaluates it through
            # cross-validation
            if model_type == 'random_forest':
                model = RandomForestClassifier(n_estimators=300, max_depth=param, random_state=42)
            else:
                model = LogisticRegression(C=param, max_iter=200, random_state=42)
            # Sklearn's built-in function was good as well, but you know, why make it simple ?
            # let's pretend it's because we need the standard deviation
            (train_acc, train_std), (acc, std) = cross_validate(model, x, labels, _CROSS_VALIDATION_FOLDS_)

            # Puts the frequency bands in the format 'band1_band2_..'
            if isinstance(band, str):
                band = [band]
            band = '_'.join(band)
            results['frequencies'].append(band)
            results['electrodes'].append(elec_group)
            results['train_acc'].append(train_acc)
            results['train_std'].append(train_std)
            results['acc'].append(acc)
            results['std'].append(std)
            results['param'].append(param)
            results['n_features'].append(x.shape[1])

    # Saves the results as a CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(_OUT_DIR_, f"{model_type}_features_evaluation.csv"))
