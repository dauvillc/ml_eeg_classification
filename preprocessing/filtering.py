"""
Defines filtering and decomposition functions.
"""
import emd
import matplotlib.pyplot as plt
import numpy as np
import mne
from mne.preprocessing import (ICA, create_eog_epochs, create_ecg_epochs,
                               corrmap)


def plot_emd_imfs(signal, nb_imfs=10):
    """
    Plots the first nb_imfs IMF components obtained through EMD decomposition.
    :param signal: ndarray of shape (n_channels, n_timesteps) giving the signal.
    """
    # Estimates the IMFs composites
    imfs = emd.sift.sift(signal, max_imfs=nb_imfs)
    emd.plotting.plot_imfs(imfs, scale_y=True, cmap=True)
    plt.show()

def emd_filtering(epochs, which_imfs=[1, 2]):
    """
    Decomposes the signal included in each epoch into IMFs using EMD decomposition.
    Selects only the IMFs indicated in which_imfs, and returns their sums.
    :param epochs: ndarray of shape (N_epochs, N_channels, N_timesteps).
    :param which_imfs: list of integers, which IMFs to keep for the result
        signals.
    :return: an ndarray of shape (N_epochs, N_channels, N_timesteps)
        giving the filtered signals.
    """
    max_imf = max(which_imfs)
    result_epochs = []
    for epoch in epochs:
        channels = []
        for channel in epoch:
            # Computes the IMFs of the current channel
            # (transposition is to get shape (N_imfs, n_timesteps))
            imfs = emd.sift.sift(channel, max_imfs=max_imf + 1).T
            result_channel = np.sum(imfs[-(imfs.shape[0] - 1):], axis=0)
            channels.append(result_channel)

        # Converts the list of arrays to a 2D array of shape
        # (N_channels, N_timesteps)
        result_epochs.append(np.stack(channels))
    return np.stack(result_epochs)

""" ICA to be finished """

def ica_artifact_rejection(raw):
    ica.exclude = []
    
    # find which ICs match the EOG pattern
    eog_indices, eog_scores = ica.find_bads_eog(raw)
    ica.exclude = eog_indices

    # barplot of ICA component "EOG match" scores
    ica.plot_scores(eog_scores)

    # plot diagnostics
    ica.plot_properties(raw, picks=eog_indices)

    # plot ICs applied to raw data, with EOG matches highlighted
    ica.plot_sources(raw, show_scrollbars=False)
    
    # refit the ICA with 30 components this time
    new_ica = ICA(n_components=30, max_iter='auto', random_state=97)
    new_ica.fit(filt_raw)

    # find which ICs match the ECG pattern
    ecg_indices, ecg_scores = new_ica.find_bads_ecg(raw, method='correlation',
                                                threshold='auto')
    new_ica.exclude = ecg_indices

    # barplot of ICA component "ECG match" scores
    new_ica.plot_scores(ecg_scores)

    # plot diagnostics
    new_ica.plot_properties(raw, picks=ecg_indices)

    # plot ICs applied to raw data, with ECG matches highlighted
    new_ica.plot_sources(raw, show_scrollbars=False)
