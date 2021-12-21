"""
Defines filtering and decomposition functions.
"""
import emd
import scipy
import matplotlib.pyplot as plt
import numpy as np
from mne.preprocessing import (ICA, create_eog_epochs, create_ecg_epochs,
                               corrmap)


_FREQ_BANDS_ = {'delta': [1, 4], 'theta': [4, 8], 'alpha':[8, 12], 'beta':[12, 25], 'lgamma': [25, 40],
                'hgamma':[25, 40]}


def select_frequency_bands(epochs, sampling_freq, freq_band):
    """
    Filters the epochs to keep only a specifc band of frequencies
    from the usual bands of EEG data handling.
    :param epochs: array of shape (N_epochs, N_channels, N_timesteps)
    :param sampling_freq: int, sampling rate of the signal
    :param freq_band: One of 'delta', 'theta', 'alpha', 'beta', 'lgamma'
        or 'hgamma'
    :return: the epochs filtered
    """
    print(f'Selecting band frequency {freq_band}')
    low, high = _FREQ_BANDS_[freq_band]
    return apply_bandpass(epochs, sampling_freq, low, high)


def apply_bandpass(epochs, sampling_freq, low, high):
    """
    Filters each channel with a bandpass filter of critical frequencies
    [low, high]
    :param epochs: array of shape (N_epochs, N_channels, N_timesteps)
    :param sampling_freq: Sampling frequency of the signals
    :param low: int, low critical frequency
    :param high: int, high critical frequency
    :return: the filtered epochs of same shape as epochs
    """
    # Scipy requires the cutoff frequencies to be indicated as
    # proportions of the Nyquist freq (which is half the sampling rate).
    nyq_freq = sampling_freq * 0.5
    low, high = low / nyq_freq, high / nyq_freq

    b, a = scipy.signal.butter(3, [low, high], 'band')
    return scipy.signal.lfilter(b, a, epochs, axis=-1)


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
