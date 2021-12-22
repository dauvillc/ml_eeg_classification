"""
Defines filtering and decomposition functions.
"""
import emd
import scipy
import matplotlib.pyplot as plt
import numpy as np
from mne.preprocessing import (ICA, create_eog_epochs, create_ecg_epochs,
                               corrmap)

_FREQ_BANDS_ = {'delta': [1, 4], 'theta': [4, 8], 'alpha': [8, 12], 'beta': [12, 25], 'lgamma': [25, 40],
                'hgamma': [40, 70]}


def select_frequency_bands(epochs, sampling_freq, freq_bands):
    """
    Filters the epochs to keep only a specifc band of frequencies
    from the usual bands of EEG data handling.
    :param epochs: array of shape (N_epochs, N_channels, N_timesteps)
    :param sampling_freq: int, sampling rate of the signal
    :param freq_bands: str or list of str. Which frequency bands to keep,
        among 'delta', 'theta', 'alpha', 'beta', 'lgamma', hgamma'.
    :return: the epochs filtered
    """
    if isinstance(freq_bands, str):
        freq_bands = [freq_bands]
    # Low and high frequency of each selected band
    lows = [_FREQ_BANDS_[band][0] for band in freq_bands]
    highs = [_FREQ_BANDS_[band][1] for band in freq_bands]
    # Min and max frequencies overall
    low, high = min(lows), max(highs)
    print(f'Selecting frequency band {[low, high]}')
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


def select_time_window(epochs, start, end, sampling_rate=512):
    """
    Selects a specific time window of the signals and discards
    the rest.
    :param epochs: array of shape (N_epochs, N_channels, N_timesteps)
    :param start: Beginning of the window in seconds
    :param end: End of the window in seconds
    :param sampling_rate: Sampling frequency in Hz
    :return: the shortened epochs
    """
    print(f'Selecting time window t=[{start}s, {end}s]')
    first_indx = int(start * sampling_rate)
    end_indx = int(end * sampling_rate)
    return epochs[:, :, first_indx:end_indx + 1]


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
