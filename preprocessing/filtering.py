"""
Defines filtering and decomposition functions.
"""
import emd
import matplotlib.pyplot as plt
import numpy as np


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
