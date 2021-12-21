"""
Defines various simple scaling functions.
"""
import numpy as np


def rescale(epochs, method="minmax"):
    """
    Rescales the epochs using a specified method.
    :param epochs: array of shape (N_epochs, N_channels, N_timesteps)
    :param method: One of 'minmax', 'normalization".
        - 'minmax': performs x = (x - x.min) / (x.max - x.min)
        - 'normalization': x = (x - x.mean) / x.std
    :return: the rescaled epochs
    """
    if method == 'minmax':
        scale = epochs.max() - epochs.min()
        if scale < 0.01:
            raise ValueError('The max and min of the data are too close')
        return (epochs - epochs.min()) / scale

    if method == 'normalization':
        if epochs.std() < 0.01:
            raise ValueError('The std of the data is too small for normalization')
        return (epochs - epochs.mean()) / epochs.std()