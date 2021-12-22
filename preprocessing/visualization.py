"""
Defines visualization functions to help choose the preprocessing.
"""
import matplotlib.pyplot as plt
import numpy as np

def plot_channels(epoch, channels=None, x=None, to_file=None):
    """
    Plots the channels from an epoch in separate figures.
    :param epoch: ndarray of shape (N_channels, N_timesteps)
    :param channels: list of integers within [0, N_channels - 1] indicating
        which channels should be plotted. If None (default), then all channels
        are plotted.
    :param x: values to use as x axis, of shape (N_timesteps).
    :param to_file: image file to which the figure should be saved. If None,
        the figure is simply displayed.
    """
    plt.figure(figsize=(14, 14))
    if channels is None:
        channels = np.arange(epoch.shape[0])

    if x is None:
        x = np.arange(epoch.shape[-1])
    for k, channel in enumerate(channels):
        plt.subplot(len(channels), 1, k + 1)
        plt.title(f"Channel {k}")
        plt.plot(x, epoch[channel], "-")
    if to_file:
        plt.savefig(to_file)
        plt.close()
    else:
        plt.show()