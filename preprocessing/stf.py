"""
Defines method to work with space-time-frequency domain.
"""
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from scipy.signal import spectrogram, get_window
from scipy.fft import rfft, rfftfreq

_DEFAULT_IMG_DIR_ = "ready_data/fft_images"


def to_fft_electrode_difference(epochs, save_images=False, output_dir=_DEFAULT_IMG_DIR_, sampling_rate=512,
                                keep_freqs=None):
    """
    Transforms the EEG signals into a matrix of dimensions nb_freq x C(C-1)/2
    where C is the number of channels.
    Column i * C + j corresponds to the spectral power difference
                |FFT(Ci) - FFT(Cj)|
    The frequencies are divided into nb_freq frequency bands.
    Each row corresponds to a frequency.
    :param epochs: array of shape (N_epochs, N_channels, N_timesteps), EEG data divided into epochs.
    :param save_images: Boolean, optional, whether to save the matrix as images for the sake
        of visualization.
    :param output_dir: if save_images is True, directory to save the images into.
    :param sampling_rate: Sampling rate in Hz. Only required if keep_freqs is given.
    :param keep_freqs: optional tuple of frequencies (a, b). The FFTs of the signals will be cropped
        between a and b.
    :return: an array of shape (N_epochs, nb_freq, C(C-1)/2) giving the 2D version of each epoch.
    """
    nb_epochs, nb_ch, nb_timesteps = epochs.shape
    # Creates the tensor V containing Ci - Cj for all j > i
    ch_diffs = np.empty((nb_epochs, nb_ch * (nb_ch - 1) // 2, nb_timesteps))
    print("Processing the pairwise difference...")
    for k_epoch, epoch in enumerate(epochs):
        cnt = 0
        for i in range(nb_ch):
            for j in range(i + 1, nb_ch):
                ch_diffs[k_epoch, cnt] = epoch[i] - epoch[j]
                cnt += 1
        print(f'{k_epoch}/{nb_epochs}        ', end='\r')

    # Computes the FFT of each difference signal
    # Since the data is reals, the fft would be symmetrical. Therefore we only need to
    # return the positive part of the spectrum, using rfft()
    print("Processing FFT...")
    fft_imgs = np.abs(rfft(ch_diffs))
    if keep_freqs is not None:
        fft_freqs = rfftfreq(ch_diffs.shape[-1], 1 / sampling_rate)
        a, b = keep_freqs
        fft_imgs = fft_imgs[:, :, (fft_freqs >= a) & (fft_freqs <= b)]

    # Take the transpose to have the rows be the frequencies and columns
    # the electrodes difference, as in the paper
    fft_imgs = np.swapaxes(fft_imgs, 1, 2)

    # Save the images if requested
    if save_images:
        print(f"Saving FFT imgs to {output_dir}")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        for k, img in enumerate(fft_imgs):
            path = os.path.join(output_dir, f"fft_{k}.jpg")
            cv2.imwrite(path, img)

    return fft_imgs


def to_spectrograms(epochs, sampling_freq, save_images=False, output_dir=_DEFAULT_IMG_DIR_,
                    window_size=128):
    """
    Given a set of epochs, return their space-time frequency representation.
    :param epochs: signals, ndarray of shape (N_epochs, N_channels, N_timesteps)
    :param sampling_freq: integer, sampling frequency of the signals.
    :param save_images: boolean, whether to save the spectrograms as images for
        visualisation purpose.
    :param output_dir: if save_images is True, directory to which save the images.
    :param window_size: int, number of samples in the spectrogram window.
    :return: an ndarray of shape (N_epochs, N_channels, N_timesteps, N_frequencies)
    """
    window = get_window('tukey', window_size)
    print(f'Computing the spectrogram for {epochs.shape[0]} * {epochs.shape[1]} signals')
    freqs, times, epoch_spectres = spectrogram(epochs, sampling_freq, window)

    if save_images:
        for epoch_indx, epoch in enumerate(epoch_spectres):
            for k_ch, channel in enumerate(epoch):
                path = os.path.join(output_dir, f'spectre_{epoch_indx}_{k_ch}.png')
                plt.imshow(channel)
                plt.savefig(path)
                plt.close()
    return epoch_spectres



def group_frequencies(fft_images, freq_groups):
    """
    Reduces the number of rows of the images by grouping together
    neighbour frequencies.
    The frequencies are aggregated by taking the average amplitude of
    all frequencies in each group.
    :param fft_images: ndarray of shape (N_images, H, W).
    :param freq_groups: int between 2 and the images' height.
        Number of frequency groups in the resulting images.
    :return: an ndarray of shape (N_images, freq_groups, W)
        giving the condensed images.
    """
    nb_imgs, height, width = fft_images.shape
    reduced_imgs = np.empty((nb_imgs, freq_groups, width))

    grps_size = height // freq_groups
    for grp in range(freq_groups):
        freqs = fft_images[:, grp * grps_size: (grp + 1) * grps_size]
        reduced_imgs[:, grp] = np.mean(freqs, axis=1)

    return reduced_imgs
