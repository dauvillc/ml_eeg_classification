"""
Defines functions to handle epochs. This includes splitting the signals
based on events, and merging epochs together.
"""
import numpy as np
import matplotlib.pyplot as plt


def split_to_epochs(raw, events):
    """
    Splits the EEG signals into epochs based on an array of events.
    :param raw: mne.Raw object containing the signals.
    :param events: np array of shape (N_ev, 3) giving the timestep and event ids.
    :return: E, L where:
    - E is an np array of shape (N_epochs, N_channels, N_timesteps) giving the channels
        for each epoch;
    - L is an array of shape (N_epochs) giving the label (1 for 'gi', 0 for 'fo')
    """
    # Retrieves the timesteps at which a 9 or 11 event happens
    epoch_starts = events[np.where((events[:, 2] == 9) | (events[:, 2] == 11)), 0][0]
    # Same for event 13
    epoch_ends = events[np.where(events[:, 2] == 13), 0][0]
    if epoch_ends.shape != epoch_starts.shape:
        raise ValueError(f"Found {epoch_starts} epoch starts (14) for" +
                         f"{epoch_ends}epoch ends (13)")
    # Extracts each epoch and puts them into a list using the previous arrays
    epochs = [raw.get_data()[:, st:end + 1] for st, end in zip(epoch_starts, epoch_ends)]

    # All epochs don't have exactly the same length. Thus
    # we cut all of them to minimum length across all epochs
    epochs_lengths = [ep.shape[1] for ep in epochs]
    epochs = [ep[:, :min(epochs_lengths)] for ep in epochs]

    # Assembles the epochs into an array of shape
    # (n_epochs, n_channels, n_timesteps)
    epochs = np.stack(epochs)

    # Returns all events 12 or 10 that happened, in chronoligical order
    # shape = (N_epochs,), values in {12, 10}
    label_events = events[np.where(np.isin(events[:, 2], [11, 9])), 2][0]
    if label_events.shape[0] != epochs.shape[0]:
        raise ValueError(f'Found {epochs.shape[0]} epochs but only' +
                         f'{label_events.shape[0]} events 12 or 10')
    # shape = (N_epochs,) values in {1, 0}
    epochs_true_labels = (label_events == 11).astype(int)

    return epochs, epochs_true_labels


def assemble_epochs(epochs_list):
    """
    Assembles multiple epochs arrays into a single array.
    :param epochs_list: list of arrays. Each array has shape
        (N_epochs, N_channels, N_timesteps).
    :return: E, L where:
    - E is an array of shape (N_epochs_total, N_channels, N_timesteps).
    """
    # We can't simply concatenate all epochs arrays as they might not have
    # exactly the same amount of timesteps (different 3rd dimension).
    # What we can do is compute the minimum length accross ALL epochs
    # and cut all arrays" 3rd dim to that length
    min_length = min((min((ep.shape[1] for ep in epochs)) for epochs in epochs_list))
    epochs_list = [epochs[:, :, :min_length] for epochs in epochs_list]
    return np.concatenate(epochs_list, axis=0)


def plot_epochs(epochs):
    """
    Plots and displays information about the epochs, to help
    spot bad data.
    :param epochs: ndarray of shape (N_epochs, N_channels, N_timesteps)
    """
    fig, ax = plt.subplots(figsize=(12, 10), nrows=1, ncols=1)
    ax.bar(np.arange(epochs.shape[0]), [ep.std() for ep in epochs])
    ax.set_xticks(np.arange(epochs.shape[0]))
    fig.show()