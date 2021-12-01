"""
Defines functions to handle epochs. This includes splitting the signals
based on events, and merging epochs together.
"""
import numpy as np


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
    # Retrieves the timesteps at which a 14 event happens
    epoch_starts = events[np.where(events[:, 2] == 14), 0][0]
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
    label_events = events[np.where(np.isin(events[:, 2], [12, 10])), 2][0]
    if label_events.shape[0] != epochs.shape[0]:
        raise ValueError(f'Found {epochs.shape[0]} epochs but only' +
                         f'{label_events.shape[0]} events 12 or 10')
    # shape = (N_epochs,) values in {1, 0}
    epochs_true_labels = (label_events == 12).astype(int)

    return epochs, epochs_true_labels
