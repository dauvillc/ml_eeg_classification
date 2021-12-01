"""
Defines functions to load events from the -eve.txt files.
"""
import pandas as pd
import numpy as np


def load_eve_file(file_path, sampling_freq):
    """
    Loads the events from an -eve.txt file.
    :param file_path: path to the -eve.txt file
    :param sampling_freq: Sampling frequency in Hz.
    :return: the events as an array of shape (N_events, 3)
        (MNE format).
        The columns are:
        - Timestep of the event
        - 0
        - Event ID
    """
    events = pd.read_csv(file_path, sep=None, names=['time', 'misc', 'event_id'])
    # Translates the time of all events so that the first of them happens
    # at time 0
    events['time'] = events['time'] - events['time'][0]
    events = np.array(events)
    # Converts the actual time of each event to the corresponding timestep,
    # depending on the sampling frequency
    events_timesteps = np.round(sampling_freq * events[:, 0])
    events[:, 0] = events_timesteps
    return events.astype(int)