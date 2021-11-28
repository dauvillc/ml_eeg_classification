"""
Clément Dauvilliers - EPFL - ML Project 2
A first script to explore the FIF files.
"""
import mne
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

FIF_PATH = os.path.join('data', 'fif_data', 'offline', 'fif',
                        '20210812-113427-WS-default-raw.fif')
EVE_PATH = os.path.join('data', 'fif_data', 'offline',
                        '20210812-113427-eve.txt')

if __name__ == "__main__":
    raw = mne.io.read_raw_fif(FIF_PATH, preload=True)
    print(raw.info)

    # Extracting events
    events = pd.read_csv(EVE_PATH, sep=None, names=['time', 'misc', 'event_id'])
    events['time'] = events['time'] - events['time'][0]

    # Actual times of the events
    events_times = np.array(events['time'])
    # Timesteps
    # The time difference between each timestep is equal,
    # and is 3.33333.. ms
    time_diff = raw.times[1] - raw.times[0]
    events_timesteps = np.round(events_times / time_diff).astype(int)
    events['time'] = events_timesteps
    events = np.array(events)


    raw = raw.pick(['P3', 'C3', 'F3', 'Fz', 'F4', 'C4', 'P4',
                    'Cz', 'Pz', 'Fp1', 'Fp2', 'T3', 'T5', 'O1',
                    'O2', 'F7', 'F8', 'T6', 'T4'])
    raw.plot(events=events, start=0, duration=40, color='gray', scalings='auto',
             event_color={9: 'r', 10: 'g', 11: 'b', 12: 'm', 13: 'y', 14: 'k', 15:'c'})
    plt.show()
