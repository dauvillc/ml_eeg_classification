"""
Clément Dauvilliers - EPFL - ML Project 2
A first script to explore the FIF files.
"""
import mne
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy as sp
import cv2

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
    
    
    """ EMMA - I added the part you wrote in the notebook for epoching & retrieving groundtruth labels
    and what I wrote, but I'm not sure it does what its supposed to do """
    
    # Bandpass filtering in frequency range 1-100 Hz
    raw.filter(1, 100)

    # Notch filtering
    raw.notch_filter(50)

    # Re-referencing to EEG average
    raw.set_eeg_reference('average', projection=True)

    # Plot pre-processed signals
    raw.plot(events=events, start=0, duration=10, color='gray', scalings=1e2,
             event_color={9: 'r', 10: 'g', 11: 'b', 12: 'm', 13: 'y', 14: 'k', 15: 'c'},
             title='Preprocessed EEG signals')
    plt.show()
    
    # Retrieves the timesteps at which a 14 event happens
    epoch_starts = events[np.where(events[:, 2] == 14), 0][0]
    # Same for event 13
    epoch_ends = events[np.where(events[:, 2] == 13), 0][0]
    # Extracts each epoch and puts them into a list using the previous arrays
    epochs = [raw.get_data()[:, st:end + 1] for st, end in zip(epoch_starts, epoch_ends)]
    # Cuts each epoch at the minimum length among all epochs
    epochs_lengths = [ep.shape[1] for ep in epochs]
    epochs = [ep[:, :min(epochs_lengths)] for ep in epochs]
    # Assembles the epochs into an array of shape (n_epochs, n_channels, n_timesteps)
    epochs = np.stack(epochs)

    # Returns all events 12 or 10 that happened, in chronological order
    label_events = events[np.where(np.isin(events[:, 2], [12, 10])), 2][0]
    # Converts all the labels to 'ghi'=0 or 'fo'=1
    epochs_true_labels = (label_events == 12).astype(int)
    
    # Transformation between electrode pairs for each epoch: abs(FFT(Ei)-FFT(Ej))
    for z in range(epochs.shape[0]):
        data = (abs(sp.ifft(epochs[z, 0, :]) - sp.ifft(epochs[z, 1, :]))).T
        for i in range(1, epochs.shape[1]):
            for j in range(i, epochs.shape[1]):
                col = (abs(sp.ifft(epochs[z, i, :]) - sp.ifft(epochs[z, j, :]))).T
                data = np.column_stack((data, col))
        # Normalization between 0 and 1
        data = (data - np.min(data)) / (np.max(data) - np.min(data))
        # Saves images
        IMG_PATH = os.path.join('SZ_dry_D3-speechmiAdaptive', 'offline', 'images')
        cv2.imwrite(f"{IMG_PATH}\\image%d.jpg" % z, data)
