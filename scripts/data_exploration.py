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
from data_loading.events import load_eve_file
from data_loading.epochs import split_to_epochs

FIF_PATH = os.path.join('data', 'fif_data', 'offline', 'fif',
                        '20210812-113427-WS-default-raw.fif')
EVE_PATH = os.path.join('data', 'fif_data', 'offline',
                        '20210812-113427-eve.txt')

if __name__ == "__main__":
    raw = mne.io.read_raw_fif(FIF_PATH, preload=True)
    print(raw.info)

    # Extracting events
    events = load_eve_file(EVE_PATH, 300)


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

    # Splits the channels into epochs and retrieves the true labels
    epochs, epochs_true_labels = split_to_epochs(raw, events)

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
