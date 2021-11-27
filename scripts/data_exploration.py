"""
Clément Dauvilliers - EPFL - ML Project 2
A first script to explore the FIF files.
"""
import mne
import os
import numpy as np
import matplotlib.pyplot as plt

FIF_PATH = os.path.join('data', 'fif_data', 'offline', 'fif',
                        '20210812-113427-WS-default-raw.fif')
EVE_PATH = os.path.join('data', 'fif_data', 'offline',
                        '20210812-113427-eve.txt')

if __name__ == "__main__":
    raw = mne.io.read_raw_fif(FIF_PATH, preload=True)
    print(raw.info)
    raw = raw.pick(['P3', 'C3', 'F3', 'Fz', 'F4', 'C4', 'P4',
                    'Cz', 'Pz', 'Fp1', 'Fp2', 'T3', 'T5', 'O1',
                    'O2', 'F7', 'F8', 'T6', 'T4'])
    raw.pick('eeg')
    events = mne.read_events(EVE_PATH)
    # for chan in t3_chan.get_data():
    #    plt.plot(t3_chan.times, chan, linewidth=0.1)
    epochs = mne.Epochs(raw, events, event_id=14, tmin=0, baseline=(0, 0), tmax=4.5, preload=True)
    epochs.plot(scalings='auto')
