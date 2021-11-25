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


if __name__ == "__main__":
    raw = mne.io.read_raw_fif(FIF_PATH)
    print(raw.info)
    t3_chan = raw.pick(['P3', 'C3', 'F3', 'Fz', 'F4', 'C4', 'P4',
                        'Cz', 'Pz', 'Fp1', 'Fp2', 'T3', 'T5', '01',
                        '02', 'F7', 'F8', 'T6', 'T4'])
    for chan in t3_chan.get_data():
        plt.plot(t3_chan.times, chan, linewidth=0.1)
