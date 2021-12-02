"""
Assembles the data from a given subject and day.
Usage:
    - set the Subject and the day in the constants at the beginning.
    - check the output path
    - execute without arguments from the root directory
Creates two files output_path/data_{subject}_{day}.np
                  output_path/labels_{subject}_{day}.np
"""
import mne
import os
import numpy as np
from data_loading.epochs import split_to_epochs, assemble_epochs
from data_loading.events import load_eve_file
from preprocessing import select_channels

_SUBJECT_ = '01'
_DAY_ = '1'
_OUTPUT_DIR_ = 'ready_data'
_DATA_DIR_ = os.path.join('data', 'Data_ML_Internship')
_SAMPLING_FREQ_ = 512

if __name__ == "__main__":
    # Compute the path to the subject and day data
    input_path = os.path.join(_DATA_DIR_, 'Sub_' + _SUBJECT_,
                              'Sub_' + _SUBJECT_ + '_day' + str(_DAY_) + '-speechmi',
                              'offline')
    print(f"Data repertory:\n{input_path}")
    if not os.path.exists(input_path):
        raise ValueError(f"Data repertory not found:\n{input_path}")

    # For each of the three experiments, load the EEG data from the fif file
    # and the event from the -eve.txt file
    fif_path = os.path.join(input_path, 'fif')
    raws_objects, events_list = [], []
    # Raw data
    for fif_file in os.listdir(fif_path):
        # Ignore all non-fif files (such as the channelsList.txt file)
        if fif_file.endswith('-raw.fif'):
            # Load the raw data
            fif_file = os.path.join(fif_path, fif_file)
            raw = mne.io.read_raw_fif(fif_file, preload=True)
            ###########################################################
            # APPLY HERE ALL NON-GLOBAL PREPROCESSING
            # (ex: bandpass filtering)

            # Picks only the right channels for the analysis
            select_channels(raw)
            ###########################################################
            raws_objects.append(raw)
    # Events
    for eve_file in os.listdir(input_path):
        # Ignore all non-events files
        if eve_file.endswith('-eve.txt'):
            # Load the events data
            eve_file = os.path.join(input_path, eve_file)
            events_list.append(load_eve_file(eve_file, _SAMPLING_FREQ_))
    print(f"Found {len(raws_objects)} fif files and {len(events_list)} events files.")

    # Create the epochs from each Raw object
    # split_to_epochs returns a couple (epochs, true labels)
    epochs_and_labels = [split_to_epochs(raw, events) for raw, events in zip(raws_objects, events_list)]
    epochs = [epoch for epoch, _ in epochs_and_labels]
    labels = [label for _, label in epochs_and_labels]
    # Assembles all epochs into a single array
    epochs = assemble_epochs(epochs)
    labels = np.concatenate(labels)
    print(f"Loaded {epochs.shape[0]} epochs in total with {epochs.shape[1]} channels and " +
          f"{epochs.shape[2]} timesteps")

    # Save the epochs and labels array in the output path
    # If the output dir doesn't exist, it is created
    if not os.path.exists(_OUTPUT_DIR_):
        os.makedirs(_OUTPUT_DIR_)
    output_path_epochs = os.path.join(_OUTPUT_DIR_, f'data_Sub{_SUBJECT_}_day{_DAY_}.np')
    output_path_labels = os.path.join(_OUTPUT_DIR_, f'labels_Sub{_SUBJECT_}_day{_DAY_}.np')
    print(f"Saving epochs to {output_path_epochs}")
    with open(output_path_epochs, 'wb') as epochs_file:
        np.save(epochs_file, epochs)
    print(f"Saving labels to {output_path_labels}")
    with open(output_path_labels, 'wb') as labels_file:
        np.save(labels_file, labels)
