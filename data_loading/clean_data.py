"""
Defines functions to load the cleaned data.
"""
import os
import h5py
import numpy as np

_DEFAULT_CLEAN_DATA_PATH_ = os.path.join("data", "clean_data")
_DEFAULT_OUT_DIR_ = "ready_data"


def assemble_clean_data(day, out_dir=_DEFAULT_OUT_DIR_, path=_DEFAULT_CLEAN_DATA_PATH_):
    """
    Loads the clean data from the .mat files and formats it.
    :param day: int or str, which day's data to load
    :param out_dir: directory to which the arrays are saved.
    :param path: path to the .mat files
    """
    matfile = f'clean_data_sub_01_Day-{str(day)}.mat'
    with h5py.File(os.path.join(path, matfile), 'r') as f:
        # The H5 file actually contains references to where the arrays are stored
        references = f['clean_data']['trial'][:, 0]
        # We can retrieve the time series of each epoch by using the reference
        epochs = [
            np.array(f[ref]) for ref in references
            ]

        # Retrieves the sampling times. Those go from -2 sec to 6 seconds.
        # Times 0 to 5s correspond to the trials
        # We also apply baseline correction: we substract the average of the signals
        # between -1.5s and -1s.
        times = [np.array(f[ref][:, 0]) for ref in f['clean_data']['time'][:, 0]]
        for k, (epoch, time) in enumerate(zip(epochs, times)):
            baseline = (time >= -1.5) & (time <= -1)
            baseline_avg = np.mean(epoch[baseline], axis=0)
            epochs[k] -= baseline_avg
            epochs[k] = epoch[(time >= 0) & (time <= 5)]

        # Stacks the epochs into a single np array of shape (N_ep, N_timesteps, N_channels)
        epochs = np.stack(epochs)
        # Swaps the time and channel axis to be coherent with the ML pipeline
        epochs = np.swapaxes(epochs, 1, -1)

        # Loads the labels associated with the epochs
        labels = np.array(f['clean_data']['trialinfo']).flatten()
        # The labels are given as 9 or 11, but we need to convert them to 0 and 1
        labels = (labels == 11).astype(int)

        print(f"Loaded mat file {matfile} into an array of shape {epochs.shape}")

    with open(os.path.join(out_dir, f"clean_data_sub01_day{day}.np"), "wb") as savefile:
        np.save(savefile, epochs)
    with open(os.path.join(out_dir, f"clean_labels_sub01_day{day}.np"), "wb") as sfile:
        np.save(sfile, labels)

    return epochs, labels


# If executed, loads the .mat file directly
if __name__ == "__main__":
    all_days_epochs, all_days_labels = [], []
    out_dir = _DEFAULT_OUT_DIR_
    for day in range(1, 6):
        # Loads the clean data from a single day and
        # writes it into np files
        epochs, labels = assemble_clean_data(day, out_dir=out_dir)
        all_days_epochs.append(epochs)
        all_days_labels.append(labels)
    # Assembles all days together
    all_days_epochs = np.concatenate(all_days_epochs, axis=0)
    all_days_labels = np.concatenate(all_days_labels, axis=0)

    print(f"Saving the clean data for all days, of shape {all_days_epochs.shape}")
    with open(os.path.join(out_dir, f"clean_data_sub01_dayall.np"), "wb") as savefile:
        np.save(savefile, epochs)
    with open(os.path.join(out_dir, f"clean_labels_sub01_dayall.np"), "wb") as sfile:
        np.save(sfile, labels)

