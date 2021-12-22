import numpy as np
import os

def load_data(data_dir, subject, day, use_clean_data):
    """
    Loads the pre-assembled data, either raw or clean.
    The data must be assembled using the corresponding script beforehand
    (assemble_subject_data, assemble_daily_data or clean_data.py).
    :param data_dir: directory in which are contained the .np files.
    :param subject: Either '01' or '02'.
    :param day: int or str between 1 and 5 included. Can also be 'all' to
        load the data from all five days combined.
    :param use_clean_data: boolean. If True, loads the clean data.
    :return: (Epochs, labels) which are two ndarrays of shapes
        (N_epochs, N_channels, N_timesteps) and (N_epochs,).
    """
    if use_clean_data:
        epochs_path = os.path.join(data_dir, f"clean_data_sub{subject}_day{day}.np")
        labels_path = os.path.join(data_dir, f"clean_labels_sub{subject}_day{day}.np")
    else:
        epochs_path = os.path.join(data_dir, "data_Sub" + subject + "_day" + day + ".np")
        labels_path = os.path.join(data_dir, "labels_Sub" + subject + "_day" + day + ".np")
    with open(epochs_path, "rb") as epochs_file:
        epochs = np.load(epochs_file)
        print(f"Loaded the epochs of shape {epochs.shape}")
    with open(labels_path, "rb") as labels_file:
        labels = np.load(labels_file)
        print(f"Loaded the labels of shape {labels.shape}")
    return epochs, labels


def load_channels_names(data_dir):
    """
    Loads the names of each of the 62 channels.
    :param data_dir: directory in which the data is assembled.
    :return: an array of strings giving the name of each electrode.
    """
    with open(os.path.join(data_dir, "channels_names.np") , "rb") as channels_file:
        ch_names = np.load(channels_file)
    return ch_names