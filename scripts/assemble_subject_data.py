"""
Groups together the data from a subject for all days.
Should be executed after the data for each day has been assembled
using assemble_daily_data.py.
"""
import os
import numpy as np

_DATA_DIR_ = "ready_data"
_SUBJECT_ = '01'

if __name__ == "__main__":
    # Loads the epochs and labels for each day, for
    # the specified subject
    all_epochs, all_labels = [], []
    for day in range(1, 6):
        data_path = os.path.join(_DATA_DIR_,
                            f"data_Sub{_SUBJECT_}_day{day}.np")
        labels_path = os.path.join(_DATA_DIR_,
                                   f"labels_Sub{_SUBJECT_}_day{day}.np")
        with open(data_path, "rb") as data_file:
            all_epochs.append(np.load(data_file))
        with open(labels_path, "rb") as labels_file:
            all_labels.append(np.load(labels_file))
        print(f'Loaded data for day {day}')

    # All sets of epochs do not have exactly the same length.
    # Thus we cut them at the min length
    min_epoch_len = min(ep.shape[2] for ep in all_epochs)
    all_epochs = [epochs[:, :, :min_epoch_len] for epochs in all_epochs]

    # We can now concatenate those lists into numpy arrays
    # and save them
    all_epochs = np.concatenate(all_epochs, axis=0)
    all_labels = np.concatenate(all_labels)

    save_path = os.path.join(_DATA_DIR_,
                             f'data_Sub{_SUBJECT_}_dayall.np')
    with open(save_path, "wb") as savefile:
        np.save(savefile, all_epochs)
    save_path = os.path.join(_DATA_DIR_,
                             f'labels_Sub{_SUBJECT_}_dayall.np')
    with open(save_path, "wb") as savefile:
        np.save(savefile, all_labels)
