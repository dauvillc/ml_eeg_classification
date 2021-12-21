"""
Defines functions related to channels, such as selecting specific channels.
"""

# Define which channels should be excluded from the analysis
# by default
_UNINTERESTING_CHANNELS_ = set(['M1', 'M2'])


_CHANNELS_GROUPS_ = {
    'frontal': ['FP1', 'FPZ', 'FP2', 'F7', 'F5', 'F3', 'F1', 'FZ',
                'F2', 'F4', 'F6', 'F8', 'FT7', 'FC5', 'FC3', 'FC1',
                'FCZ', 'FC2', 'FC4', 'FC6', 'FC8'],
    'central': ['FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'C5'
                'C3', 'C1', 'CZ', 'C2', 'C4', 'C6',
                'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6']
}


def select_channels(raw, exclude=_UNINTERESTING_CHANNELS_):
    """
    Selects only the interesting EEG channels in a Raw object.
    :param raw: MNE Raw object containing the data, modified inplace.
    :param exclude: list of channels names that should be excluded from
        the features.
    """
    good_channels = set(raw.ch_names) - set(exclude)
    raw.pick('eeg')
    raw.pick(list(good_channels))


def select_electrodes_group(epochs, cha_names, group_name, verbose=True):
    """
    Selects a group of electrodes from a specific region of the brain.
    :param epochs: array of shape (N_epochs, N_channels, N_timesteps)
    :param cha_names: list or array of shape (N_channels) giving
        the name of each channel.
    :param group_name: One of 'frontal', 'central', 'parietal',
        'occipital', 'right temporal' or 'left temporal";
        Group of electrode to select.
    :param verbose: well that's the verbose parameter
    :return: an array of shape (N_epochs, N_channels_selected, N_timesteps)
        containing the channels of the epochs which were selected.
    """
    if verbose:
        print(f'Selecting electrodes group {group_name}')
    group_channels_names = _CHANNELS_GROUPS_[group_name]

    selected_channels = [i for i in range(epochs.shape[1])
                         if cha_names[i] in group_channels_names]
    return epochs[:, selected_channels]
