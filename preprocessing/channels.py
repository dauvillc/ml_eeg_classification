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
                'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6'],
    'parietal':['TP7', 'TP8',
                'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6',
                'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8',
                'PO7', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'PO8'],
    'occipital':['PO7', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'PO8',
                 'O1', 'OZ', 'O2'],
    'right temporal': ['FT7', 'T7', 'TP7', 'FT8', 'T8', 'TP8',
                       'FC6', 'C6', 'CP6', 'FC4', 'C4', 'CP4'],
    'left temporal': ['FT7', 'T7', 'TP7', 'FT8', 'T8', 'TP8',
                      'FC5', 'C5', 'CP5', 'FC3', 'C3', 'CP3']
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


def select_electrodes_groups(epochs, cha_names, groups, verbose=True):
    """
    Selects a group of electrodes from a specific region of the brain.
    :param epochs: array of shape (N_epochs, N_channels, N_timesteps)
    :param cha_names: list or array of shape (N_channels) giving
        the name of each channel.
    :param groups: str or list of str, electrodes groups (areas of the brain) to
        keep. The possible areas are 'all', 'frontal', 'central', 'parietal',
        'occipital', 'right temporal' and 'left temporal".
    :param verbose: well that's the verbose parameter
    :return: an array of shape (N_epochs, N_channels_selected, N_timesteps)
        containing the channels of the epochs which were selected.
    """
    # Converts to a list if a single str was passed
    if isinstance(groups, str):
        groups = [groups]
    if 'all' in groups:
        return epochs
    if verbose:
        print(f'Selecting electrodes groups {" - ".join(groups)}')
    # Since an electrode can be included in several groups,
    # we use a set to avoid including it multiple times in the returned
    # epochs
    kept_channels = set()
    for group in groups:
        electrodes = _CHANNELS_GROUPS_[group]
        for electrode in electrodes:
            kept_channels.add(electrode)

    selected_channels = [i for i in range(epochs.shape[1])
                         if cha_names[i] in kept_channels]
    return epochs[:, selected_channels]
