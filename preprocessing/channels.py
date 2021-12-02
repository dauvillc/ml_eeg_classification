"""
Defines functions related to channels, such as selecting specific channels.
"""
import mne

# Define which channels should be excluded from the analysis
_UNINTERESTING_CHANNELS_ = set(['M1', 'M2', 'Trigger'])

def select_channels(raw):
    """
    Selects only the interesting EEG channels in a Raw object.
    :param raw: MNE Raw object containing the data, modified inplace.
    """
    good_channels = set(raw.ch_names) - set(_UNINTERESTING_CHANNELS_)
    raw.pick('eeg')
    raw.pick(list(good_channels))
