"""
Sample data builders for testing open_dvm.analysis.CTF.

These construct synthetic mne.EpochsArray + behavioral DataFrame pairs
with a genuine, decodable spatial signal (channel i biased when the
trial's position bin == i), so that CTF/IEM reconstruction has a real,
checkable pattern to recover rather than pure noise.
"""

import mne
import numpy as np
import pandas as pd


def make_spatial_epochs(
    nr_bins=8,
    n_ch=8,
    n_trials_per_bin=50,
    n_samples=15,
    sfreq=100,
    signal_amp=3.0,
    noise_sd=1.0,
    seed=0,
    label_column="position",
):
    """
    Build synthetic epochs + behavioral dataframe with a genuine spatial
    signal: channel (bin % n_ch) is biased by `signal_amp` on trials
    whose position bin == bin, for every bin in range(nr_bins).

    Returns
    -------
    epochs : mne.EpochsArray
    df : pd.DataFrame with a single `label_column` column
    """
    rng = np.random.default_rng(seed)
    ch_names = [f"Ch{i+1}" for i in range(n_ch)]
    info = mne.create_info(ch_names, sfreq, ch_types="eeg")

    pos = np.repeat(np.arange(nr_bins), n_trials_per_bin)
    rng.shuffle(pos)
    n_trials = pos.size

    data = rng.normal(0, noise_sd, size=(n_trials, n_ch, n_samples))
    for tr in range(n_trials):
        data[tr, pos[tr] % n_ch, :] += signal_amp

    epochs = mne.EpochsArray(data, info, tmin=-0.1, verbose=False)
    df = pd.DataFrame({label_column: pos})
    return epochs, df


def make_localizer_and_ping(
    nr_bins=8,
    n_ch=8,
    n_trials_per_bin=50,
    n_ping=200,
    n_samples=15,
    sfreq=100,
    signal_amp=3.0,
    noise_sd=1.0,
    ping_special_loc=2,
    seed=0,
    label_column="position",
    cnd_column="task",
):
    """
    Build a combined localizer+ping dataset for cross-condition CTF
    testing: 'localizer' trials have real, varied position bins with a
    genuine decodable signal; 'ping' trials are homogeneous (no true
    position) but carry the SAME channel-bias pattern as bin
    `ping_special_loc`, simulating a neutral probe display biased
    toward a hypothesized reference location. The ping rows'
    `label_column` values are left at an arbitrary placeholder (0), on
    the assumption tests will use `special_loc` to override them.

    Returns
    -------
    epochs : mne.EpochsArray (localizer + ping trials concatenated)
    df : pd.DataFrame with `label_column` and `cnd_column` columns
    """
    rng = np.random.default_rng(seed)
    ch_names = [f"Ch{i+1}" for i in range(n_ch)]
    info = mne.create_info(ch_names, sfreq, ch_types="eeg")

    pos_loc = np.repeat(np.arange(nr_bins), n_trials_per_bin)
    rng.shuffle(pos_loc)
    n_loc = pos_loc.size
    data_loc = rng.normal(0, noise_sd, size=(n_loc, n_ch, n_samples))
    for tr in range(n_loc):
        data_loc[tr, pos_loc[tr] % n_ch, :] += signal_amp

    pos_ping = np.zeros(n_ping, dtype=int)  # placeholder, override via special_loc
    data_ping = rng.normal(0, noise_sd, size=(n_ping, n_ch, n_samples))
    data_ping[:, ping_special_loc % n_ch, :] += signal_amp

    data = np.concatenate([data_loc, data_ping], axis=0)
    pos = np.concatenate([pos_loc, pos_ping])
    task = np.array(["localizer"] * n_loc + ["ping"] * n_ping)

    epochs = mne.EpochsArray(data, info, tmin=-0.1, verbose=False)
    df = pd.DataFrame({label_column: pos, cnd_column: task})
    return epochs, df
