"""
Sample data fixtures for testing open_dvm.analysis.TFR functionality.
"""

import mne
import numpy as np
import pandas as pd


def make_epochs(
    ch_names=("C3", "C4"),
    n_trials=10,
    n_samples=100,
    sfreq=200.0,
    tmin=-0.2,
    seed=0,
):
    """Plain noise epochs with no particular oscillatory content."""
    info = mne.create_info(list(ch_names), sfreq, ch_types="eeg")
    rng = np.random.default_rng(seed)
    data = rng.normal(0, 1, (n_trials, len(ch_names), n_samples))
    return mne.EpochsArray(data, info, tmin=tmin)


def make_oscillating_epochs(
    freq,
    ch_names=("C3", "C4"),
    n_trials=10,
    n_samples=200,
    sfreq=200.0,
    tmin=-0.2,
    amplitude=5.0,
    noise_sd=0.1,
    phase_locked=True,
    seed=0,
):
    """
    Epochs with a known sinusoidal oscillation at `freq` Hz on every
    channel, for testing whether TFR power correctly peaks there.

    If phase_locked=True, every trial has the identical oscillation
    (usable for testing induced-power subtraction: removing the evoked
    average should leave near-zero residual). If False, each trial
    gets an independent random phase (oscillation survives in total
    power but averages out of the evoked response).
    """
    rng = np.random.default_rng(seed)
    info = mne.create_info(list(ch_names), sfreq, ch_types="eeg")
    t = np.arange(n_samples) / sfreq
    data = np.zeros((n_trials, len(ch_names), n_samples))
    for i in range(n_trials):
        phase = 0.0 if phase_locked else rng.uniform(0, 2 * np.pi)
        signal = amplitude * np.sin(2 * np.pi * freq * t + phase)
        for c in range(len(ch_names)):
            data[i, c] = signal + rng.normal(0, noise_sd, n_samples)
    return mne.EpochsArray(data, info, tmin=tmin)


def make_behavioral_df(n_trials, **columns):
    """
    Simple behavioral dataframe. Each keyword is a column; values are
    used as-is if already list-like, otherwise repeated n_trials times.
    """
    data = {}
    for key, val in columns.items():
        if isinstance(val, (list, np.ndarray)):
            data[key] = val
        else:
            data[key] = [val] * n_trials
    return pd.DataFrame(data)
