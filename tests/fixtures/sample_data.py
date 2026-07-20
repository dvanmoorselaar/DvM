"""
Sample data fixtures for testing open_dvm functionality.

This module provides utilities for generating synthetic EEG data suitable
for testing ERP analysis functions.
"""

from typing import Any, Dict, Tuple

import mne
import numpy as np
import pandas as pd
from mne import EpochsArray, create_info


def create_sample_epochs(
    n_trials: int = 100,
    n_channels: int = 64,
    sfreq: int = 500,
    n_samples: int = 256,
    seed: int = 42,
) -> Tuple[mne.Epochs, np.ndarray]:
    """
    Create synthetic EEG epochs for testing.

    Parameters
    ----------
    n_trials : int, default=100
        Number of epochs (trials) to generate.
    n_channels : int, default=64
        Number of EEG channels (standard 10-20 system).
    sfreq : int, default=500
        Sampling frequency in Hz.
    n_samples : int, default=256
        Number of time samples per epoch.
    seed : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    epochs : mne.Epochs
        Synthetic EEG epochs.
    data : np.ndarray
        Raw epoch data (n_trials, n_channels, n_samples).
    """
    np.random.seed(seed)

    # Create standard 10-20 channel names
    montage = mne.channels.make_standard_montage("standard_1020")
    ch_names = montage.ch_names[:n_channels]

    # Generate synthetic EEG data (white noise + signal)
    data = np.random.randn(n_trials, n_channels, n_samples) * 10  # 10 µV std dev

    # Add a negative component (like N2pc) to some channels
    # Peak at ~200 ms
    peak_idx = int(0.2 * sfreq)  # 200 ms
    peak_width = int(0.05 * sfreq)  # 50 ms width

    # Add N2pc-like component to posterior-contralateral channels
    component_channels = ["PO3", "PO4", "O1", "O2"]
    for ch in component_channels:
        if ch in ch_names:
            ch_idx = ch_names.index(ch)
            # Add negative component (roughly -2 µV)
            data[:, ch_idx, peak_idx - peak_width : peak_idx + peak_width] -= 2

    # Create info structure
    info = create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")

    # Create Epochs object
    times = np.linspace(-0.2, 0.4, n_samples)  # -200 to 400 ms
    epochs = EpochsArray(data, info, tmin=-0.2)

    return epochs, data


def create_sample_erp_dataframe(n_trials: int = 100, seed: int = 42) -> pd.DataFrame:
    """
    Create a sample dataframe with trial metadata for ERP analysis.

    Parameters
    ----------
    n_trials : int, default=100
        Number of trials.
    seed : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    df : pd.DataFrame
        Dataframe with columns: trial, condition, target_loc, dist_loc, correct, rt
    """
    np.random.seed(seed)

    df = pd.DataFrame(
        {
            "trial": np.arange(n_trials),
            "condition": np.random.choice(["target_present", "target_absent"], n_trials),
            "target_loc": np.random.choice([1, 2, 3, 4, 5, 6, 7, 8], n_trials),  # Clock positions
            "dist_loc": np.random.choice([1, 2, 3, 4, 5, 6, 7, 8], n_trials),
            "dist2_loc": np.random.choice([0, 1, 2, 3, 4, 5, 6, 7, 8], n_trials),  # 0 = absent
            "correct": np.random.choice([True, False], n_trials, p=[0.8, 0.2]),
            "rt": np.random.uniform(400, 1000, n_trials),
        }
    )

    return df


def create_sample_waveforms() -> Dict[str, np.ndarray]:
    """
    Create synthetic waveforms for latency analysis testing.

    Returns
    -------
    waveforms : dict
        Dictionary with keys 'absent' and 'present' containing sample waveforms.
    """
    times = np.linspace(0, 0.4, 100)
    n_trials = 20

    # Create realistic N2pc-like waveforms
    np.random.seed(42)

    # "Absent" condition: peaks slightly earlier
    x_absent = np.random.randn(n_trials, len(times)) * 2
    peak_idx = 50
    for i in range(n_trials):
        # Add individual variability to peak
        peak_var = np.random.normal(0, 2)
        x_absent[i, peak_idx - 5 : peak_idx + 5] = -4 + peak_var

    # "Present" condition: peaks slightly later (higher latency)
    x_present = np.random.randn(n_trials, len(times)) * 2
    peak_idx = 52
    for i in range(n_trials):
        peak_var = np.random.normal(0, 2)
        x_present[i, peak_idx - 5 : peak_idx + 5] = -3.5 + peak_var

    waveforms = {"absent": x_absent, "present": x_present}

    return waveforms, times


def create_lateralization_test_data() -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Create test data specifically for lateralization/spatial restriction testing.

    Returns spatial restriction test data with known indices that should pass filters.

    Returns
    -------
    data : np.ndarray
        Dummy EEG data (10 trials, 2 channels, 100 samples).
    trial_info : pd.DataFrame
        Trial metadata for testing spatial restrictions.
    """
    np.random.seed(42)

    # 10 trials
    data = np.random.randn(10, 2, 100)

    # Create trial info with specific location patterns for testing AND logic
    trial_info = pd.DataFrame(
        {
            "trial": np.arange(10),
            "dist1_loc": [0, 0, 0, 0, 1, 2, 3, 4, 5, 6],
            "dist2_loc": [0, 4, 1, 2, 4, 4, 4, 4, 4, 4],
            "target_loc": [2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
        }
    )

    # Expected AND logic results for {'dist1_loc': [0], 'dist2_loc': [4]}:
    # Trials where dist1_loc == 0 AND dist2_loc == 4: indices [1] (1 trial)

    return data, trial_info


def create_lateralized_flip_epochs() -> Tuple[mne.Epochs, pd.DataFrame]:
    """
    Create deterministic epochs for testing ERP.flip_topography.

    Data is constant across time so the only thing that can change a
    value is the flip itself: for trial ``t`` and channel index ``c``,
    every sample equals ``t * 10 + c``. This makes the expected
    post-flip values trivial to compute by hand.

    Returns
    -------
    epochs : mne.Epochs
        3 trials x 5 channels (O1, O2, P7, P8, HEOG) x 10 samples.
    trial_info : pd.DataFrame
        Contains 'target_loc' with trials 0 and 2 marked as left (1)
        and trial 1 marked as right (2).
    """
    ch_names = ["O1", "O2", "P7", "P8", "HEOG"]
    n_trials, n_samples, sfreq = 3, 10, 100

    data = np.zeros((n_trials, len(ch_names), n_samples))
    for t in range(n_trials):
        for c in range(len(ch_names)):
            data[t, c, :] = t * 10 + c

    info = mne.create_info(ch_names, sfreq, ch_types=["eeg", "eeg", "eeg", "eeg", "eog"])
    epochs = mne.EpochsArray(data, info, tmin=-0.05)
    trial_info = pd.DataFrame({"target_loc": [1, 2, 1]})

    return epochs, trial_info


def create_peaked_erps(
    peak_times: Tuple[float, float] = (0.1, 0.2),
    peak_polarities: Tuple[int, int] = (1, -1),
    n_subjects: int = 5,
    seed: int = 42,
) -> Dict[str, list]:
    """
    Create two conditions of synthetic evoked data with known peaks.

    Used for testing peak-detection/latency methods (select_erp_window,
    compare_latencies) where the true peak location must be known in
    advance to check the result.

    Returns
    -------
    erps : dict
        {'cond1': [mne.Evoked, ...], 'cond2': [mne.Evoked, ...]}, each
        with a +/-5 uV spike at the specified peak time (plus small
        per-subject noise so jackknife variance is non-zero) on a
        2-channel ('Cz', 'Pz') montage spanning -0.1 to 0.3 s at 100 Hz.
    """
    rng = np.random.default_rng(seed)
    ch_names = ["Cz", "Pz"]
    sfreq = 100
    tmin, tmax = -0.1, 0.3
    n_samples = int(round((tmax - tmin) * sfreq)) + 1
    times = np.round(np.linspace(tmin, tmax, n_samples), 8)
    info = mne.create_info(ch_names, sfreq, ch_types="eeg")

    erps = {}
    for cnd, peak_t, sign in zip(["cond1", "cond2"], peak_times, peak_polarities):
        peak_idx = int(np.argmin(np.abs(times - peak_t)))
        evokeds = []
        for _ in range(n_subjects):
            d = rng.normal(0, 0.1, size=(len(ch_names), n_samples))
            d[:, peak_idx] += sign * 5
            evokeds.append(mne.EvokedArray(d, info, tmin=tmin))
        erps[cnd] = evokeds

    return erps


def create_biosemi64_evoked_pair() -> list:
    """
    Create two evoked objects on a real biosemi64 montage with a known
    contra/ipsi amplitude difference, for testing
    ERP.group_lateralized_erp without depending on montage-name luck.

    O1=5, O2=2, P7=3, P8=1 (all other channels/timepoints are 0), so
    the expected contra ('O1','P7') minus ipsi ('O2','P8') difference,
    averaged over those two electrode pairs, is (3 + 2) / 2 = 2.5.

    Returns
    -------
    list of mne.Evoked
        Two (identical) subject-level evoked objects.
    """
    ch_names = mne.channels.make_standard_montage("biosemi64").ch_names
    sfreq, n_samples = 100, 10
    info = mne.create_info(ch_names, sfreq, ch_types="eeg")

    data = np.zeros((len(ch_names), n_samples))
    data[ch_names.index("O1")] = 5
    data[ch_names.index("O2")] = 2
    data[ch_names.index("P7")] = 3
    data[ch_names.index("P8")] = 1

    return [mne.EvokedArray(data.copy(), info, tmin=-0.05) for _ in range(2)]


def create_residual_eye_data() -> Tuple[mne.Epochs, pd.DataFrame, float]:
    """
    Create epochs with a HEOG channel and known per-trial values for
    testing ERP.residual_eye.

    Returns
    -------
    epochs : mne.Epochs
        4 trials x 2 channels ('Cz', 'HEOG') x 20 samples, HEOG constant
        across time per trial.
    trial_info : pd.DataFrame
        'target_loc' marks trials 0, 2 as left (1) and 1, 3 as right (2).
    expected : float
        The residual eye value ERP.residual_eye should compute:
        mean(mean(left HEOG trials), -mean(right HEOG trials)).
    """
    ch_names = ["Cz", "HEOG"]
    sfreq, n_samples = 100, 20
    info = mne.create_info(ch_names, sfreq, ch_types=["eeg", "eog"])

    heog_vals = [2.0, -3.0, 6.0, 1.5]
    data = np.zeros((len(heog_vals), len(ch_names), n_samples))
    for t, val in enumerate(heog_vals):
        data[t, 1, :] = val

    epochs = mne.EpochsArray(data, info, tmin=-0.1)
    trial_info = pd.DataFrame({"target_loc": [1, 2, 1, 2]})

    left_wave = np.mean([heog_vals[0], heog_vals[2]])
    right_wave = np.mean([heog_vals[1], heog_vals[3]])
    expected = float(np.mean([left_wave, right_wave * -1]))

    return epochs, trial_info, expected


def create_multilocation_test_data() -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Create test data with multiple location values for testing AND logic with multiple constraints.

    Returns
    -------
    data : np.ndarray
        Dummy EEG data (12 trials, 2 channels, 100 samples).
    trial_info : pd.DataFrame
        Trial metadata with multiple location values.
    """
    np.random.seed(42)

    data = np.random.randn(12, 2, 100)

    trial_info = pd.DataFrame(
        {
            "trial": np.arange(12),
            "dist1_loc": [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3],
            "dist2_loc": [0, 4, 1, 0, 4, 1, 0, 4, 1, 0, 4, 1],
            "target_loc": [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
        }
    )

    # Expected AND logic results for {'dist1_loc': [0, 1], 'dist2_loc': [0, 4]}:
    # dist1_loc in [0, 1] AND dist2_loc in [0, 4]:
    # Trials 0 (0,0), 1 (0,4), 3 (1,0), 4 (1,4) = indices [0, 1, 3, 4]

    return data, trial_info
