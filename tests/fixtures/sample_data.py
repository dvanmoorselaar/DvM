"""
Sample data fixtures for testing open_dvm functionality.

This module provides utilities for generating synthetic EEG data suitable
for testing ERP analysis functions.
"""

import numpy as np
import pandas as pd
import mne
from mne import create_info, EpochsArray
from typing import Tuple, Dict, Any


def create_sample_epochs(
    n_trials: int = 100,
    n_channels: int = 64,
    sfreq: int = 500,
    n_samples: int = 256,
    seed: int = 42
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
    montage = mne.channels.make_standard_montage('standard_1020')
    ch_names = montage.ch_names[:n_channels]

    # Generate synthetic EEG data (white noise + signal)
    data = np.random.randn(n_trials, n_channels, n_samples) * 10  # 10 µV std dev

    # Add a negative component (like N2pc) to some channels
    # Peak at ~200 ms
    peak_idx = int(0.2 * sfreq)  # 200 ms
    peak_width = int(0.05 * sfreq)  # 50 ms width

    # Add N2pc-like component to posterior-contralateral channels
    component_channels = ['PO3', 'PO4', 'O1', 'O2']
    for ch in component_channels:
        if ch in ch_names:
            ch_idx = ch_names.index(ch)
            # Add negative component (roughly -2 µV)
            data[:, ch_idx, peak_idx - peak_width:peak_idx + peak_width] -= 2

    # Create info structure
    info = create_info(
        ch_names=ch_names,
        sfreq=sfreq,
        ch_types='eeg'
    )

    # Create Epochs object
    times = np.linspace(-0.2, 0.4, n_samples)  # -200 to 400 ms
    epochs = EpochsArray(data, info, tmin=-0.2)

    return epochs, data


def create_sample_erp_dataframe(
    n_trials: int = 100,
    seed: int = 42
) -> pd.DataFrame:
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

    df = pd.DataFrame({
        'trial': np.arange(n_trials),
        'condition': np.random.choice(['target_present', 'target_absent'], n_trials),
        'target_loc': np.random.choice([1, 2, 3, 4, 5, 6, 7, 8], n_trials),  # Clock positions
        'dist_loc': np.random.choice([1, 2, 3, 4, 5, 6, 7, 8], n_trials),
        'dist2_loc': np.random.choice([0, 1, 2, 3, 4, 5, 6, 7, 8], n_trials),  # 0 = absent
        'correct': np.random.choice([True, False], n_trials, p=[0.8, 0.2]),
        'rt': np.random.uniform(400, 1000, n_trials),
    })

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
        x_absent[i, peak_idx - 5:peak_idx + 5] = -4 + peak_var

    # "Present" condition: peaks slightly later (higher latency)
    x_present = np.random.randn(n_trials, len(times)) * 2
    peak_idx = 52
    for i in range(n_trials):
        peak_var = np.random.normal(0, 2)
        x_present[i, peak_idx - 5:peak_idx + 5] = -3.5 + peak_var

    waveforms = {
        'absent': x_absent,
        'present': x_present
    }

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
    trial_info = pd.DataFrame({
        'trial': np.arange(10),
        'dist1_loc': [0, 0, 0, 0, 1, 2, 3, 4, 5, 6],
        'dist2_loc': [0, 4, 1, 2, 4, 4, 4, 4, 4, 4],
        'target_loc': [2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
    })

    # Expected AND logic results for {'dist1_loc': [0], 'dist2_loc': [4]}:
    # Trials where dist1_loc == 0 AND dist2_loc == 4: indices [1] (1 trial)

    return data, trial_info


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

    trial_info = pd.DataFrame({
        'trial': np.arange(12),
        'dist1_loc': [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3],
        'dist2_loc': [0, 4, 1, 0, 4, 1, 0, 4, 1, 0, 4, 1],
        'target_loc': [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
    })

    # Expected AND logic results for {'dist1_loc': [0, 1], 'dist2_loc': [0, 4]}:
    # dist1_loc in [0, 1] AND dist2_loc in [0, 4]:
    # Trials 0 (0,0), 1 (0,4), 3 (1,0), 4 (1,4) = indices [0, 1, 3, 4]

    return data, trial_info
