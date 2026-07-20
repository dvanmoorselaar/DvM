"""
Sample data fixtures for testing open_dvm.analysis.EEG functionality.
"""

import os

import mne
import numpy as np
import pandas as pd

from open_dvm.analysis.EEG import RAW, Epochs


def make_synthetic_raw(
    ch_names: list,
    ch_types,
    data: np.ndarray = None,
    sfreq: float = 250,
    n_samples: int = 1000,
    seed: int = 0,
) -> RAW:
    """
    Build a RAW instance from synthetic in-memory data, without going
    through file I/O.

    RAW's __init__ only supports reading from disk, and its own
    constructor does nothing beyond `self.__dict__.update(raw.__dict__)`
    on a freshly-read mne Raw object. Reassigning __class__ on an
    already-built mne.io.RawArray is equivalent in effect and avoids a
    real file round-trip for tests that don't care about file-reading
    itself.

    Parameters
    ----------
    ch_names : list of str
        Channel names.
    ch_types : str or list of str
        MNE channel type(s), e.g. 'eeg' or ['eeg', 'eeg', 'stim'].
    data : np.ndarray, optional
        Data array of shape (n_channels, n_samples). If None, random
        Gaussian data scaled to volts (~1e-5) is generated.
    sfreq : float, default=250
        Sampling frequency in Hz.
    n_samples : int, default=1000
        Number of time samples (ignored if data is provided).
    seed : int, default=0
        Random seed for reproducibility.

    Returns
    -------
    raw : RAW
    """
    info = mne.create_info(ch_names, sfreq, ch_types=ch_types)
    if data is None:
        rng = np.random.default_rng(seed)
        data = rng.normal(0, 1e-5, size=(len(ch_names), n_samples))
    raw = mne.io.RawArray(data, info)
    raw.__class__ = RAW
    return raw


def make_synthetic_raw_with_stim(
    trigger_samples_values: list,
    n_channels: int = 2,
    sfreq: float = 250,
    n_samples: int = 1000,
    baseline_value: float = 0.0,
) -> RAW:
    """
    Build a synthetic RAW with EEG channels plus a stim channel
    carrying the given trigger codes at the given sample indices.

    Parameters
    ----------
    trigger_samples_values : list of (int, float)
        (sample_index, trigger_value) pairs written onto the stim
        channel; all other samples are set to baseline_value.
    n_channels : int, default=2
        Number of EEG channels to include alongside the stim channel.
    sfreq : float, default=250
        Sampling frequency in Hz.
    n_samples : int, default=1000
        Number of time samples.
    baseline_value : float, default=0.0
        Value written to all non-trigger stim samples. Use a nonzero
        value (e.g. matching a `binary` offset) to emulate hardware
        (e.g. BioSemi) status-channel conventions where the trigger
        channel never actually rests at zero.

    Returns
    -------
    raw : RAW
    """
    stim = np.full(n_samples, float(baseline_value))
    for sample, value in trigger_samples_values:
        stim[sample] = value

    ch_names = [f"F{i+1}" for i in range(n_channels)] + ["STI"]
    ch_types = ["eeg"] * n_channels + ["stim"]
    rng = np.random.default_rng(0)
    eeg_data = rng.normal(0, 1e-5, size=(n_channels, n_samples))
    data = np.vstack([eeg_data, stim[None, :]])

    return make_synthetic_raw(ch_names, ch_types, data=data, sfreq=sfreq)


def make_synthetic_epochs(
    triggers: list,
    event_id,
    sj: int = 1,
    session: int = 1,
    ch_names: list = None,
    ch_types="eeg",
    sfreq: float = 250,
    tmin: float = -0.2,
    tmax: float = 0.2,
    flt_pad=None,
    n_samples: int = 6000,
    seed: int = 0,
) -> Epochs:
    """
    Build an Epochs instance from synthetic continuous data with events
    evenly spaced across the recording (with margin for tmin/tmax/flt_pad).

    Parameters
    ----------
    triggers : list of int
        Trigger value for each event, in event order.
    event_id : int, list, or dict
        Passed straight through to Epochs.__init__.
    sj, session : int, default=1
        Subject/session numbers.
    ch_names : list of str, optional
        Defaults to ['F1', 'F2', 'Cz'].
    ch_types : str or list, default='eeg'
    sfreq : float, default=250
    tmin, tmax : float
        Epoch window relative to event, in seconds.
    flt_pad : float or tuple, optional
    n_samples : int, default=6000
        Total continuous recording length in samples.
    seed : int, default=0

    Returns
    -------
    epochs : Epochs
    """
    if ch_names is None:
        ch_names = ["F1", "F2", "Cz"]

    info = mne.create_info(ch_names, sfreq, ch_types=ch_types)
    rng = np.random.default_rng(seed)
    data = rng.normal(0, 1e-5, size=(len(ch_names), n_samples))
    raw = mne.io.RawArray(data, info)

    n_events = len(triggers)
    margin = int(0.3 * n_samples / max(n_events, 1)) + 500
    sample_positions = np.linspace(margin, n_samples - margin, n_events).astype(int)
    events = np.zeros((n_events, 3), dtype=int)
    events[:, 0] = sample_positions
    events[:, 2] = triggers

    return Epochs(
        sj,
        session,
        raw,
        events,
        event_id=event_id,
        tmin=tmin,
        tmax=tmax,
        flt_pad=flt_pad,
        baseline=None,
    )


def write_behavioral_csv(sj: int, session: int, df: pd.DataFrame, name: str = "main"):
    """
    Write a behavioral CSV file to the location Epochs.align_meta_data
    (via FolderStructure.read_raw_beh) expects to find it, relative to
    the current working directory. Callers must chdir into a temp
    project folder first (e.g. via monkeypatch.chdir(tmp_path)).
    """
    beh_dir = os.path.join("behavioral", "raw")
    os.makedirs(beh_dir, exist_ok=True)
    fname = os.path.join(beh_dir, f"sub_{sj:02d}_ses_{session:02d}_{name}.csv")
    df.to_csv(fname, index=False)
    return fname


def make_artefact_epochs(
    n_ch: int = 8,
    n_epochs: int = 10,
    sfreq: float = 500,
    n_samples: int = 500,
    tmin: float = -0.5,
    seed: int = 0,
    inject_noise: bool = False,
) -> mne.EpochsArray:
    """
    Build synthetic epochs on a real standard_1020 montage (required
    for interpolate_bads to work) for testing ArtefactReject.

    Parameters
    ----------
    n_ch : int, default=8
        Number of channels (first N of standard_1020).
    n_epochs : int, default=10
    sfreq : float, default=500
    n_samples : int, default=500
    tmin : float, default=-0.5
    seed : int, default=0
    inject_noise : bool, default=False
        If True, injects a strong 120Hz burst (within the default
        110-140Hz muscle-artifact detection band) into channels 1-4,
        samples 200:300, of epoch index 3 -- large and broad enough
        across channels to reliably clear the data-driven z-threshold
        (verified empirically; a single-channel or whole-epoch-duration
        burst is too weak/diluted to trigger detection at this sample
        size).

    Returns
    -------
    epochs : mne.EpochsArray
    """
    montage = mne.channels.make_standard_montage("standard_1020")
    ch_names = montage.ch_names[:n_ch]
    info = mne.create_info(ch_names, sfreq, ch_types="eeg")
    rng = np.random.default_rng(seed)
    data = rng.normal(0, 1e-6, size=(n_epochs, n_ch, n_samples))
    epochs = mne.EpochsArray(data.copy(), info, tmin=tmin)
    epochs.set_montage(montage)

    if inject_noise:
        t = np.arange(n_samples) / sfreq
        burst_mask = slice(200, 300)
        burst = np.sin(2 * np.pi * 120 * t[burst_mask]) * 0.02
        for ch in [1, 2, 3, 4]:
            epochs._data[3, ch, burst_mask] += burst

    return epochs


def make_eog_correlated_raw(
    n_ch: int = 6,
    sfreq: float = 250,
    n_samples: int = 30000,
    seed: int = 2,
) -> mne.io.RawArray:
    """
    Build synthetic continuous raw data with one EEG channel driven by
    a shared "blink-like" signal that's also present (amplified) on a
    dedicated EOG channel, for testing automated_ica_blink_selection's
    has-EOG-channels path.
    """
    ch_names = [f"Ch{i+1}" for i in range(n_ch)] + ["EOG1"]
    info = mne.create_info(ch_names, sfreq, ch_types=["eeg"] * n_ch + ["eog"])
    rng = np.random.default_rng(seed)
    eeg_data = rng.normal(0, 1e-5, size=(n_ch, n_samples))
    blink = rng.normal(0, 1e-5, size=n_samples)
    eeg_data[0] += blink * 3
    eog_data = blink[None, :] * 5
    raw_data = np.vstack([eeg_data, eog_data])
    return mne.io.RawArray(raw_data, info)
