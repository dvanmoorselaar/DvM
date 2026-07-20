"""
Sample data fixtures for testing open_dvm.support.FolderStructure.

All writer functions take an explicit `root` directory and write into
the standard DvM folder convention (eeg/processed, behavioral/raw,
erp/evoked, tfr/<method>, bdm/<path>, ctf/<path>) beneath it. Callers
are expected to have already chdir'd (or monkeypatch.chdir'd) into
`root`, matching FolderStructure.folder_tracker's use of os.getcwd().
"""

import os
import pickle

import mne
import numpy as np
import pandas as pd


def write_epochs(
    root, sj, fname, modality="eeg", n_trials=5, n_ch=2, sfreq=100, metadata=None, seed=0
):
    """Write a synthetic -epo.fif file to {root}/{modality}/processed/."""
    folder = os.path.join(root, modality, "processed")
    os.makedirs(folder, exist_ok=True)
    ch_names = [f"Ch{i+1}" for i in range(n_ch)]
    info = mne.create_info(ch_names, sfreq, ch_types="eeg")
    data = np.random.default_rng(seed).normal(0, 1, (n_trials, n_ch, 20))
    epochs = mne.EpochsArray(data, info, tmin=-0.1)
    if metadata is not None:
        epochs.metadata = metadata
    path = os.path.join(folder, f"sub_{sj}_{fname}-epo.fif")
    epochs.save(path, overwrite=True)
    return path


def write_processed_beh(root, sj, fname, df):
    """Write a behavioral CSV to {root}/behavioral/processed/."""
    folder = os.path.join(root, "behavioral", "processed")
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"sub_{sj}_{fname}.csv")
    df.to_csv(path, index=False)
    return path


def write_raw_beh(root, sj, session, suffix, df):
    """Write a raw behavioral CSV to {root}/behavioral/raw/."""
    folder = os.path.join(root, "behavioral", "raw")
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"sub_{sj}_ses_{session}_{suffix}.csv")
    df.to_csv(path, index=False)
    return path


def write_evoked(root, sj, cnd, erp_name, evoked):
    """Write an Evoked object to {root}/erp/evoked/."""
    folder = os.path.join(root, "erp", "evoked")
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"sub_{sj}_{cnd}_{erp_name}-ave.fif")
    evoked.save(path, overwrite=True)
    return path


def make_evoked(ch_names=("Cz",), sfreq=100, n_samples=10, value=0.0, tmin=-0.1):
    info = mne.create_info(list(ch_names), sfreq, ch_types="eeg")
    data = np.full((len(ch_names), n_samples), value, dtype=float)
    return mne.EvokedArray(data, info, tmin=tmin)


def write_tfr(root, tfr_folder_path, sj, tfr_name, cnd, tfr):
    """Write an AverageTFR object to {root}/tfr/{tfr_folder_path}/."""
    folder = os.path.join(root, "tfr", *tfr_folder_path)
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"sub_{sj}_{tfr_name}_{cnd}-tfr.h5")
    tfr.save(path, overwrite=True)
    return path


def make_tfr(ch_names=("Cz",), freqs=None, n_samples=10, sfreq=100, nave=5):
    if freqs is None:
        freqs = np.linspace(4, 20, 4)
    info = mne.create_info(list(ch_names), sfreq, ch_types="eeg")
    data = np.zeros((len(ch_names), len(freqs), n_samples))
    times = -0.1 + np.arange(n_samples) / sfreq
    return mne.time_frequency.AverageTFRArray(
        info=info, data=data, times=times, freqs=freqs, nave=nave
    )


def write_bdm_pickle(root, bdm_folder_path, sj, bdm_name, data):
    """Write a BDM results dict to {root}/bdm/{bdm_folder_path}/."""
    folder = os.path.join(root, "bdm", *bdm_folder_path)
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"sub_{sj}_{bdm_name}.pickle")
    with open(path, "wb") as f:
        pickle.dump(data, f)
    return path


def write_ctf_pickle(root, ctf_folder_path, sj, ctf_name, output, data):
    """Write a CTF results object to {root}/ctf/{ctf_folder_path}/."""
    folder = os.path.join(root, "ctf", *ctf_folder_path)
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"sub_{sj}_{ctf_name}_{output}.pickle")
    with open(path, "wb") as f:
        pickle.dump(data, f)
    return path
