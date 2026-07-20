"""
Sample data fixtures for testing open_dvm.visualization.plot functionality.
"""

import mne
import numpy as np


def make_condition_evokeds(
    amplitudes: dict,
    ch_names=("C3", "C5", "C4", "C6"),
    n_subjects: int = 5,
    sfreq: float = 200,
    tmin: float = -0.1,
    n_samples: int = 60,
    noise_sd: float = 0.05e-6,
    seed: int = 0,
):
    """
    Build a list of per-subject mne.Evoked objects with a known, constant
    (in time) amplitude per channel (in volts), plus small per-subject
    noise so group-level operations (e.g. jackknife variance) don't
    degenerate on identical data.

    Parameters
    ----------
    amplitudes : dict
        channel name -> amplitude in volts. Channels not present default
        to 0.
    """
    rng = np.random.default_rng(seed)
    info = mne.create_info(list(ch_names), sfreq, ch_types="eeg")
    evokeds = []
    for _ in range(n_subjects):
        data = np.zeros((len(ch_names), n_samples))
        for i, ch in enumerate(ch_names):
            data[i, :] = amplitudes.get(ch, 0.0) + rng.normal(0, noise_sd, n_samples)
        evokeds.append(mne.EvokedArray(data, info, tmin=tmin))
    return evokeds


def make_average_tfr(
    ch_names=("C3", "C4"),
    freqs=None,
    amplitude_by_channel: dict = None,
    sfreq: float = 100,
    tmin: float = -0.2,
    n_samples: int = 40,
    nave: int = 10,
):
    """
    Build a single mne.time_frequency.AverageTFRArray with a known,
    constant (over freq/time) power value per channel.
    """
    if freqs is None:
        freqs = np.linspace(4, 30, 8)
    ch_names = list(ch_names)
    info = mne.create_info(ch_names, sfreq, ch_types="eeg")
    times = tmin + np.arange(n_samples) / sfreq
    data = np.zeros((len(ch_names), len(freqs), n_samples))
    if amplitude_by_channel:
        for i, ch in enumerate(ch_names):
            data[i] = amplitude_by_channel.get(ch, 0.0)
    return mne.time_frequency.AverageTFRArray(
        info=info, data=data, times=times, freqs=freqs, nave=nave
    )


def make_bdm_result(
    dec_scores: np.ndarray,
    times: np.ndarray,
    test_times: np.ndarray = None,
    freqs=None,
    cnd: str = "A",
    n_subjects: int = 5,
    noise_sd: float = 0.02,
    seed: int = 0,
):
    """
    Build a list of per-subject BDM result dicts (the structure consumed
    by plot_bdm_timecourse), each with small noise added around a shared
    base dec_scores array.
    """
    rng = np.random.default_rng(seed)
    info = {"times": times}
    if test_times is not None:
        info["test_times"] = test_times
    if freqs is not None:
        info["freqs"] = freqs

    bdms = []
    for _ in range(n_subjects):
        scores = dec_scores + rng.normal(0, noise_sd, dec_scores.shape)
        bdms.append({cnd: {"dec_scores": scores}, "info": dict(info)})
    return bdms


def make_ctf_result(
    raw_slopes: np.ndarray,
    times: np.ndarray,
    bands,
    cnd: str = "A",
    n_subjects: int = 5,
    noise_sd: float = 0.02,
    seed: int = 0,
):
    """
    Build a list of per-subject CTF result dicts (the structure consumed
    by plot_ctf_timecourse), each with small noise added around a shared
    base raw_slopes array. raw_slopes shape convention matches
    plot_ctf_timecourse's internal np.stack: (n_bands, n_times) or
    (n_bands, n_times, n_bins).
    """
    rng = np.random.default_rng(seed)
    ctfs = []
    for _ in range(n_subjects):
        slopes = raw_slopes + rng.normal(0, noise_sd, raw_slopes.shape)
        ctfs.append(
            {
                cnd: {"raw_slopes": slopes},
                "info": {"times": times, "bands": bands},
            }
        )
    return ctfs
