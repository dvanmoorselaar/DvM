"""
Synthetic data generators for demonstrating open_dvm's plotting and
statistical-testing functionality.

These functions build minimal, plot-ready ERP/TFR/BDM/CTF objects with a
deliberately strong, time-window-bounded effect (rather than real
experimental data) -- used by tutorials/00_visualization_and_statistics
to show what a clean, statistically reliable result looks like in each
plot style, independent of any real dataset's sample size or power.

Functions
---------
make_condition_evokeds : Synthetic ERP data (list of mne.EvokedArray)
make_average_tfr : Synthetic TFR data (list of mne.time_frequency.AverageTFRArray)
make_bdm_result : Synthetic BDM decoding results (list of per-subject dicts)
make_ctf_result : Synthetic CTF reconstruction results (list of per-subject dicts)
"""

from typing import List, Optional, Tuple, Union

import mne
import numpy as np
from scipy.signal.windows import tukey


def _ramp_envelope(window_len: int, ramp_fraction: float = 0.5) -> np.ndarray:
    """Raised-cosine (Tukey) envelope for a gradual on/offset.

    Ramps smoothly from 0 up to 1 over the first `ramp_fraction / 2` of
    the window, stays flat at 1 in the middle, then ramps back down to 0
    over the last `ramp_fraction / 2` -- used to avoid an abrupt,
    square-wave-shaped effect onset/offset. `ramp_fraction=0` gives a
    hard rectangular step; `ramp_fraction=1` gives a single smooth bump
    with no flat plateau. The envelope is exactly 0 at both endpoints,
    so it connects continuously to the baseline value just outside the
    window.
    """
    if window_len <= 1:
        return np.ones(window_len)
    return tukey(window_len, alpha=ramp_fraction)


def make_condition_evokeds(
    ch_names: Union[list, tuple] = ("C3", "C5", "C4", "C6"),
    contra_ch: Union[list, tuple] = ("C3", "C5"),
    ipsi_ch: Union[list, tuple] = ("C4", "C6"),
    contra_amp: float = 6e-6,
    ipsi_amp: float = 1e-6,
    effect_window: Tuple[int, int] = (20, 35),
    baseline: float = 0.0,
    ramp_fraction: float = 0.5,
    noise_sd: float = 2e-6,
    n_subjects: int = 20,
    sfreq: float = 200,
    tmin: float = -0.1,
    n_samples: int = 60,
    seed: int = 0,
) -> List[mne.EvokedArray]:
    """Build synthetic per-subject ERP data with a time-window-bounded,
    laterality-specific effect.

    Every channel sits flat at `baseline` outside `effect_window`. Inside
    it, channels in `contra_ch` ramp to `contra_amp` and channels in
    `ipsi_ch` to `ipsi_amp` (different values) -- so both the raw
    per-condition waveform and the lateralized contra-minus-ipsi
    difference (`plot_erp_timecourse(..., lateralized=True)`) show a
    genuine, bounded effect rather than a flat, whole-epoch constant. The
    transition in and out of `effect_window` is a smooth ramp (see
    `ramp_fraction`), not an abrupt step.

    Parameters
    ----------
    ch_names : list or tuple
        All channel names in the synthetic montage.
    contra_ch, ipsi_ch : list or tuple
        Subsets of `ch_names` designated as the "contralateral" and
        "ipsilateral" hemisphere for the lateralization demo.
    contra_amp, ipsi_amp : float
        Amplitude (volts) for contra/ipsi channels inside `effect_window`.
    effect_window : tuple of int
        (start, stop) sample indices where the effect is present.
    baseline : float
        Amplitude (volts) everywhere outside `effect_window`.
    ramp_fraction : float, default=0.5
        Fraction of `effect_window` used for the gradual onset/offset
        ramp (raised-cosine), split evenly between the start and end.
        0 gives an abrupt rectangular step; 1 gives a single smooth bump
        with no flat plateau.
    noise_sd : float
        Standard deviation (volts) of per-subject Gaussian noise.
    n_subjects : int
        Number of synthetic subjects (per-subject Evoked objects) to build.
    sfreq : float
        Sampling rate (Hz).
    tmin : float
        Epoch start time (seconds).
    n_samples : int
        Number of time samples per epoch.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    list of mne.EvokedArray
        One Evoked object per synthetic subject.
    """
    rng = np.random.default_rng(seed)
    ch_names = list(ch_names)
    info = mne.create_info(ch_names, sfreq, ch_types="eeg")
    start, stop = effect_window
    start, stop = max(0, start), min(n_samples, stop)
    envelope = _ramp_envelope(stop - start, ramp_fraction)

    base = np.full((len(ch_names), n_samples), baseline, dtype=float)
    for ch in contra_ch:
        if ch in ch_names:
            base[ch_names.index(ch), start:stop] = baseline + (contra_amp - baseline) * envelope
    for ch in ipsi_ch:
        if ch in ch_names:
            base[ch_names.index(ch), start:stop] = baseline + (ipsi_amp - baseline) * envelope

    evokeds = []
    for _ in range(n_subjects):
        data = base + rng.normal(0, noise_sd, base.shape)
        evokeds.append(mne.EvokedArray(data, info, tmin=tmin))
    return evokeds


def make_average_tfr(
    ch_names: Union[list, tuple] = ("C3", "C5", "C4", "C6"),
    contra_ch: Union[list, tuple] = ("C3", "C5"),
    ipsi_ch: Union[list, tuple] = ("C4", "C6"),
    freqs: Optional[np.ndarray] = None,
    contra_amp: float = 3.0,
    ipsi_amp: float = 0.3,
    effect_window: Tuple[int, int] = (20, 35),
    effect_freq_range: Tuple[float, float] = (8, 12),
    baseline: float = 0.0,
    ramp_fraction: float = 0.5,
    noise_sd: float = 0.5,
    n_subjects: int = 20,
    sfreq: float = 100,
    tmin: float = -0.2,
    n_samples: int = 60,
    seed: int = 0,
) -> List[mne.time_frequency.AverageTFRArray]:
    """Build synthetic per-subject TFR data with a time- and
    frequency-window-bounded, laterality-specific power effect.

    Power sits flat at `baseline` everywhere outside the
    (`effect_window`, `effect_freq_range`) box. Inside it, channels in
    `contra_ch` ramp to `contra_amp` and channels in `ipsi_ch` to
    `ipsi_amp` -- mirroring a lateralized alpha-suppression effect, for
    both the raw heatmap and `plot_tfr_timecourse(..., lateralized=True)`.
    The transition in and out of `effect_window` (in time; the frequency
    band edges stay a hard cutoff) is a smooth ramp (see `ramp_fraction`),
    not an abrupt step.

    Parameters
    ----------
    ch_names, contra_ch, ipsi_ch : list or tuple
        See `make_condition_evokeds`.
    freqs : np.ndarray, optional
        Frequency axis (Hz). Defaults to `np.linspace(4, 30, 20)`.
    contra_amp, ipsi_amp : float
        Power (arbitrary units) inside the effect box for contra/ipsi
        channels.
    effect_window : tuple of int
        (start, stop) sample indices of the effect.
    effect_freq_range : tuple of float
        (low, high) Hz bounds of the effect.
    baseline : float
        Power everywhere outside the effect box.
    ramp_fraction : float, default=0.5
        Fraction of `effect_window` used for the gradual onset/offset
        ramp (raised-cosine) in time. See `make_condition_evokeds`.
    noise_sd : float
        Standard deviation of per-subject Gaussian noise.
    n_subjects : int
        Number of synthetic subjects to build.
    sfreq, tmin, n_samples : float, float, int
        Timing parameters for the synthetic epoch.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    list of mne.time_frequency.AverageTFRArray
        One TFR object per synthetic subject.
    """
    rng = np.random.default_rng(seed)
    if freqs is None:
        freqs = np.linspace(4, 30, 20)
    ch_names = list(ch_names)
    info = mne.create_info(ch_names, sfreq, ch_types="eeg")
    times = tmin + np.arange(n_samples) / sfreq
    start, stop = effect_window
    start, stop = max(0, start), min(n_samples, stop)
    freq_mask = (freqs >= effect_freq_range[0]) & (freqs <= effect_freq_range[1])
    envelope = _ramp_envelope(stop - start, ramp_fraction)

    base = np.full((len(ch_names), len(freqs), n_samples), baseline, dtype=float)
    for ch in contra_ch:
        if ch in ch_names:
            idx = ch_names.index(ch)
            base[idx, freq_mask, start:stop] = baseline + (contra_amp - baseline) * envelope
    for ch in ipsi_ch:
        if ch in ch_names:
            idx = ch_names.index(ch)
            base[idx, freq_mask, start:stop] = baseline + (ipsi_amp - baseline) * envelope

    out = []
    for _ in range(n_subjects):
        data = base + rng.normal(0, noise_sd, base.shape)
        out.append(
            mne.time_frequency.AverageTFRArray(
                info=info, data=data, times=times, freqs=freqs, nave=1
            )
        )
    return out


def make_bdm_result(
    effect_window: Tuple[int, int] = (20, 35),
    peak_auc: float = 0.85,
    chance_level: float = 0.5,
    ramp_fraction: float = 0.5,
    noise_sd: float = 0.03,
    n_subjects: int = 20,
    times: Optional[np.ndarray] = None,
    n_samples: int = 60,
    shape: str = "1d",
    freqs: Optional[np.ndarray] = None,
    effect_freq_idx: Optional[int] = None,
    cnd: str = "A",
    seed: int = 0,
) -> list:
    """Build synthetic per-subject BDM decoding results with a
    time-window-bounded above-chance effect.

    Decoding performance sits at `chance_level` outside `effect_window`,
    rising to `peak_auc` inside it, via a smooth ramp rather than an
    abrupt step (see `ramp_fraction`). Matches the per-subject
    result-dict format `plot_bdm_timecourse` consumes directly.

    Parameters
    ----------
    effect_window : tuple of int
        (start, stop) sample indices where decoding is above chance.
    peak_auc : float
        Decoding performance inside `effect_window`.
    chance_level : float
        Decoding performance (chance) everywhere else.
    ramp_fraction : float, default=0.5
        Fraction of `effect_window` used for the gradual onset/offset
        ramp (raised-cosine) in time. See `make_condition_evokeds`. For
        `shape='tfr'`, only the time axis is ramped -- the frequency
        selection stays a hard cutoff.
    noise_sd : float
        Standard deviation of per-subject Gaussian noise.
    n_subjects : int
        Number of synthetic subjects to build.
    times : np.ndarray, optional
        Time axis (seconds). Defaults to `np.linspace(-0.2, 0.8, n_samples)`.
    n_samples : int
        Number of time samples (ignored if `times` is given).
    shape : {'1d', 'gat', 'tfr'}
        Output structure:
        - '1d': a single decoding timecourse.
        - 'gat': a (train_time x test_time) generalization matrix, with
          the effect confined to the block where both train and test
          time fall inside `effect_window` -- for `timecourse='2d_GAT'`.
        - 'tfr': a (frequency x time) matrix, with the effect confined
          to `effect_freq_idx` (default: the middle frequency) crossed
          with `effect_window` -- for `timecourse='2d_tfr'`.
    freqs : np.ndarray, optional
        Frequency axis (Hz), only used when `shape='tfr'`. Defaults to
        `np.linspace(4, 30, 15)`.
    effect_freq_idx : int, optional
        Index into `freqs` where the effect is present, only used when
        `shape='tfr'`. Defaults to the middle frequency.
    cnd : str
        Condition key under which results are stored.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    list of dict
        One `{cnd: {'dec_scores': ...}, 'info': {...}}` dict per subject.
    """
    rng = np.random.default_rng(seed)
    if times is None:
        times = np.linspace(-0.2, 0.8, n_samples)
    else:
        n_samples = len(times)
    start, stop = effect_window
    start, stop = max(0, start), min(n_samples, stop)
    indicator = np.zeros(n_samples)
    indicator[start:stop] = _ramp_envelope(stop - start, ramp_fraction)

    info = {"times": times}

    if shape == "1d":
        base = chance_level + indicator * (peak_auc - chance_level)
    elif shape == "gat":
        base = chance_level + np.outer(indicator, indicator) * (peak_auc - chance_level)
        info["test_times"] = times
    elif shape == "tfr":
        if freqs is None:
            freqs = np.linspace(4, 30, 15)
        info["freqs"] = freqs
        if effect_freq_idx is None:
            effect_freq_idx = len(freqs) // 2
        freq_effect = np.zeros(len(freqs))
        freq_effect[effect_freq_idx] = 1.0
        base = chance_level + np.outer(freq_effect, indicator) * (peak_auc - chance_level)
    else:
        raise ValueError(f"shape must be '1d', 'gat', or 'tfr', got {shape!r}")

    bdms = []
    for _ in range(n_subjects):
        scores = base + rng.normal(0, noise_sd, base.shape)
        bdms.append({cnd: {"dec_scores": scores}, "info": dict(info)})
    return bdms


def make_ctf_result(
    effect_window: Tuple[int, int] = (20, 35),
    peak_slope: float = 0.02,
    baseline_slope: float = 0.0,
    ramp_fraction: float = 0.5,
    noise_sd: float = 0.01,
    n_subjects: int = 20,
    times: Optional[np.ndarray] = None,
    n_samples: int = 60,
    bands: Optional[list] = None,
    shape: str = "1d",
    effect_band_idx: Optional[int] = None,
    cnd: str = "A",
    seed: int = 0,
) -> list:
    """Build synthetic per-subject CTF reconstruction results with a
    time-window-bounded elevated tuning slope.

    Slope sits near `baseline_slope` outside `effect_window`, rising to
    `peak_slope` inside it, via a smooth ramp rather than an abrupt step
    (see `ramp_fraction`). Matches the per-subject result-dict format
    `plot_ctf_timecourse` consumes directly.

    Parameters
    ----------
    effect_window : tuple of int
        (start, stop) sample indices where the slope is elevated.
    peak_slope : float
        Tuning slope inside `effect_window`.
    baseline_slope : float
        Tuning slope everywhere else.
    ramp_fraction : float, default=0.5
        Fraction of `effect_window` used for the gradual onset/offset
        ramp (raised-cosine) in time. See `make_condition_evokeds`. For
        `shape='tfr'`, only the time axis is ramped -- the band
        selection stays a hard cutoff.
    noise_sd : float
        Standard deviation of per-subject Gaussian noise.
    n_subjects : int
        Number of synthetic subjects to build.
    times : np.ndarray, optional
        Time axis (seconds). Defaults to `np.linspace(-0.2, 0.8, n_samples)`.
    n_samples : int
        Number of time samples (ignored if `times` is given).
    bands : list, optional
        Frequency-band labels. Defaults to `['all']` (single band) for
        `shape='1d'`/`'gat'`, or `[[4,8],[8,12],[12,20],[20,30]]` for
        `shape='tfr'`.
    shape : {'1d', 'gat', 'tfr'}
        Output structure:
        - '1d': a single tuning-slope timecourse (one band).
        - 'gat': a (train_time x test_time) matrix, effect confined to
          the block where both times fall inside `effect_window` -- for
          `timecourse='2d_gat'`.
        - 'tfr': a (band x time) matrix, effect confined to
          `effect_band_idx` crossed with `effect_window` -- for
          `timecourse='2d_tfr'`.
    effect_band_idx : int, optional
        Index into `bands` where the effect is present, only used when
        `shape='tfr'`. Defaults to the second band (index 1).
    cnd : str
        Condition key under which results are stored.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    list of dict
        One `{cnd: {'raw_slopes': ...}, 'info': {...}}` dict per subject.
    """
    rng = np.random.default_rng(seed)
    if times is None:
        times = np.linspace(-0.2, 0.8, n_samples)
    else:
        n_samples = len(times)
    start, stop = effect_window
    start, stop = max(0, start), min(n_samples, stop)
    indicator = np.zeros(n_samples)
    indicator[start:stop] = _ramp_envelope(stop - start, ramp_fraction)

    if shape == "1d":
        bands = bands or ["all"]
        base_1band = baseline_slope + indicator * (peak_slope - baseline_slope)
        base = np.tile(base_1band, (len(bands), 1))
        info = {"times": times, "bands": bands}
    elif shape == "gat":
        bands = bands or ["all"]
        if len(bands) != 1:
            raise ValueError("shape='gat' requires a single band")
        gat_2d = baseline_slope + np.outer(indicator, indicator) * (peak_slope - baseline_slope)
        base = gat_2d[np.newaxis]
        info = {"times": times, "bands": bands}
    elif shape == "tfr":
        bands = bands or [[4, 8], [8, 12], [12, 20], [20, 30]]
        if effect_band_idx is None:
            effect_band_idx = min(1, len(bands) - 1)
        base_bands = np.full((len(bands), n_samples), baseline_slope, dtype=float)
        base_bands[effect_band_idx] = baseline_slope + indicator * (peak_slope - baseline_slope)
        base = base_bands  # (n_bands, n_times)
        info = {"times": times, "bands": bands, "freqs": bands}
    else:
        raise ValueError(f"shape must be '1d', 'gat', or 'tfr', got {shape!r}")

    ctfs = []
    for _ in range(n_subjects):
        slopes = base + rng.normal(0, noise_sd, base.shape)
        ctfs.append({cnd: {"raw_slopes": slopes}, "info": dict(info)})
    return ctfs
