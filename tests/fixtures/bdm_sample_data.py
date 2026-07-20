"""
Sample data fixtures for testing open_dvm.analysis.BDM functionality.
"""

import mne
import numpy as np
import pandas as pd


def make_separable_epochs(
    n_trials: int = 80,
    n_ch: int = 4,
    n_samples: int = 50,
    sfreq: float = 100,
    seed: int = 0,
    separable_from_sample=None,
    label_column: str = "label",
    cnd_column: str = "block_type",
    cnd_value: str = "main",
):
    """
    Build synthetic epochs + behavioral dataframe for BDM decoding tests.

    Labels are drawn randomly (0/1). If separable_from_sample is given,
    channel 0 gets a strong, constant offset for label==1 trials from
    that sample index onward -- making decoding accuracy trivially
    checkable (chance before the offset starts, near-perfect after).

    Parameters
    ----------
    n_trials, n_ch, n_samples, sfreq, seed : see np.random usage below
    separable_from_sample : int, optional
        Sample index from which the separating signal is injected. If
        None, no signal is injected (pure noise, chance-level decoding
        expected everywhere).
    label_column, cnd_column, cnd_value : str
        Column names/values for the returned behavioral dataframe.

    Returns
    -------
    epochs : mne.EpochsArray
    df : pd.DataFrame
    """
    rng = np.random.default_rng(seed)
    ch_names = [f"Ch{i+1}" for i in range(n_ch)]
    info = mne.create_info(ch_names, sfreq, ch_types="eeg")
    labels = rng.integers(0, 2, size=n_trials)
    data = rng.normal(0, 1, size=(n_trials, n_ch, n_samples))
    if separable_from_sample is not None:
        data[labels == 1, 0, separable_from_sample:] += 5.0
    epochs = mne.EpochsArray(data, info, tmin=-0.1)
    df = pd.DataFrame({label_column: labels, cnd_column: [cnd_value] * n_trials})
    return epochs, df


def make_localizer_epoch_pair(
    n_trials_tr: int = 60,
    n_trials_te: int = 40,
    n_ch: int = 4,
    n_samples: int = 50,
    sfreq: float = 100,
    seed_tr: int = 1,
    seed_te: int = 2,
    label_column: str = "label",
):
    """
    Build a (train_epochs, train_df, test_epochs, test_df) tuple for
    localizer_classify tests -- two independent, strongly-separable
    synthetic epoch sets (matching the BDM(epochs=[tr, te], df=[tr, te])
    constructor convention).
    """
    epochs_tr, df_tr = make_separable_epochs(
        n_trials=n_trials_tr,
        n_ch=n_ch,
        n_samples=n_samples,
        sfreq=sfreq,
        seed=seed_tr,
        separable_from_sample=0,
        label_column=label_column,
        cnd_value="loc",
    )
    epochs_te, df_te = make_separable_epochs(
        n_trials=n_trials_te,
        n_ch=n_ch,
        n_samples=n_samples,
        sfreq=sfreq,
        seed=seed_te,
        separable_from_sample=0,
        label_column=label_column,
        cnd_value="loc",
    )
    return epochs_tr, df_tr, epochs_te, df_te


def make_cross_condition_priming_epochs(
    n_trials_tr: int = 60,
    n_trials_te: int = 200,
    n_ch: int = 4,
    n_samples: int = 50,
    sfreq: float = 100,
    separable_from_sample: int = 25,
    seed: int = 0,
    label_column: str = "label",
    priming_column: str = "prev_label",
    cnd_column: str = "block_type",
    train_cnd: str = "localizer",
    test_cnd: str = "main",
):
    """
    Build a single combined (epochs, df) pair for classify()'s
    single-object cross-condition cnds format, for testing special_col.

    Train-condition trials: `label_column` genuinely drives the
    injected channel-0 signal (as in make_separable_epochs).

    Test-condition trials: `label_column` is an uncorrelated
    placeholder (decoding on it alone should sit at chance), while
    `priming_column` is what actually drives the injected signal --
    mimicking e.g. a previous-trial location that a decoder trained on
    the current-trial location might still pick up on.
    """
    rng = np.random.default_rng(seed)
    ch_names = [f"Ch{i+1}" for i in range(n_ch)]
    info = mne.create_info(ch_names, sfreq, ch_types="eeg")

    label_tr = rng.integers(0, 2, size=n_trials_tr)
    data_tr = rng.normal(0, 1, size=(n_trials_tr, n_ch, n_samples))
    data_tr[label_tr == 1, 0, separable_from_sample:] += 5.0

    label_te = rng.integers(0, 2, size=n_trials_te)
    priming_te = rng.integers(0, 2, size=n_trials_te)
    data_te = rng.normal(0, 1, size=(n_trials_te, n_ch, n_samples))
    data_te[priming_te == 1, 0, separable_from_sample:] += 5.0

    data = np.concatenate([data_tr, data_te], axis=0)
    epochs = mne.EpochsArray(data, info, tmin=-0.1)

    df = pd.DataFrame(
        {
            label_column: np.concatenate([label_tr, label_te]),
            priming_column: np.concatenate([label_tr, priming_te]),
            cnd_column: [train_cnd] * n_trials_tr + [test_cnd] * n_trials_te,
        }
    )
    return epochs, df
