"""
Sample data builders for testing open_dvm.analysis.EYE (EYE and
SaccadeDetector classes).

Two kinds of builders are provided:
- Trial-dict builders (`make_trial`) for unit-testing methods that
  operate directly on the parsed trial-dict contract (interp_trial,
  get_xy, ...), without needing real files on disk.
- Raw-file builders (`write_asc_session`, `write_beh_csv`) for
  integration-testing get_eye_data/link_eye_to_eeg end to end.
"""

import numpy as np


def make_trial(x, y, trackertime, msgs=None, blinks=None):
    """Build a single trial dict matching the eye_readers contract."""
    return {
        'x': np.asarray(x, dtype=float),
        'y': np.asarray(y, dtype=float),
        'trackertime': np.asarray(trackertime, dtype=int),
        'events': {
            'msg': msgs if msgs is not None else [],
            'Eblk': blinks if blinks is not None else [],
        },
    }


def _asc_sample(t, x, y, pupil):
    return f"{t}\t{x}\t{y}\t{pupil}\t...\n"


def _asc_msg(t, message):
    return f"MSG\t{t} {message}\n"


def _asc_eblink(start, end):
    return f"EBLINK R {start}\t{end}\t{end - start}\n"


def write_asc_session(path, sj=1, ses=1, n_trials=2, trigger_msg='Onset search',
                       start='start trial', rec_lead_ms=300, trial_len_ms=200,
                       blink=None):
    """
    Write a synthetic multi-trial EyeLink .asc file with a real
    filename convention (sub_<sj>_ses_<ses>.asc), suitable for both
    direct path use and glob-based 'all' discovery.

    Each trial: recording starts `rec_lead_ms` before its trigger
    (ample margin for blink interpolation), runs for `trial_len_ms`
    after the trigger, with a constant gaze position of (500, 400).
    `blink`, if given, is a (rel_start, rel_end) tuple in ms relative
    to the trigger, inserted into every trial.
    """
    lines = []
    t0 = 1000
    for tr in range(1, n_trials + 1):
        rec_start = t0
        trigger_t = rec_start + rec_lead_ms
        rec_end = trigger_t + trial_len_ms
        lines.append(_asc_msg(rec_start, f'{start}: {tr}'))
        for t in range(rec_start, rec_end):
            if t == trigger_t:
                lines.append(_asc_msg(trigger_t, trigger_msg))
            if blink is not None:
                b_start, b_end = trigger_t + blink[0], trigger_t + blink[1]
                if t == b_start:
                    lines.append(_asc_eblink(b_start, b_end))
                if b_start <= t <= b_end:
                    lines.append(_asc_sample(t, 0.0, 0.0, 0.0))
                    continue
            lines.append(_asc_sample(t, 500.0, 400.0, 5000.0))
        t0 = rec_end + 500

    with open(path, 'w') as f:
        f.writelines(lines)


def write_beh_csv(path, n_trials=2, extra_cols=None):
    """Write a minimal behavioral .csv with `n_trials` rows."""
    import pandas as pd
    data = {'nr_trials': list(range(1, n_trials + 1))}
    if extra_cols:
        data.update(extra_cols)
    pd.DataFrame(data).to_csv(path, index=False)


def fixation_saccade_profile(sfreq=1000, pre_ms=200, sac_ms=20, post_ms=200,
                              start_xy=(500.0, 400.0), end_xy=(700.0, 400.0),
                              noise_amp=0.0, seed=0):
    """
    Build a synthetic gaze trace: a still fixation, a linear (roughly
    constant-velocity) saccade to a new position, then a still
    fixation -- for exercising SaccadeDetector's velocity-threshold
    pipeline. Returns (x, y) arrays.
    """
    rng = np.random.RandomState(seed)
    n_pre = int(pre_ms * sfreq / 1000)
    n_sac = int(sac_ms * sfreq / 1000)
    n_post = int(post_ms * sfreq / 1000)

    x = np.concatenate([
        np.full(n_pre, start_xy[0]),
        np.linspace(start_xy[0], end_xy[0], n_sac),
        np.full(n_post, end_xy[0]),
    ])
    y = np.concatenate([
        np.full(n_pre, start_xy[1]),
        np.linspace(start_xy[1], end_xy[1], n_sac),
        np.full(n_post, end_xy[1]),
    ])
    if noise_amp:
        x = x + rng.normal(0, noise_amp, x.size)
        y = y + rng.normal(0, noise_amp, y.size)
    return x, y
