"""
Sample raw eye-tracker file builders for testing
open_dvm.support.eye_readers.

These construct synthetic .asc (EyeLink EDF-derived ASCII) and .tsv
(EyeTribe) file content matching the real formats closely enough to
exercise the parsers, without needing real recorded data.
"""


def _asc_event(prefix9, *fields):
    """Build an EDF event line: a fixed 9-character prefix (e.g.
    'EBLINK R ') followed directly by tab-separated values."""
    assert len(prefix9) == 9, f"prefix must be exactly 9 chars: {prefix9!r}"
    return prefix9 + '\t'.join(str(f) for f in fields) + '\n'


def _asc_sample(t, x, y, pupil):
    return f"{t}\t{x}\t{y}\t{pupil}\t...\n"


def _asc_msg(t, message):
    return f"MSG\t{t} {message}\n"


def write_asc_two_trials(path, missing_sample=True):
    """
    Two trials, segmented purely by trial-start markers (stop=None).
    Trial 1 has a missing sample (pupil=0.0) and one blink; trial 1
    also has fixation/saccade/blink-start summary lines that should be
    ignored. Trial 2 has a second blink. A TRIALID message in each
    trial carries the trial number.
    """
    lines = []
    lines.append(_asc_msg(1000, 'start_trial: 1'))
    lines.append(_asc_sample(1001, 500.0, 400.0, 5000.0))
    if missing_sample:
        lines.append(_asc_sample(1002, 501.0, 401.0, 0.0))  # missing (pupil=0)
    lines.append(_asc_event('EBLINK R ', 1003, 1010, 7, '...'))
    lines.append(_asc_sample(1011, 505.0, 402.0, 5010.0))
    lines.append(_asc_msg(1012, 'TRIALID 5'))
    lines.append(_asc_event('SFIX R   ', 1013))
    lines.append(_asc_event('EFIX R   ', 1013, 1020, 7, 500.0, 400.0))
    lines.append(_asc_event('SSACC R  ', 1021))
    lines.append(_asc_event('ESACC R  ', 1021, 1030, 9, 500.0, 400.0, 520.0, 420.0))
    lines.append(_asc_event('SBLINK R ', 1031))

    lines.append(_asc_msg(2000, 'start_trial: 2'))
    lines.append(_asc_sample(2001, 600.0, 300.0, 6000.0))
    lines.append(_asc_event('EBLINK R ', 2003, 2015, 12, '...'))
    lines.append(_asc_sample(2016, 605.0, 302.0, 6005.0))
    lines.append(_asc_msg(2017, 'TRIALID 6'))
    lines.append(_asc_sample(2018, 606.0, 303.0, 6010.0))

    with open(path, 'w') as f:
        f.writelines(lines)


def write_asc_overlapping_trials(path):
    """
    Two trials using explicit start_trial/stop_trial markers, where
    trial 2 starts before trial 1 stops (the read_edf_time_overlap
    use case).
    """
    lines = []
    lines.append(_asc_msg(3000, 'start_trial: 1'))
    lines.append(_asc_sample(3001, 700.0, 500.0, 7000.0))
    lines.append(_asc_msg(3005, 'start_trial: 2'))
    lines.append(_asc_sample(3006, 701.0, 501.0, 7010.0))
    lines.append(_asc_event('EBLINK R ', 3007, 3009, 2, '...'))
    lines.append(_asc_msg(3010, 'stop_trial: 1'))
    lines.append(_asc_sample(3011, 702.0, 502.0, 7020.0))
    lines.append(_asc_msg(3015, 'stop_trial: 2'))

    with open(path, 'w') as f:
        f.writelines(lines)


def write_asc_mismatched_trial(path):
    """A start/stop pair whose trial numbers don't match."""
    lines = [
        _asc_msg(100, 'start_trial: 1'),
        _asc_sample(101, 1.0, 1.0, 100.0),
        _asc_msg(110, 'stop_trial: 2'),
    ]
    with open(path, 'w') as f:
        f.writelines(lines)


def _tsv_sample(t, x, y):
    return (f"2024-01-01 00:00:00.000\t{t}\tFalse\t7\t"
            f"{x}\t{y}\t{x}\t{y}\t16.0\n")


def _tsv_msg(t, message):
    return f"MSG\t2024-01-01 00:00:00.000\t{t}\t{message}\n"


def write_tsv_two_trials(path):
    """
    Two trials (stop=None, split purely on trial-start markers).
    Trial 1 has a 12-sample missing run (>= the 10-sample blink
    threshold) and trial 2 has only a 3-sample missing run (below
    threshold, should not be detected as a blink).
    """
    lines = []
    lines.append(_tsv_msg(1000, 'start_trial: 1'))
    for i in range(5):
        lines.append(_tsv_sample(1001 + i, 500.0 + i, 400.0 + i))
    for i in range(12):
        lines.append(_tsv_sample(1010 + i, 0.0, 0.0))
    for i in range(5):
        lines.append(_tsv_sample(1030 + i, 510.0 + i, 410.0 + i))

    lines.append(_tsv_msg(2000, 'start_trial: 2'))
    for i in range(3):
        lines.append(_tsv_sample(2001 + i, 600.0 + i, 300.0 + i))
    for i in range(3):
        lines.append(_tsv_sample(2010 + i, 0.0, 0.0))
    for i in range(3):
        lines.append(_tsv_sample(2020 + i, 610.0 + i, 310.0 + i))

    with open(path, 'w') as f:
        f.writelines(lines)


def write_tsv_with_stop(path):
    """Two trials using explicit start/stop markers."""
    lines = []
    lines.append(_tsv_msg(1000, 'start_trial: 1'))
    for i in range(3):
        lines.append(_tsv_sample(1001 + i, 500.0 + i, 400.0 + i))
    lines.append(_tsv_msg(1010, 'stop_trial: 1'))
    lines.append(_tsv_msg(2000, 'start_trial: 2'))
    for i in range(3):
        lines.append(_tsv_sample(2001 + i, 600.0 + i, 300.0 + i))
    lines.append(_tsv_msg(2010, 'stop_trial: 2'))

    with open(path, 'w') as f:
        f.writelines(lines)
