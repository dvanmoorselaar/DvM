"""
Eye-tracker raw file readers.

Parses the raw text files produced by eye-tracking software into a
per-trial dict format, for the specific subset of fields this toolbox
actually uses: gaze position, tracker timestamps, trial-relevant
messages, and blink events. Fixation/saccade extraction is not
performed here -- EYE.py implements its own adaptive detector
(SaccadeDetector, Nystrom & Holmqvist 2010) instead.

Trial dict contract
--------------------
Each parsed trial is a dict with keys:

    x            : np.ndarray of x gaze positions
    y            : np.ndarray of y gaze positions
    trackertime  : np.ndarray of sample timestamps (device/tracker time)
    events       : dict with keys
        msg   - list of [time, message] pairs
        Eblk  - list of [start, end, duration] blink events

Adding a new file format later means writing another read_<format>
function that returns this same trial-dict shape.
"""

import os
import re
import warnings

import numpy as np


def _new_trial():
    return {
        "x": [],
        "y": [],
        "trackertime": [],
        "events": {"msg": [], "Eblk": []},
    }


def _finalize_trial(trial):
    trial["x"] = np.array(trial["x"])
    trial["y"] = np.array(trial["y"])
    trial["trackertime"] = np.array(trial["trackertime"])
    return trial


def _extract_trial_nr(messages, trial_info):
    if trial_info is not None:
        for _, msg in messages:
            if trial_info in msg:
                match = re.search(r"\d+", msg)
                if match:
                    return int(match.group())
    return np.nan


def read_edf(filename, start, stop=None, trial_info=None, missing=0.0):
    """
    Parse an EyeLink EDF-derived ASCII (.asc) file into per-trial dicts.

    Trials are segmented on lines containing `start` (or `stop`, if
    given). Sample lines are tab-separated as
    "timestamp\\tx\\ty\\tpupil_size\\t...", with missing gaze samples
    marked by a pupil size of 0.0 (in which case x/y are also zeroed).
    Blink boundaries are read directly from EDF's own SBLINK/EBLINK
    event lines.

    Parameters
    ----------
    filename : str
        Path to the .asc file.
    start : str
        Substring identifying a trial-start line.
    stop : str, optional
        Substring identifying a trial-end line. If None, a trial ends
        as soon as the next trial-start line appears.
    trial_info : str, optional
        Substring identifying a message line that also contains the
        trial number (extracted as the first run of digits in that
        message). If None, or not found for a given trial, that
        trial's entry in the returned trial-number list is NaN.
    missing : float, default=0.0
        Value substituted for missing gaze samples.

    Returns
    -------
    data : list of dict
        One trial dict (see module docstring) per trial.
    trial_nr : list
        Trial number per entry in `data`, parsed via `trial_info`
        (NaN where not found).
    """
    if not os.path.isfile(filename):
        raise FileNotFoundError(f"read_edf: file '{filename}' does not exist")

    with open(filename, "r") as f:
        raw = f.readlines()

    data = []
    trial_nr = []
    trial = _new_trial()
    starttime = 0
    started = False
    finalline = raw[-1] if raw else ""

    for line in raw:
        trial_ended = False

        if started:
            if stop is not None:
                if stop in line:
                    started = False
                    trial_ended = True
            else:
                if (start in line) or (line == finalline):
                    started = True
                    trial_ended = True

            if trial_ended:
                data.append(_finalize_trial(trial))
                trial_nr.append(_extract_trial_nr(trial["events"]["msg"], trial_info))
                trial = _new_trial()
        else:
            if start in line:
                started = True
                starttime = int(line[line.find("\t") + 1 : line.find(" ")])

        if started:
            if line[0:3] == "MSG":
                ms = line.find(" ")
                t = int(line[4:ms])
                trial["events"]["msg"].append([t, line[ms + 1 :]])
            elif line[0:6] == "EBLINK":
                st, et, dur = (int(v) for v in line[9:].split("\t")[:3])
                trial["events"]["Eblk"].append([st, et, dur])
            elif (
                line[0:4] in ("SFIX", "EFIX")
                or line[0:5] in ("SSACC", "ESACC")
                or line[0:6] == "SBLINK"
            ):
                pass  # fixation/saccade events not needed downstream
            else:
                parts = line.split("\t")
                try:
                    int(parts[0])
                except (ValueError, IndexError):
                    continue  # not a sample line
                x, y = float(parts[1]), float(parts[2])
                if float(parts[3]) == 0.0:  # pupil size 0.0 == missing
                    x, y = missing, missing
                trial["x"].append(x)
                trial["y"].append(y)
                trial["trackertime"].append(int(parts[0]))

    return data, trial_nr


def read_edf_time_overlap(filename, start, stop, missing=0.0):
    """
    Like `read_edf`, but for files where consecutive trials overlap
    (trial N+1 starts before trial N's stop line appears). Start/stop
    marker pairs are located up front by matching trial number, then
    each trial's data is parsed from the line range between them.

    Parameters
    ----------
    filename : str
        Path to the .asc file.
    start : str
        Substring identifying a trial-start line. The line is expected
        to end with ": <trial_nr>".
    stop : str
        Substring identifying a trial-end line, in the same format.
    missing : float, default=0.0
        Value substituted for missing gaze samples.

    Returns
    -------
    data : list of dict
        One trial dict (see module docstring) per matched start/stop pair.
    trial_nr : list
        Trial number per entry in `data`, taken from the start marker.
    """
    if not os.path.isfile(filename):
        raise FileNotFoundError(f"read_edf_time_overlap: file '{filename}' does not exist")

    with open(filename, "r") as f:
        raw = f.readlines()

    def _marker_info(line, idx):
        return {
            "l_idx": idx,
            "timing": int(line[4 : line.find(" ")]),
            "nr": int(line[line.find(":") + 2 : -1]),
        }

    starts = [_marker_info(line, idx) for idx, line in enumerate(raw) if start in line]
    stops = [_marker_info(line, idx) for idx, line in enumerate(raw) if stop in line]

    data = []
    trial_nr = []

    for s, e in zip(starts, stops):
        if s["nr"] != e["nr"]:
            warnings.warn(f"start_trial_{s['nr']} and stop_trial do not match", UserWarning)
            continue

        trial = _new_trial()
        starttime = s["timing"]

        for line in raw[s["l_idx"] : e["l_idx"]]:
            if line[0:3] == "MSG":
                ms = line.find(" ")
                t = int(line[4:ms])
                msg = line[ms + 1 :]
                if start in line and t != starttime:
                    msg = "next trl already starts"
                trial["events"]["msg"].append([t, msg])
            elif line[0:6] == "EBLINK":
                st, et, dur = (int(v) for v in line[9:].split("\t")[:3])
                trial["events"]["Eblk"].append([st, et, dur])
            elif (
                line[0:4] in ("SFIX", "EFIX")
                or line[0:5] in ("SSACC", "ESACC")
                or line[0:6] == "SBLINK"
            ):
                pass
            else:
                parts = line.split("\t")
                try:
                    int(parts[0])
                except (ValueError, IndexError):
                    continue
                x, y = float(parts[1]), float(parts[2])
                if float(parts[3]) == 0.0:
                    x, y = missing, missing
                trial["x"].append(x)
                trial["y"].append(y)
                trial["trackertime"].append(int(parts[0]))

        data.append(_finalize_trial(trial))
        trial_nr.append(s["nr"])

    return data, trial_nr


def _detect_blinks(x, y, time, missing=0.0, minlen=10):
    """
    Detect blinks as runs of at least `minlen` consecutive samples
    where both x and y equal `missing`.

    Returns
    -------
    Eblk : list of [start, end, duration]
    """
    is_missing = (x == missing) & (y == missing)
    diff = np.diff(is_missing.astype(int))
    starts = np.where(diff == 1)[0] + 1
    ends = np.where(diff == -1)[0] + 1

    Eblk = []
    for i, s in enumerate(starts):
        if i < len(ends):
            e = ends[i]
        elif len(ends) > 0:
            e = ends[-1]
        else:
            e = -1
        if e - s >= minlen:
            Eblk.append([time[s], time[e], time[e] - time[s]])
    return Eblk


def read_eyetribe(filename, start, stop=None, missing=0.0):
    """
    Parse an EyeTribe tab-separated (.tsv) file into per-trial dicts.

    Trials are segmented on MSG lines containing `start` (or `stop`,
    if given). Blinks are not marked explicitly in EyeTribe data, so
    they're detected from the gaze samples themselves: any run of at
    least 10 consecutive samples with x == y == `missing` is treated
    as a blink.

    Parameters
    ----------
    filename : str
        Path to the .tsv file.
    start : str
        Substring identifying a trial-start MSG line.
    stop : str, optional
        Substring identifying a trial-end MSG line. If None, a trial
        ends as soon as the next trial-start MSG line appears.
    missing : float, default=0.0
        Value used by the tracker to mark missing gaze samples.

    Returns
    -------
    data : list of dict
        One trial dict (see module docstring) per trial.

    Notes
    -----
    Deliberate behavior change from the PyGazeAnalyser function this
    replaces: with stop=None, that function checked for a new trial
    start via `start in line` where `line` is the tab-split field
    list -- a list-membership test that only matches if some field is
    the *exact* string `start`, not if `start` appears as a substring
    (e.g. "start_trial: 2" would never match the substring
    "start_trial"). In practice this meant every trial in a stop=None
    file was silently merged into one, since only the very first
    start marker (checked before any trial had "started") was ever
    detected. Here, the same substring check used for detecting the
    first trial start is reused consistently, so multi-trial
    stop=None files are split correctly.
    """
    if not os.path.isfile(filename):
        raise FileNotFoundError(f"read_eyetribe: file '{filename}' does not exist")

    with open(filename, "r") as f:
        raw = f.readlines()

    data = []
    trial = _new_trial()
    started = False

    for i, raw_line in enumerate(raw):
        line = raw_line.replace("\n", "").replace("\r", "").split("\t")
        is_last = i == len(raw) - 1

        trial_ended = False
        if started:
            if stop is not None:
                if (line[0] == "MSG" and stop in line[3]) or is_last:
                    started = False
                    trial_ended = True
            else:
                if (line[0] == "MSG" and start in line[3]) or is_last:
                    started = True
                    trial_ended = True

            if trial_ended:
                trial = _finalize_trial(trial)
                trial["events"]["Eblk"] = _detect_blinks(
                    trial["x"], trial["y"], trial["trackertime"], missing=missing
                )
                data.append(trial)
                trial = _new_trial()
        else:
            if line[0] == "MSG" and start in line[3]:
                started = True

        if started:
            if line[0] == "MSG":
                trial["events"]["msg"].append([int(line[2]), line[3]])
            else:
                try:
                    trial["x"].append(float(line[6]))
                    trial["y"].append(float(line[7]))
                    trial["trackertime"].append(int(line[1]))
                except (ValueError, IndexError):
                    continue

    return data
