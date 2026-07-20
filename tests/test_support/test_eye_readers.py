"""
Test suite for open_dvm.support.eye_readers.

Clean-room replacement for the third-party (GPL-licensed, unmaintained)
PyGazeAnalyser readers this toolbox used to vendor. Only the trial-dict
fields this toolbox actually consumes are covered: x, y, trackertime,
events['msg'], events['Eblk'].

Organization
------------
- TestReadEdf: EyeLink EDF-derived ASCII (.asc), stop=None segmentation
- TestReadEdfTimeOverlap: explicit start/stop markers, overlapping trials
- TestReadEyetribe: EyeTribe tab-separated (.tsv) files, blink detection
- TestRegressions: the intentional behavior fix vs. the old library
"""

import warnings

import numpy as np
import pytest

from open_dvm.support.eye_readers import (
    _detect_blinks,
    read_edf,
    read_edf_time_overlap,
    read_eyetribe,
)
from tests.fixtures.eye_readers_sample_data import (
    write_asc_mismatched_trial,
    write_asc_overlapping_trials,
    write_asc_two_trials,
    write_tsv_two_trials,
    write_tsv_with_stop,
)

# ============================================================================
# read_edf
# ============================================================================


class TestReadEdf:
    @pytest.mark.unit
    def test_splits_into_two_trials(self, tmp_path):
        path = tmp_path / "sample.asc"
        write_asc_two_trials(path)

        data, trial_nr = read_edf(str(path), start="start_trial", trial_info="TRIALID", missing=0.0)

        assert len(data) == 2
        assert trial_nr == [5, 6]

    @pytest.mark.unit
    def test_gaze_samples_and_missing_sample_zeroed(self, tmp_path):
        path = tmp_path / "sample.asc"
        write_asc_two_trials(path)

        data, _ = read_edf(str(path), start="start_trial", missing=0.0)

        # trial 1: sample 1001 real, sample 1002 has pupil=0.0 -> zeroed,
        # sample 1011 real
        np.testing.assert_array_equal(data[0]["x"], [500.0, 0.0, 505.0])
        np.testing.assert_array_equal(data[0]["y"], [400.0, 0.0, 402.0])
        np.testing.assert_array_equal(data[0]["trackertime"], [1001, 1002, 1011])

    @pytest.mark.unit
    def test_custom_missing_value_used_for_zeroing(self, tmp_path):
        path = tmp_path / "sample.asc"
        write_asc_two_trials(path)

        data, _ = read_edf(str(path), start="start_trial", missing=-999.0)

        assert data[0]["x"][1] == -999.0
        assert data[0]["y"][1] == -999.0

    @pytest.mark.unit
    def test_eblink_events_captured(self, tmp_path):
        path = tmp_path / "sample.asc"
        write_asc_two_trials(path)

        data, _ = read_edf(str(path), start="start_trial", missing=0.0)

        assert data[0]["events"]["Eblk"] == [[1003, 1010, 7]]
        assert data[1]["events"]["Eblk"] == [[2003, 2015, 12]]

    @pytest.mark.unit
    def test_msg_events_captured_including_trial_start_line(self, tmp_path):
        path = tmp_path / "sample.asc"
        write_asc_two_trials(path)

        data, _ = read_edf(str(path), start="start_trial", missing=0.0)

        assert data[0]["events"]["msg"] == [
            [1000, "start_trial: 1\n"],
            [1012, "TRIALID 5\n"],
        ]

    @pytest.mark.unit
    def test_fixation_saccade_lines_ignored(self, tmp_path):
        # regression: SFIX/EFIX/SSACC/ESACC/SBLINK lines must not be
        # mistaken for samples or crash the parser
        path = tmp_path / "sample.asc"
        write_asc_two_trials(path)

        data, _ = read_edf(str(path), start="start_trial", missing=0.0)

        assert "Sfix" not in data[0] and "Efix" not in data[0]
        # trial 1 has exactly 3 real samples despite the 5 event summary
        # lines interspersed after them
        assert len(data[0]["x"]) == 3

    @pytest.mark.unit
    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            read_edf(str(tmp_path / "nope.asc"), start="start_trial")

    @pytest.mark.unit
    def test_no_missing_sample_variant(self, tmp_path):
        path = tmp_path / "sample.asc"
        write_asc_two_trials(path, missing_sample=False)

        data, _ = read_edf(str(path), start="start_trial", missing=0.0)

        np.testing.assert_array_equal(data[0]["x"], [500.0, 505.0])

    @pytest.mark.unit
    def test_explicit_stop_marker(self, tmp_path):
        path = tmp_path / "sample.asc"
        lines = [
            "MSG\t1000 start_trial: 1\n",
            "1001\t500.0\t400.0\t5000.0\t...\n",
            "MSG\t1010 stop_trial: 1\n",
            "1011\t999.0\t999.0\t5000.0\t...\n",  # after stop, should be excluded
            "MSG\t2000 start_trial: 2\n",
            "2001\t600.0\t300.0\t6000.0\t...\n",
        ]
        path.write_text("".join(lines))

        data, _ = read_edf(str(path), start="start_trial", stop="stop_trial", missing=0.0)

        assert len(data) == 1
        np.testing.assert_array_equal(data[0]["x"], [500.0])

    @pytest.mark.unit
    def test_unparseable_line_skipped(self, tmp_path):
        path = tmp_path / "sample.asc"
        # note: the *very last* line of a stop=None file is treated as
        # both "end this trial" and "start of a new one" (matching the
        # original library), so end on a message line, not a sample,
        # to keep this test only about the unparseable-line skip
        lines = [
            "MSG\t1000 start_trial: 1\n",
            "1001\t500.0\t400.0\t5000.0\t...\n",
            "not a sample line at all\n",
            "1002\t501.0\t401.0\t5010.0\t...\n",
            "MSG\t1003 trial done\n",
        ]
        path.write_text("".join(lines))

        data, _ = read_edf(str(path), start="start_trial", missing=0.0)

        np.testing.assert_array_equal(data[0]["x"], [500.0, 501.0])

    @pytest.mark.unit
    def test_trial_info_none_does_not_crash(self, tmp_path):
        # regression: the old PyGazeAnalyser read_edf did
        # `if trial_info in event[1]:` unconditionally, which raises
        # TypeError as soon as trial_info is None (its own documented
        # default) and any message exists -- i.e. on essentially every
        # real trial, since the trial-start line itself is a message
        path = tmp_path / "sample.asc"
        lines = [
            "MSG\t1000 start_trial: 1\n",
            "1001\t500.0\t400.0\t5000.0\t...\n",
        ]
        path.write_text("".join(lines))

        data, trial_nr = read_edf(str(path), start="start_trial", missing=0.0)

        assert len(data) == 1
        assert np.isnan(trial_nr[0])


# ============================================================================
# read_edf_time_overlap
# ============================================================================


class TestReadEdfTimeOverlap:
    @pytest.mark.unit
    def test_overlapping_trials_split_correctly(self, tmp_path):
        path = tmp_path / "overlap.asc"
        write_asc_overlapping_trials(path)

        data, trial_nr = read_edf_time_overlap(
            str(path), start="start_trial", stop="stop_trial", missing=0.0
        )

        assert trial_nr == [1, 2]
        # trial 1's slice includes the overlapping sample from trial 2
        np.testing.assert_array_equal(data[0]["x"], [700.0, 701.0])
        np.testing.assert_array_equal(data[1]["x"], [701.0, 702.0])

    @pytest.mark.unit
    def test_repeated_start_marker_message_relabeled(self, tmp_path):
        # regression-relevant: a start_trial line seen mid-slice for a
        # DIFFERENT trial gets its message text replaced
        path = tmp_path / "overlap.asc"
        write_asc_overlapping_trials(path)

        data, _ = read_edf_time_overlap(
            str(path), start="start_trial", stop="stop_trial", missing=0.0
        )

        assert data[0]["events"]["msg"] == [
            [3000, "start_trial: 1\n"],
            [3005, "next trl already starts"],
        ]

    @pytest.mark.unit
    def test_eblink_captured_within_overlap_window(self, tmp_path):
        path = tmp_path / "overlap.asc"
        write_asc_overlapping_trials(path)

        data, _ = read_edf_time_overlap(
            str(path), start="start_trial", stop="stop_trial", missing=0.0
        )

        assert data[0]["events"]["Eblk"] == [[3007, 3009, 2]]

    @pytest.mark.unit
    def test_missing_sample_zeroed(self, tmp_path):
        path = tmp_path / "overlap.asc"
        lines = [
            "MSG\t1000 start_trial: 1\n",
            "1001\t500.0\t400.0\t5000.0\t...\n",
            "1002\t501.0\t401.0\t0.0\t...\n",  # missing (pupil=0)
            "MSG\t1010 stop_trial: 1\n",
        ]
        path.write_text("".join(lines))

        data, _ = read_edf_time_overlap(
            str(path), start="start_trial", stop="stop_trial", missing=-1.0
        )

        np.testing.assert_array_equal(data[0]["x"], [500.0, -1.0])
        np.testing.assert_array_equal(data[0]["y"], [400.0, -1.0])

    @pytest.mark.unit
    def test_mismatched_trial_numbers_warns_and_skips(self, tmp_path):
        path = tmp_path / "mismatch.asc"
        write_asc_mismatched_trial(path)

        with pytest.warns(UserWarning, match="do not match"):
            data, trial_nr = read_edf_time_overlap(
                str(path), start="start_trial", stop="stop_trial"
            )

        assert data == []
        assert trial_nr == []

    @pytest.mark.unit
    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            read_edf_time_overlap(
                str(tmp_path / "nope.asc"), start="start_trial", stop="stop_trial"
            )

    @pytest.mark.unit
    def test_unparseable_and_fixation_lines_skipped(self, tmp_path):
        path = tmp_path / "overlap.asc"
        lines = [
            "MSG\t1000 start_trial: 1\n",
            "1001\t500.0\t400.0\t5000.0\t...\n",
            "not a sample line\n",
            "SFIX R   1002\n",
            "1003\t501.0\t401.0\t5010.0\t...\n",
            "MSG\t1010 stop_trial: 1\n",
        ]
        path.write_text("".join(lines))

        data, trial_nr = read_edf_time_overlap(
            str(path), start="start_trial", stop="stop_trial", missing=0.0
        )

        assert trial_nr == [1]
        np.testing.assert_array_equal(data[0]["x"], [500.0, 501.0])


# ============================================================================
# read_eyetribe
# ============================================================================


class TestReadEyetribe:
    @pytest.mark.unit
    def test_splits_into_two_trials(self, tmp_path):
        path = tmp_path / "sample.tsv"
        write_tsv_two_trials(path)

        data = read_eyetribe(str(path), start="start_trial", missing=0.0)

        assert len(data) == 2

    @pytest.mark.unit
    def test_gaze_samples_captured(self, tmp_path):
        path = tmp_path / "sample.tsv"
        write_tsv_two_trials(path)

        data = read_eyetribe(str(path), start="start_trial", missing=0.0)

        # trial 1: 5 real + 12 missing + 5 real = 22 samples
        assert len(data[0]["x"]) == 22
        np.testing.assert_array_equal(data[0]["x"][:5], [500.0, 501.0, 502.0, 503.0, 504.0])

    @pytest.mark.unit
    def test_blink_detected_above_threshold(self, tmp_path):
        path = tmp_path / "sample.tsv"
        write_tsv_two_trials(path)

        data = read_eyetribe(str(path), start="start_trial", missing=0.0)

        # trial 1 has a 12-sample missing run (>= minlen=10) starting
        # at trackertime 1010 and ending at 1021 (12 samples -> 11 steps)
        assert len(data[0]["events"]["Eblk"]) == 1
        start, end, dur = data[0]["events"]["Eblk"][0]
        assert start == 1010
        assert dur == end - start

    @pytest.mark.unit
    def test_no_blink_below_threshold(self, tmp_path):
        path = tmp_path / "sample.tsv"
        write_tsv_two_trials(path)

        data = read_eyetribe(str(path), start="start_trial", missing=0.0)

        # trial 2 has only a 3-sample missing run (< minlen=10)
        assert data[1]["events"]["Eblk"] == []

    @pytest.mark.unit
    def test_explicit_stop_marker(self, tmp_path):
        path = tmp_path / "sample.tsv"
        write_tsv_with_stop(path)

        data = read_eyetribe(str(path), start="start_trial", stop="stop_trial", missing=0.0)

        assert len(data) == 2
        np.testing.assert_array_equal(data[0]["x"], [500.0, 501.0, 502.0])
        np.testing.assert_array_equal(data[1]["x"], [600.0, 601.0, 602.0])

    @pytest.mark.unit
    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            read_eyetribe(str(tmp_path / "nope.tsv"), start="start_trial")

    @pytest.mark.unit
    def test_unparseable_line_skipped(self, tmp_path):
        path = tmp_path / "sample.tsv"
        lines = [
            "MSG\t2024-01-01 00:00:00.000\t1000\tstart_trial: 1\n",
            "2024-01-01 00:00:00.000\t1001\tFalse\t7\t500.0\t400.0\t500.0\t400.0\t16.0\n",
            "too\tfew\tcolumns\n",
            "2024-01-01 00:00:00.000\t1002\tFalse\t7\t501.0\t401.0\t501.0\t401.0\t16.0\n",
            "MSG\t2024-01-01 00:00:00.000\t1003\ttrial done\n",
        ]
        path.write_text("".join(lines))

        data = read_eyetribe(str(path), start="start_trial", missing=0.0)

        np.testing.assert_array_equal(data[0]["x"], [500.0, 501.0])


# ============================================================================
# _detect_blinks
# ============================================================================


class TestDetectBlinks:
    @pytest.mark.unit
    def test_unrecovered_blink_at_end_not_counted(self):
        # a second blink that never recovers before the array ends has
        # no matching "end" transition; the fallback (reusing the last
        # real end) must not fabricate a spurious short blink for it
        x = np.array([1] * 3 + [0] * 12 + [1] * 3 + [0] * 12, dtype=float)
        y = x.copy()
        t = np.arange(len(x)) + 100

        result = _detect_blinks(x, y, t, missing=0.0, minlen=10)

        assert result == [[103, 115, 12]]

    @pytest.mark.unit
    def test_no_recovery_at_all_returns_empty(self):
        x = np.array([1] * 3 + [0] * 12, dtype=float)
        y = x.copy()
        t = np.arange(len(x)) + 200

        result = _detect_blinks(x, y, t, missing=0.0, minlen=10)

        assert result == []


# ============================================================================
# Regressions (intentional behavior fix vs. the old vendored library)
# ============================================================================


class TestRegressions:
    @pytest.mark.unit
    def test_multitrial_stop_none_file_not_merged(self, tmp_path):
        # The old PyGazeAnalyser read_eyetribe checked `start in line`
        # (list membership against tab-split fields) instead of a
        # substring match against the message field, so with
        # stop=None every trial after the first silently got merged
        # into one. This asserts the fix: each trial-start marker
        # correctly ends the previous trial.
        path = tmp_path / "sample.tsv"
        write_tsv_two_trials(path)

        data = read_eyetribe(str(path), start="start_trial", missing=0.0)

        assert len(data) == 2
        assert data[0]["x"][-1] != data[1]["x"][0]  # genuinely distinct trials


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
