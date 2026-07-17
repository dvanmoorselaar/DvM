"""
Test suite for open_dvm.analysis.EYE (EYE and SaccadeDetector classes).

Organization
------------
- TestEyeInit: EYE.__init__
- TestGetEyeData: EYE.get_eye_data (file discovery, session sync, edge cases)
- TestInterpTrial: EYE.interp_trial
- TestGetXy: EYE.get_xy
- TestSetXy: EYE.set_xy
- TestAnglesFromXyAndBins: EYE.angles_from_xy, EYE.create_angle_bins
- TestLinkEyeToEeg: EYE.link_eye_to_eeg
- TestCalculateAngleAndDegreesPerPixel: EYE.calculate_angle, EYE.degrees_per_pixel
- TestSaccadeDetectorInit: SaccadeDetector.__init__
- TestCalcVelocity: SaccadeDetector.calc_velocity
- TestNoiseDetect: SaccadeDetector.noise_detect
- TestEstimateThresh: SaccadeDetector.estimate_thresh
- TestSaccadeDetection: SaccadeDetector.saccade_detection
- TestDetectEvents: SaccadeDetector.detect_events
- TestRegressions: regression tests for bugs found & fixed during review
"""

import importlib
import math
import os
from math import degrees, atan2

import numpy as np
import pandas as pd
import pytest

from open_dvm.analysis.EYE import EYE, SaccadeDetector

from tests.fixtures.eye_sample_data import (
    make_trial,
    write_asc_session,
    write_beh_csv,
    fixation_saccade_profile,
)


# ============================================================================
# EYE.__init__
# ============================================================================

class TestEyeInit:
    @pytest.mark.unit
    def test_default_attributes(self):
        eye_obj = EYE()
        assert eye_obj.view_dist == 60
        assert eye_obj.scr_res == (1920, 1080)
        assert eye_obj.scr_h == 30
        assert eye_obj.sfreq == 500

    @pytest.mark.unit
    def test_custom_attributes(self):
        eye_obj = EYE(viewing_dist=90, screen_res=(3840, 2160), screen_h=35,
                       sfreq=1000)
        assert eye_obj.view_dist == 90
        assert eye_obj.scr_res == (3840, 2160)
        assert eye_obj.scr_h == 35
        assert eye_obj.sfreq == 1000


# ============================================================================
# EYE.get_eye_data
# ============================================================================

class TestGetEyeData:
    @pytest.mark.unit
    def test_explicit_files_two_trials(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        write_asc_session('sample.asc', n_trials=2)
        write_beh_csv('sample.csv', n_trials=2)

        eye_obj = EYE(sfreq=1000)
        eye, df, trial_info = eye_obj.get_eye_data(
            sj='', eye_files=['sample.asc'], beh_files=['sample.csv'],
            start='start trial', stop=None,
        )
        assert len(eye) == 2
        assert df.shape[0] == 2
        # no trigger-msg trial_info requested -> filthy hack kicks in:
        # sequential 0..n-1
        np.testing.assert_array_equal(trial_info, [0, 1])

    @pytest.mark.unit
    def test_glob_discovery_via_all(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        os.makedirs(tmp_path / 'eye' / 'raw', exist_ok=True)
        os.makedirs(tmp_path / 'behavioral' / 'raw', exist_ok=True)
        write_asc_session(tmp_path / 'eye' / 'raw' / 'sub_01_ses_1.asc',
                           n_trials=2)
        write_beh_csv(tmp_path / 'behavioral' / 'raw' / 'sub_01_ses_1.csv',
                       n_trials=2)

        eye_obj = EYE(sfreq=1000)
        eye, df, trial_info = eye_obj.get_eye_data(
            sj=1, eye_files='all', beh_files='all',
            start='start trial', stop=None,
        )
        assert len(eye) == 2
        assert df.shape[0] == 2

    @pytest.mark.unit
    def test_multi_session_glob_missing_eye_file_drops_matching_beh(
        self, tmp_path, monkeypatch
    ):
        """
        Regression test: beh_files filtering used to mutate the list
        while iterating over it (`beh_files.pop(i)`), which silently
        dropped/kept the WRONG file whenever more than one session was
        missing eye data. With sessions 1,3,5 present in eye but
        1,2,3,4,5 present in beh, sessions 2 and 4 should be dropped
        and 1,3,5 kept -- not some other combination.
        """
        monkeypatch.chdir(tmp_path)
        os.makedirs(tmp_path / 'eye' / 'raw', exist_ok=True)
        os.makedirs(tmp_path / 'behavioral' / 'raw', exist_ok=True)

        for ses in (1, 3, 5):
            write_asc_session(
                tmp_path / 'eye' / 'raw' / f'sub_01_ses_{ses}.asc', n_trials=1
            )
        for ses in (1, 2, 3, 4, 5):
            write_beh_csv(
                tmp_path / 'behavioral' / 'raw' / f'sub_01_ses_{ses}.csv',
                n_trials=1,
            )

        eye_obj = EYE(sfreq=1000)
        eye, df, trial_info = eye_obj.get_eye_data(
            sj=1, eye_files='all', beh_files='all',
            start='start trial', stop=None,
        )
        # 3 eye sessions (1 trial each) -> 3 trials total, matched by 3
        # surviving beh files (sessions 1, 3, 5)
        assert len(eye) == 3
        assert df.shape[0] == 3

    @pytest.mark.unit
    def test_tsv_files_dispatch_to_eyetribe_reader(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        lines = [
            "MSG\t2024-01-01 00:00:00.000\t1000\tstart trial: 1\n",
            "2024-01-01 00:00:00.000\t1001\tFalse\t7\t500.0\t400.0\t500.0\t400.0\t16.0\n",
            "MSG\t2024-01-01 00:00:00.000\t2000\tstart trial: 2\n",
            "2024-01-01 00:00:00.000\t2001\tFalse\t7\t600.0\t300.0\t600.0\t300.0\t16.0\n",
        ]
        with open('sample.tsv', 'w') as f:
            f.writelines(lines)
        write_beh_csv('sample.csv', n_trials=2)

        eye_obj = EYE(sfreq=1000)
        eye, df, trial_info = eye_obj.get_eye_data(
            sj='', eye_files=['sample.tsv'], beh_files=['sample.csv'],
            start='start trial', stop=None,
        )
        assert len(eye) == 2

    @pytest.mark.unit
    def test_stop_marker_uses_time_overlap_reader(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        lines = [
            "MSG\t1000 start trial: 1\n",
            "1001\t500.0\t400.0\t5000.0\t...\n",
            "MSG\t1005 start trial: 2\n",
            "1006\t600.0\t300.0\t6000.0\t...\n",
            "MSG\t1010 stop trial: 1\n",
            "1011\t501.0\t401.0\t5000.0\t...\n",
            "MSG\t1015 stop trial: 2\n",
        ]
        with open('sample.asc', 'w') as f:
            f.writelines(lines)
        write_beh_csv('sample.csv', n_trials=2)

        eye_obj = EYE(sfreq=1000)
        eye, df, trial_info = eye_obj.get_eye_data(
            sj='', eye_files=['sample.asc'], beh_files=['sample.csv'],
            start='start trial', stop='stop trial',
        )
        assert len(eye) == 2

    @pytest.mark.unit
    def test_more_beh_than_eye_trials_removes_extra_beh_rows(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.chdir(tmp_path)
        write_asc_session('sample.asc', n_trials=2)
        write_beh_csv('sample.csv', n_trials=5)

        eye_obj = EYE(sfreq=1000)
        eye, df, trial_info = eye_obj.get_eye_data(
            sj='', eye_files=['sample.asc'], beh_files=['sample.csv'],
            start='start trial', stop=None,
        )
        assert len(eye) == 2
        assert df.shape[0] == 2

    @pytest.mark.unit
    def test_more_eye_than_beh_trials_removes_extra_eye_trials(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.chdir(tmp_path)
        write_asc_session('sample.asc', n_trials=5)
        write_beh_csv('sample.csv', n_trials=2)

        eye_obj = EYE(sfreq=1000)
        eye, df, trial_info = eye_obj.get_eye_data(
            sj='', eye_files=['sample.asc'], beh_files=['sample.csv'],
            start='start trial', stop=None,
        )
        assert len(eye) == 2
        assert df.shape[0] == 2

    @pytest.mark.unit
    def test_practice_trials_removed(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        write_asc_session('sample.asc', n_trials=2)
        write_beh_csv('sample.csv', n_trials=2,
                       extra_cols={'practice': ['yes', 'no']})

        eye_obj = EYE(sfreq=1000)
        eye, df, trial_info = eye_obj.get_eye_data(
            sj='', eye_files=['sample.asc'], beh_files=['sample.csv'],
            start='start trial', stop=None,
        )
        assert len(eye) == 1
        assert df.shape[0] == 1
        assert (df['practice'] == 'no').all()

    @pytest.mark.unit
    def test_tsv_files_do_not_crash_on_missing_trial_info(
        self, tmp_path, monkeypatch
    ):
        """
        Regression test: get_eye_data never populated trial_info for
        .tsv input (read_eyetribe returns only trial data, no trial
        numbers), leaving it at the function's default of None and
        crashing on `trial_info[0]`. It should now fall back to NaN
        per trial, matching how .asc files behave when no trial_info
        marker is requested (sequential fallback).
        """
        monkeypatch.chdir(tmp_path)
        lines = [
            "MSG\t2024-01-01 00:00:00.000\t1000\tstart trial: 1\n",
            "2024-01-01 00:00:00.000\t1001\tFalse\t7\t500.0\t400.0\t500.0\t400.0\t16.0\n",
            "MSG\t2024-01-01 00:00:00.000\t2000\tstart trial: 2\n",
            "2024-01-01 00:00:00.000\t2001\tFalse\t7\t600.0\t300.0\t600.0\t300.0\t16.0\n",
        ]
        with open('sample.tsv', 'w') as f:
            f.writelines(lines)
        write_beh_csv('sample.csv', n_trials=2)

        eye_obj = EYE(sfreq=1000)
        eye, df, trial_info = eye_obj.get_eye_data(
            sj='', eye_files=['sample.tsv'], beh_files=['sample.csv'],
            start='start trial', stop=None,
        )
        assert len(eye) == 2
        np.testing.assert_array_equal(trial_info, [0, 1])

    @pytest.mark.unit
    def test_multi_file_trial_info_correctly_flattened(
        self, tmp_path, monkeypatch
    ):
        """
        Regression test: with multiple eye files and an explicit
        trial_info marker requested, trial_info used to stay as a list
        of per-file arrays (np.hstack's result was discarded rather
        than assigned back), producing a ragged/2D trial_info instead
        of a flat array matching eye's row order -- or crashing
        outright when files had unequal trial counts.
        """
        monkeypatch.chdir(tmp_path)

        def write_file(path, trial_nrs, rec_start=1000):
            lines = []
            t = rec_start
            for tr in trial_nrs:
                lines.append(f"MSG\t{t} start trial: {tr}\n")
                lines.append(f"MSG\t{t + 5} TRIALID {tr}\n")
                lines.append(f"{t + 6}\t500.0\t400.0\t5000.0\t...\n")
                t += 20
            with open(path, 'w') as f:
                f.writelines(lines)

        # unequal trial counts across files: 2 then 3
        write_file('a.asc', [1, 2])
        write_file('b.asc', [3, 4, 5])
        write_beh_csv('beh.csv', n_trials=5)

        eye_obj = EYE(sfreq=1000)
        eye, df, trial_info = eye_obj.get_eye_data(
            sj='', eye_files=['a.asc', 'b.asc'], beh_files=['beh.csv'],
            start='start trial', trial_info='TRIALID', stop=None,
        )
        assert eye.shape == (5,)
        assert trial_info.shape == (5,)
        np.testing.assert_array_equal(trial_info, [1, 2, 3, 4, 5])

    @pytest.mark.unit
    def test_partial_nan_trial_info_interpolated_from_neighbors(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.chdir(tmp_path)
        lines = [
            "MSG\t1000 start trial: 1\n",
            "MSG\t1005 TRIALID 10\n",
            "1006\t500.0\t400.0\t5000.0\t...\n",
            "MSG\t2000 start trial: 2\n",
            "2001\t500.0\t400.0\t5000.0\t...\n",
            "MSG\t3000 start trial: 3\n",
            "MSG\t3005 TRIALID 30\n",
            "3006\t500.0\t400.0\t5000.0\t...\n",
        ]
        with open('sample.asc', 'w') as f:
            f.writelines(lines)
        write_beh_csv('sample.csv', n_trials=3)

        eye_obj = EYE(sfreq=1000)
        eye, df, trial_info = eye_obj.get_eye_data(
            sj='', eye_files=['sample.asc'], beh_files=['sample.csv'],
            start='start trial', trial_info='TRIALID', stop=None,
        )
        # middle trial has no TRIALID marker -> interpolated as the mean
        # of its two neighbors (10 and 30 -> 20)
        np.testing.assert_array_equal(trial_info, [10, 20, 30])

    @pytest.mark.unit
    def test_eye_trials_mismatch_uses_nr_trials_mask(
        self, tmp_path, monkeypatch
    ):
        """
        Exercises the rarely-hit branch where the number of 'start'
        marker messages found across trials doesn't equal the number
        of trials (len(eye_trials) != eye.shape[0]) -- structurally
        difficult to trigger via a real .asc file (any line matching
        the start marker also ends the current trial in read_edf), so
        the reader is mocked to return an already-parsed trial list
        with a trial whose msg list contains two matching messages.
        """
        monkeypatch.chdir(tmp_path)
        with open('sample.asc', 'w') as f:
            f.write("MSG\t1000 start trial: 1\n1001\t1\t1\t1\t\n")
        write_beh_csv('sample.csv', n_trials=1,
                      extra_cols={'nr_trials': [1, 2, 99]})

        def fake_read_edf(file, start, trial_info=None, missing=0):
            trial0 = make_trial(
                x=[500.0], y=[400.0], trackertime=[1000],
                msgs=[[1000, 'start trial: 1'],
                      [1002, 'start trial: 1 (dup)']],
            )
            trial1 = make_trial(
                x=[500.0], y=[400.0], trackertime=[2000],
                msgs=[[2000, 'start trial: 2']],
            )
            return [trial0, trial1], [np.nan, np.nan]

        eye_module = importlib.import_module('open_dvm.analysis.EYE')
        eye_obj = EYE(sfreq=1000)
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(eye_module, 'read_edf', fake_read_edf)
            eye, df, trial_info = eye_obj.get_eye_data(
                sj='', eye_files=['sample.asc'], beh_files=['sample.csv'],
                start='start trial', stop=None,
            )
        # eye_trials = [2, 2, 3] (trial 0 matched twice, trial 1 once);
        # df['nr_trials']=[1,2,99] -> only value 2 is in eye_trials
        assert len(eye) == 2
        np.testing.assert_array_equal(df['nr_trials'].values, [2])

    @pytest.mark.unit
    def test_no_behavioral_files_returns_index_only_df(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.chdir(tmp_path)
        write_asc_session('sample.asc', n_trials=2)

        eye_obj = EYE(sfreq=1000)
        eye, df, trial_info = eye_obj.get_eye_data(
            sj='', eye_files=['sample.asc'], beh_files=[],
            start='start trial', stop=None,
        )
        assert len(eye) == 2
        assert df.shape[0] == 2
        assert list(df.index) == [0, 1]


# ============================================================================
# EYE.interp_trial
# ============================================================================

class TestInterpTrial:
    @pytest.mark.unit
    def test_no_blinks_returns_data_unchanged(self):
        eye_obj = EYE(sfreq=1000)
        trial = make_trial(
            x=[100, 101, 102, 103, 104],
            y=[200, 201, 202, 203, 204],
            trackertime=[0, 1, 2, 3, 4],
            blinks=[],
        )
        x_out, y_out = eye_obj.interp_trial(trial)
        np.testing.assert_array_equal(x_out, [100, 101, 102, 103, 104])
        np.testing.assert_array_equal(y_out, [200, 201, 202, 203, 204])

    @pytest.mark.unit
    def test_blink_linearly_interpolated(self):
        eye_obj = EYE(sfreq=1000)
        n = 400
        x = np.full(n, 100.0)
        y = np.full(n, 200.0)
        x[200:206] = 0.0
        y[200:206] = 0.0
        x[-1], y[-1] = 120.0, 220.0
        trial = make_trial(x=x, y=y, trackertime=np.arange(n),
                            blinks=[(200, 205)])

        x_out, y_out = eye_obj.interp_trial(trial)

        assert not np.isnan(x_out).any()
        # far from the blink, values are unaffected
        assert x_out[190] == 100.0
        assert x_out[280] == 100.0
        # inside the padded blink window, values are linearly
        # interpolated between the surrounding real samples (both are
        # 100.0 here, so the interpolated result is also 100.0)
        assert x_out[202] == 100.0

    @pytest.mark.unit
    def test_blink_near_trial_start_now_interpolated(self):
        """
        Regression test: a blink within `pad` samples of the trial
        start used to produce a negative slice start that Python
        reinterpreted as counting from the array's end, silently
        emptying the slice and leaving the blink un-interpolated.
        """
        eye_obj = EYE(sfreq=1000)  # pad = 75 samples
        n = 200
        x = np.full(n, 500.0)
        y = np.full(n, 400.0)
        x[5:11] = 0.0
        y[5:11] = 0.0
        trial = make_trial(x=x, y=y, trackertime=np.arange(n),
                            blinks=[(5, 10)])

        x_out, y_out = eye_obj.interp_trial(trial)

        assert not np.any(x_out == 0.0)
        np.testing.assert_allclose(x_out, 500.0)

    @pytest.mark.unit
    def test_fully_missing_trial_returns_zeros(self):
        eye_obj = EYE(sfreq=1000)
        n = 50
        trial = make_trial(
            x=np.full(n, np.nan), y=np.full(n, np.nan),
            trackertime=np.arange(n), blinks=[],
        )
        x_out, y_out = eye_obj.interp_trial(trial)
        assert np.all(x_out == 0.0)
        assert np.all(y_out == 0.0)

    @pytest.mark.unit
    def test_blink_padding_covers_whole_short_trial_returns_zeros(self):
        """
        Trial isn't fully NaN up front (passes the first all-NaN
        check), but the blink's padded region ends up covering the
        entire (short) trial, so the post-padding all-NaN branch
        kicks in and zeros are returned.
        """
        eye_obj = EYE(sfreq=1000)  # pad = 75 samples
        n = 100
        x = np.full(n, 500.0)
        y = np.full(n, 400.0)
        x[40:60] = 0.0
        y[40:60] = 0.0
        trial = make_trial(x=x, y=y, trackertime=np.arange(n),
                            blinks=[(40, 59)])

        x_out, y_out = eye_obj.interp_trial(trial)
        assert np.all(x_out == 0.0)
        assert np.all(y_out == 0.0)

    @pytest.mark.unit
    def test_partial_nan_still_interpolated_not_early_returned(self):
        """
        Regression test: interp_trial used to bail out and return the
        raw, still-NaN data whenever ANY sample was NaN (`.any()`
        instead of `.all()`), rather than only when the whole trial
        was missing.
        """
        eye_obj = EYE(sfreq=1000)
        n = 400
        x = np.full(n, 100.0)
        y = np.full(n, 200.0)
        x[200:206] = np.nan
        y[200:206] = np.nan
        trial = make_trial(x=x, y=y, trackertime=np.arange(n), blinks=[])

        x_out, y_out = eye_obj.interp_trial(trial)
        assert not np.isnan(x_out).any()
        assert x_out[202] == 100.0


# ============================================================================
# EYE.get_xy
# ============================================================================

class TestGetXy:
    @pytest.mark.unit
    def test_basic_extraction_aligned_to_event(self):
        eye_obj = EYE(sfreq=1000)
        n = 100
        trackertime = np.arange(1000, 1000 + n)
        x = 500.0 + np.arange(n) * 0.1
        y = 400.0 + np.arange(n) * 0.05
        trial = make_trial(
            x=x, y=y, trackertime=trackertime,
            msgs=[[1000, 'start trial: 1'], [1050, 'trigger']],
        )
        eye = np.array([trial])

        x_out, y_out, times = eye_obj.get_xy(eye, -20, 20, 'trigger',
                                              interpolate_blinks=False)
        # times grid: -20..19 ms at 1000Hz
        np.testing.assert_array_equal(times, np.arange(-20, 20))
        # grid point t maps exactly to raw sample at trackertime=1050+t
        for t in range(-20, 20):
            expected = 500.0 + (1050 + t - 1000) * 0.1
            j = np.where(times == t)[0][0]
            assert x_out[0, j] == pytest.approx(expected)

    @pytest.mark.unit
    def test_sparse_data_no_longer_crashes(self):
        """
        Regression test: get_xy used to copy raw samples into the
        uniform output grid by *position*
        (x[i,target_idx] = x_[idx][:len(target_idx)]), which crashed
        with a shape mismatch whenever the number of raw samples in
        the window differed from the number of grid points -- e.g.
        with a sparse/gappy trial. It now maps by timestamp via
        np.interp instead.
        """
        eye_obj = EYE(sfreq=1000)
        trial = make_trial(
            x=[500.0, 501.0], y=[400.0, 401.0],
            trackertime=[1001, 1006],
            msgs=[[1000, 'start trial: 1'], [1005, 'trigger']],
        )
        eye = np.array([trial])
        # window (-10, 10) at 1000Hz -> 20 grid points, only 2 raw samples
        x_out, y_out, times = eye_obj.get_xy(eye, -10, 10, 'trigger',
                                              interpolate_blinks=False)
        assert x_out.shape == (1, 20)

    @pytest.mark.unit
    def test_interpolate_blinks_requires_full_pre_window_coverage(self):
        """
        Regression test: the gate deciding whether to run blink
        interpolation was inverted (`tr_times[0] >= start` instead of
        `<= start`), so trials with LESS pre-event data coverage got
        interpolated while trials with MORE coverage did not. It
        should now require the recording to start at or before the
        window start.
        """
        eye_obj = EYE(sfreq=1000)
        n = 900
        trigger_t = 1250
        blink_s, blink_e = 1200, 1214

        def make(rec_start):
            trackertime = np.arange(rec_start, rec_start + n)
            x = np.full(n, 500.0)
            y = np.full(n, 400.0)
            blink_idx = slice(blink_s - rec_start, blink_e - rec_start + 1)
            x[blink_idx] = 0.0
            y[blink_idx] = 0.0
            return make_trial(
                x=x, y=y, trackertime=trackertime,
                msgs=[[rec_start, 'start trial: 1'], [trigger_t, 'trigger']],
                blinks=[(blink_s, blink_e)],
            )

        # ample coverage: recording starts 550ms before window start (-500)
        eye_full = np.array([make(rec_start=trigger_t - 550)])
        x_full, _, times = eye_obj.get_xy(eye_full, -500, 500, 'trigger')
        blink_rel = np.where((times >= blink_s - trigger_t) &
                              (times <= blink_e - trigger_t))[0]
        assert not np.any(x_full[0, blink_rel] == 0.0)

        # insufficient coverage: recording starts only 150ms before
        # window start (-500) -- not enough to reliably interpolate
        eye_partial = np.array([make(rec_start=trigger_t - 150)])
        x_partial, _, times2 = eye_obj.get_xy(eye_partial, -500, 500,
                                               'trigger')
        blink_rel2 = np.where((times2 >= blink_s - trigger_t) &
                               (times2 <= blink_e - trigger_t))[0]
        assert np.all(x_partial[0, blink_rel2] == 0.0)

    @pytest.mark.unit
    def test_trial_without_start_event_stays_zero(self):
        eye_obj = EYE(sfreq=1000)
        trial = make_trial(
            x=[500.0] * 10, y=[400.0] * 10, trackertime=np.arange(10),
            msgs=[[0, 'some other event']],
        )
        eye = np.array([trial])
        x_out, y_out, times = eye_obj.get_xy(eye, -5, 5, 'trigger')
        assert np.all(x_out == 0.0)
        assert np.all(y_out == 0.0)

    @pytest.mark.unit
    def test_empty_trial_stays_zero(self):
        eye_obj = EYE(sfreq=1000)
        trial = make_trial(
            x=[], y=[], trackertime=[],
            msgs=[[0, 'trigger']],
        )
        eye = np.array([trial])
        x_out, y_out, times = eye_obj.get_xy(eye, -5, 5, 'trigger')
        assert np.all(x_out == 0.0)


# ============================================================================
# EYE.set_xy
# ============================================================================

class TestSetXy:
    @pytest.mark.unit
    def test_globally_zero_edge_columns_excluded_from_processing(self):
        eye_obj = EYE(sfreq=1000, screen_res=(1000, 800), viewing_dist=60,
                       screen_h=30)
        n = 400
        times = np.arange(-200, 200)
        x = np.full((2, n), 550.0)
        y = np.full((2, n), 430.0)
        # no data at all (for either trial) in the first 10 samples --
        # e.g. a zero-padded edge from get_xy
        x[:, :10] = 0.0
        y[:, :10] = 0.0

        x_out, y_out = eye_obj.set_xy(x.copy(), y.copy(), times,
                                       drift_correct=None)
        assert np.all(x_out[:, :10] == 0.0)
        np.testing.assert_allclose(x_out[:, 10:], 550.0)

    @pytest.mark.unit
    def test_drift_correction_recenters_stable_fixation(self):
        eye_obj = EYE(sfreq=1000, screen_res=(1000, 800), viewing_dist=60,
                       screen_h=30)
        n = 400
        times = np.arange(-200, 200)
        x = np.full((1, n), 550.0)
        y = np.full((1, n), 430.0)

        x_out, y_out = eye_obj.set_xy(x.copy(), y.copy(), times,
                                       drift_correct=(-100, 0))
        np.testing.assert_allclose(x_out, 500.0)
        np.testing.assert_allclose(y_out, 400.0)

    @pytest.mark.unit
    def test_no_drift_correct_leaves_clean_data_unchanged(self):
        eye_obj = EYE(sfreq=1000, screen_res=(1000, 800), viewing_dist=60,
                       screen_h=30)
        n = 400
        times = np.arange(-200, 200)
        x = np.full((1, n), 550.0)
        y = np.full((1, n), 430.0)

        x_out, y_out = eye_obj.set_xy(x.copy(), y.copy(), times,
                                       drift_correct=None)
        np.testing.assert_allclose(x_out, 550.0)
        np.testing.assert_allclose(y_out, 430.0)

    @pytest.mark.unit
    def test_missing_data_in_drift_window_skips_that_trial_only(self):
        eye_obj = EYE(sfreq=1000, screen_res=(1000, 800), viewing_dist=60,
                       screen_h=30)
        n = 400
        times = np.arange(-200, 200)
        # trial 0: blink inside the drift-correct window; trial 1: clean
        x = np.full((2, n), 550.0)
        y = np.full((2, n), 430.0)
        x[0, 120:130] = 0.0
        y[0, 120:130] = 0.0

        x_out, y_out = eye_obj.set_xy(x.copy(), y.copy(), times,
                                       drift_correct=(-100, 0))
        # trial 0 stays uncorrected (still at its original 550 offset,
        # apart from the blink samples which noise_detect marks NaN)
        assert np.unique(x_out[0][~np.isnan(x_out[0])]).item() == 550.0
        # trial 1 has no missing data -> gets corrected to screen center
        np.testing.assert_allclose(x_out[1], 500.0)

    @pytest.mark.unit
    def test_saccade_in_drift_window_prevents_correction(self):
        eye_obj = EYE(sfreq=1000, screen_res=(1000, 800), viewing_dist=60,
                       screen_h=30)
        x1, y1 = fixation_saccade_profile(
            sfreq=1000, pre_ms=200, sac_ms=20, post_ms=200,
            start_xy=(550.0, 430.0), end_xy=(700.0, 400.0),
        )
        n = x1.size
        times = np.arange(-200, -200 + n)
        x = x1[None, :].copy()
        y = y1[None, :].copy()

        # drift window spans the saccade itself (times ~0-19)
        x_out, y_out = eye_obj.set_xy(x, y, times, drift_correct=(-10, 10))
        # pre-saccade fixation should remain uncorrected (still ~550)
        assert np.unique(np.round(x_out[0, :190], 3)).item() == 550.0

    @pytest.mark.unit
    def test_clean_pre_saccade_window_is_corrected(self):
        eye_obj = EYE(sfreq=1000, screen_res=(1000, 800), viewing_dist=60,
                       screen_h=30)
        x1, y1 = fixation_saccade_profile(
            sfreq=1000, pre_ms=200, sac_ms=20, post_ms=200,
            start_xy=(550.0, 430.0), end_xy=(700.0, 400.0),
        )
        n = x1.size
        times = np.arange(-200, -200 + n)
        x = x1[None, :].copy()
        y = y1[None, :].copy()

        # drift window entirely within the pre-saccade fixation, no saccade
        x_out, y_out = eye_obj.set_xy(x, y, times, drift_correct=(-100, 0))
        assert np.unique(np.round(x_out[0, :190], 3)).item() == 500.0


# ============================================================================
# EYE.angles_from_xy / EYE.create_angle_bins
# ============================================================================

class TestAnglesFromXyAndBins:
    @pytest.mark.unit
    def test_create_angle_bins_sustained_deviation(self):
        eye_obj = EYE(sfreq=1000, screen_res=(1920, 1080), viewing_dist=60,
                       screen_h=30)
        n = 100
        x = np.full(n, 960.0)
        y = np.full(n, 540.0)
        # 40px offset for 50 samples (50ms), independently verified to be
        # ~1.0397 degrees of visual angle
        x[50:100] = 1000.0

        bins, angles = eye_obj.create_angle_bins(
            x[None, :], y[None, :], start=0, stop=1.5, step=0.5,
            min_segment=40,
        )
        assert bins == [1.0]
        assert angles[0][99] == pytest.approx(1.0397217383649244)
        assert angles[0][0] == 0.0

    @pytest.mark.unit
    def test_create_angle_bins_brief_deviation_not_counted(self):
        eye_obj = EYE(sfreq=1000, screen_res=(1920, 1080), viewing_dist=60,
                       screen_h=30)
        n = 100
        x = np.full(n, 960.0)
        y = np.full(n, 540.0)
        # only 10 samples above threshold, below the 40-sample min_segment
        x[50:60] = 1000.0

        bins, angles = eye_obj.create_angle_bins(
            x[None, :], y[None, :], start=0, stop=1.5, step=0.5,
            min_segment=40,
        )
        assert np.isnan(bins[0])

    @pytest.mark.unit
    def test_angles_from_xy_matches_create_angle_bins_hardcoded_params(self):
        eye_obj = EYE(sfreq=1000, screen_res=(1920, 1080), viewing_dist=60,
                       screen_h=30)
        n = 100
        x = np.full((1, n), 960.0)
        y = np.full((1, n), 540.0)
        times = np.arange(n)

        angles = eye_obj.angles_from_xy(x, y, times)
        _, expected_angles = eye_obj.create_angle_bins(
            x, y, 0, 3, 0.25, 40
        )
        np.testing.assert_array_equal(angles[0], expected_angles[0])

    @pytest.mark.unit
    def test_angles_from_xy_applies_drift_correct_first(self):
        eye_obj = EYE(sfreq=1000, screen_res=(1000, 800), viewing_dist=60,
                       screen_h=30)
        n = 400
        times = np.arange(-200, 200)
        x = np.full((1, n), 550.0)
        y = np.full((1, n), 430.0)

        angles = eye_obj.angles_from_xy(x.copy(), y.copy(), times,
                                         drift_correct=(-100, 0))
        # after drift correction the trace sits exactly at screen center,
        # so deviation angle should be ~0 throughout
        np.testing.assert_allclose(angles[0], 0.0, atol=1e-9)


# ============================================================================
# EYE.link_eye_to_eeg
# ============================================================================

class TestLinkEyeToEeg:
    @pytest.mark.unit
    def test_end_to_end_pipeline(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        write_asc_session('sample.asc', n_trials=3)
        write_beh_csv('sample.csv', n_trials=3)

        eye_obj = EYE(sfreq=1000, screen_res=(1920, 1080))
        x, y, times, bins, angles, trial_info = eye_obj.link_eye_to_eeg(
            eye_file=['sample.asc'], beh_file=['sample.csv'],
            start_trial='start trial', stop_trial=None,
            window_oi=(-50, 50), trigger_msg='Onset search',
            drift_correct=None,
        )
        assert x.shape == (3, 100)
        assert y.shape == (3, 100)
        np.testing.assert_array_equal(times, np.arange(-50, 50))
        assert len(bins) == 3
        assert len(angles) == 3
        np.testing.assert_array_equal(trial_info, [0, 1, 2])

    @pytest.mark.unit
    def test_positional_arguments_reach_get_eye_data_correctly(
        self, tmp_path, monkeypatch
    ):
        """
        Regression test: link_eye_to_eeg used to call get_eye_data
        positionally (self.get_eye_data('', eye_file, beh_file,
        start_trial, trigger_msg, stop_trial)), which silently passed
        trigger_msg into get_eye_data's `trial_info` parameter and
        stop_trial into nothing (positional slot mismatch against the
        real signature). It now uses explicit keyword arguments.
        """
        monkeypatch.chdir(tmp_path)
        write_asc_session('sample.asc', n_trials=2)
        write_beh_csv('sample.csv', n_trials=2)

        eye_obj = EYE(sfreq=1000)
        x, y, times, bins, angles, trial_info = eye_obj.link_eye_to_eeg(
            eye_file=['sample.asc'], beh_file=['sample.csv'],
            start_trial='start trial', stop_trial=None,
            window_oi=(-20, 20), trigger_msg='Onset search',
            drift_correct=None,
        )
        # trial_info was never explicitly requested -> sequential fallback
        np.testing.assert_array_equal(trial_info, [0, 1])

    @pytest.mark.unit
    def test_drift_correct_applied_when_specified(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        write_asc_session('sample.asc', n_trials=1)
        write_beh_csv('sample.csv', n_trials=1)

        eye_obj = EYE(sfreq=1000, screen_res=(1920, 1080))
        x, y, times, bins, angles, trial_info = eye_obj.link_eye_to_eeg(
            eye_file=['sample.asc'], beh_file=['sample.csv'],
            start_trial='start trial', stop_trial=None,
            window_oi=(-50, 50), trigger_msg='Onset search',
            drift_correct=(-50, 0),
        )
        # gaze was recorded at a constant (500, 400): drift correction
        # should recenter it to screen center (960, 540), so deviation
        # angle should now be ~0
        np.testing.assert_allclose(angles[0], 0.0, atol=1e-6)


# ============================================================================
# EYE.calculate_angle / EYE.degrees_per_pixel
# ============================================================================

class TestCalculateAngleAndDegreesPerPixel:
    @pytest.mark.unit
    def test_calculate_angle_matches_independent_formula(self):
        eye_obj = EYE(viewing_dist=60, screen_res=(1920, 1080), screen_h=30)
        x = np.array([960.0, 1000.0, 920.0])
        y = np.array([540.0, 540.0, 540.0])

        angles = eye_obj.calculate_angle(x, y)
        deg_per_px = degrees(atan2(0.5 * 30, 60)) / (0.5 * 1080)
        expected = np.abs(x - 960.0) * deg_per_px
        np.testing.assert_allclose(angles, expected)

    @pytest.mark.unit
    def test_calculate_angle_custom_center(self):
        eye_obj = EYE(viewing_dist=60, screen_res=(1920, 1080), screen_h=30)
        x = np.array([100.0, 140.0])
        y = np.array([100.0, 100.0])

        angles = eye_obj.calculate_angle(x, y, xc=100.0, yc=100.0)
        deg_per_px = degrees(atan2(0.5 * 30, 60)) / (0.5 * 1080)
        np.testing.assert_allclose(angles, [0.0, 40.0 * deg_per_px])

    @pytest.mark.unit
    def test_degrees_per_pixel_static_method_matches_instance_formula(self):
        val = EYE.degrees_per_pixel(h=30, d=60, r=1080)
        expected = degrees(atan2(0.5 * 30, 60)) / (0.5 * 1080)
        assert val == pytest.approx(expected)

    @pytest.mark.unit
    def test_degrees_per_pixel_default_arguments(self):
        val = EYE.degrees_per_pixel()
        expected = degrees(atan2(0.5 * 30, 60)) / (0.5 * 1080)
        assert val == pytest.approx(expected)


# ============================================================================
# SaccadeDetector.__init__
# ============================================================================

class TestSaccadeDetectorInit:
    @pytest.mark.unit
    def test_attributes_and_deg_per_pixel(self):
        sd = SaccadeDetector(sfreq=1000, screen_h=30, viewing_dist=60,
                              screen_res=(1920, 1080))
        assert sd.sfreq == 1000.0
        assert sd.min_sac == 0.01
        expected = degrees(atan2(0.5 * 30, 60)) / (0.5 * 1080)
        assert sd.deg_per_pixel == pytest.approx(expected)


# ============================================================================
# SaccadeDetector.calc_velocity
# ============================================================================

class TestCalcVelocity:
    @pytest.mark.unit
    def test_linear_ramp_gives_constant_velocity_zero_acceleration(self):
        sd = SaccadeDetector(sfreq=1000, screen_h=30, viewing_dist=60,
                              screen_res=(1920, 1080))
        n = 300
        t = np.arange(n).astype(float)
        vx_true, vy_true = 0.5, -0.3
        x = 500.0 + vx_true * t
        y = 400.0 + vy_true * t

        V, A = sd.calc_velocity(x, y)
        expected_V = (np.sqrt(vx_true**2 + vy_true**2) * sd.deg_per_pixel
                      * sd.sfreq)
        np.testing.assert_allclose(V[20:280], expected_V, rtol=1e-6)
        np.testing.assert_allclose(A[20:280], 0.0, atol=1e-6)

    @pytest.mark.unit
    def test_quadratic_ramp_gives_constant_acceleration(self):
        sd = SaccadeDetector(sfreq=1000, screen_h=30, viewing_dist=60,
                              screen_res=(1920, 1080))
        n = 300
        t = np.arange(n).astype(float)
        ax_true, ay_true = 0.01, -0.005
        vx0, vy0 = 0.2, 0.1
        x = 500.0 + vx0 * t + 0.5 * ax_true * t**2
        y = 400.0 + vy0 * t + 0.5 * ay_true * t**2

        V, A = sd.calc_velocity(x, y)
        expected_A = (np.sqrt(ax_true**2 + ay_true**2) * sd.deg_per_pixel
                      * sd.sfreq**2)
        np.testing.assert_allclose(A[20:280], expected_A, rtol=1e-6)

        tt = 102
        vx_t, vy_t = vx0 + ax_true * tt, vy0 + ay_true * tt
        expected_V_t = (np.sqrt(vx_t**2 + vy_t**2) * sd.deg_per_pixel
                        * sd.sfreq)
        assert V[tt] == pytest.approx(expected_V_t, rel=1e-6)

    @pytest.mark.unit
    def test_insufficient_samples_returns_zeros(self):
        sd = SaccadeDetector(sfreq=1000)
        V, A = sd.calc_velocity(np.array([1.0, 2.0]), np.array([1.0, 2.0]))
        np.testing.assert_array_equal(V, [0.0, 0.0])
        np.testing.assert_array_equal(A, [0.0, 0.0])


# ============================================================================
# SaccadeDetector.noise_detect
# ============================================================================

class TestNoiseDetect:
    @pytest.mark.unit
    def test_blink_marked_nan_no_extension_when_velocity_low(self):
        sd = SaccadeDetector(sfreq=1000)
        n = 50
        x = np.full(n, 500.0)
        y = np.full(n, 400.0)
        x[20:25] = 0.0
        V = np.full(n, 1.0)  # on_off = 2*median(V) = 2.0, never exceeded
        A = np.full(n, 1.0)

        x_out, y_out, V_out, A_out = sd.noise_detect(x, y, V, A)
        np.testing.assert_array_equal(np.where(np.isnan(x_out))[0],
                                       [20, 21, 22, 23, 24])

    @pytest.mark.unit
    def test_blink_extended_while_velocity_stays_elevated(self):
        sd = SaccadeDetector(sfreq=1000)
        n = 50
        x = np.full(n, 500.0)
        y = np.full(n, 400.0)
        x[20:25] = 0.0
        V = np.full(n, 1.0)
        V[18:20] = 5.0
        V[25:27] = 5.0
        A = np.full(n, 1.0)

        x_out, y_out, V_out, A_out = sd.noise_detect(x, y, V, A)
        np.testing.assert_array_equal(
            np.where(np.isnan(x_out))[0],
            [18, 19, 20, 21, 22, 23, 24, 25, 26],
        )

    @pytest.mark.unit
    def test_velocity_threshold_flags_single_sample(self):
        sd = SaccadeDetector(sfreq=1000)
        n = 50
        x = np.full(n, 500.0)
        y = np.full(n, 400.0)
        V = np.full(n, 1.0)
        A = np.full(n, 1.0)
        V[30] = 2000.0  # exceeds default V_thresh=1000

        x_out, y_out, V_out, A_out = sd.noise_detect(x, y, V, A)
        np.testing.assert_array_equal(np.where(np.isnan(x_out))[0], [30])

    @pytest.mark.unit
    def test_no_noise_leaves_data_unchanged(self):
        sd = SaccadeDetector(sfreq=1000)
        n = 50
        x = np.full(n, 500.0)
        y = np.full(n, 400.0)
        V = np.full(n, 1.0)
        A = np.full(n, 1.0)

        x_out, y_out, V_out, A_out = sd.noise_detect(x.copy(), y.copy(), V, A)
        np.testing.assert_array_equal(x_out, x)
        assert not np.isnan(x_out).any()


# ============================================================================
# SaccadeDetector.estimate_thresh
# ============================================================================

class TestEstimateThresh:
    @pytest.mark.unit
    def test_constant_velocity_converges_to_its_own_value(self):
        sd = SaccadeDetector(sfreq=1000)
        V = np.full(1000, 5.0)
        sd.estimate_thresh(V)
        assert sd.peak_thresh == pytest.approx(5.0)
        assert sd.sacc_thresh == pytest.approx(5.0)

    @pytest.mark.unit
    def test_matches_independently_computed_center_statistics(self):
        sd = SaccadeDetector(sfreq=1000)
        n = 1000
        V = np.tile([4.0, 6.0], n // 2)  # one long fixation, mean=5, std=1

        cent_fix = 0.04 * sd.sfreq / 6
        center = V[int(math.floor(0 + cent_fix)):
                    int(math.ceil((n - 1) - cent_fix) + 1)]
        expected_peak = center.mean() + 6 * center.std()
        expected_sacc = center.mean() + 3 * center.std()

        sd.estimate_thresh(V)
        assert sd.peak_thresh == pytest.approx(expected_peak)
        assert sd.sacc_thresh == pytest.approx(expected_sacc)

    @pytest.mark.unit
    def test_short_fixation_segment_skipped_but_others_still_used(self):
        sd = SaccadeDetector(sfreq=1000)
        n = 500
        V = np.full(n, 5.0)
        V[100:130] = 150.0   # separator 1
        V[130:150] = 5.0     # short fixation segment: 20 samples (< 40ms min_fix)
        V[150:180] = 150.0   # separator 2
        V[180:500] = 5.0     # long fixation segment

        sd.estimate_thresh(V)
        # both surviving (long) segments are constant at 5.0 -> std=0
        assert sd.peak_thresh == pytest.approx(5.0)
        assert sd.sacc_thresh == pytest.approx(5.0)

    @pytest.mark.unit
    def test_no_fixation_periods_sets_none(self):
        sd = SaccadeDetector(sfreq=1000)
        V = np.full(1000, 200.0)  # always above initial peak_thresh=100
        sd.estimate_thresh(V)
        assert sd.peak_thresh is None
        assert sd.sacc_thresh is None


# ============================================================================
# SaccadeDetector.saccade_detection
# ============================================================================

class TestSaccadeDetection:
    @pytest.mark.unit
    def test_clean_saccade_detected_with_correct_onset_offset(self):
        sd = SaccadeDetector(sfreq=1000)
        sd.peak_thresh = 50.0
        sd.sacc_thresh = 20.0
        n = 200
        V = np.full(n, 2.0)
        V[100:110] = 200.0

        assert sd.saccade_detection(V, output='mask') == 1
        result = sd.saccade_detection(V, output='dict')
        assert result['saccades'] == {'1': (99, 110)}
        assert result['glissades'] == {}

    @pytest.mark.unit
    def test_no_saccade_when_velocity_never_exceeds_peak_thresh(self):
        sd = SaccadeDetector(sfreq=1000)
        sd.peak_thresh = 50.0
        sd.sacc_thresh = 20.0
        V = np.full(200, 2.0)
        assert sd.saccade_detection(V, output='mask') == 0

    @pytest.mark.unit
    def test_single_sample_spike_rejected_as_too_brief(self):
        sd = SaccadeDetector(sfreq=1000)
        sd.peak_thresh = 50.0
        sd.sacc_thresh = 20.0
        V = np.full(200, 2.0)
        V[100] = 200.0
        assert sd.saccade_detection(V, output='mask') == 0

    @pytest.mark.unit
    def test_glissade_detected_after_saccade(self):
        sd = SaccadeDetector(sfreq=1000)
        sd.peak_thresh = 50.0
        sd.sacc_thresh = 20.0
        n = 200
        V = np.full(n, 2.0)
        V[100:110] = 200.0   # main saccade, offset lands at 110
        V[112:116] = 20.0    # low-velocity glissade shoulder afterwards

        result = sd.saccade_detection(V, output='dict')
        assert result['saccades'] == {'1': (99, 110)}
        assert result['glissades'] == {'1': (110, 115)}

    @pytest.mark.unit
    def test_high_velocity_glissade_bump_also_skipped_as_separate_saccade(
        self
    ):
        """
        A glissade shoulder that itself exceeds peak_thresh (a "high
        velocity" glissade) is both recorded via the high-velocity
        glissade branch and, since it also independently qualifies as
        a poss_sac segment, correctly skipped as a would-be second
        saccade because its samples already belong to the just
        recorded glissade.
        """
        sd = SaccadeDetector(sfreq=1000)
        sd.peak_thresh = 50.0
        sd.sacc_thresh = 20.0
        n = 300
        V = np.full(n, 2.0)
        V[100:110] = 200.0
        V[120:126] = 60.0  # exceeds peak_thresh=50 -> high-velocity glissade

        result = sd.saccade_detection(V, output='dict')
        # only ONE saccade recorded -- the bump was absorbed into the
        # glissade, not double-counted as a second saccade
        assert result['saccades'] == {'1': (99, 110)}
        assert result['glissades'] == {'1': (110, 125)}

    @pytest.mark.unit
    def test_glissade_exceeding_saccade_amplitude_rejected(self):
        sd = SaccadeDetector(sfreq=1000)
        sd.peak_thresh = 50.0
        sd.sacc_thresh = 20.0
        n = 300
        V = np.full(n, 2.0)
        V[100:110] = 200.0
        V[120:126] = 250.0  # exceeds the saccade's own peak (200)

        result = sd.saccade_detection(V, output='dict')
        assert result['saccades'] == {'1': (99, 110)}
        assert result['glissades'] == {}

    @pytest.mark.unit
    def test_glissade_offset_shifted_forward_when_velocity_still_rising(
        self
    ):
        """
        Independently verified via direct numpy computation of the
        documented algorithm: when the naive glissade offset still has
        a later uptick in velocity, the offset is shifted forward to
        the first point where velocity stops rising.
        """
        sd = SaccadeDetector(sfreq=1000)
        sd.peak_thresh = 50.0
        sd.sacc_thresh = 20.0
        n = 300
        V = np.full(n, 2.0)
        V[100:110] = 200.0
        V[110:112] = 5.0
        V[112:114] = 30.0
        V[114] = 8.0
        V[115] = 6.0
        V[116] = 4.0
        V[117] = 5.0   # uptick -> forces the forward shift
        V[118:] = 2.0

        result = sd.saccade_detection(V, output='dict')
        assert result['glissades'] == {'1': (110, 115)}

    @pytest.mark.unit
    def test_glissade_rejected_when_corrected_offset_too_far(self):
        sd = SaccadeDetector(sfreq=1000)
        sd.peak_thresh = 50.0
        sd.sacc_thresh = 20.0
        n = 400
        V = np.full(n, 2.0)
        V[100:110] = 200.0
        V[110:112] = 5.0
        V[112:114] = 30.0
        # a "complete" peak within the 40-sample poss_gliss window
        # (falls back below threshold right at 114, giving a valid
        # naive gliss_w_off), but velocity keeps slowly declining far
        # beyond that window before finally ticking back up --
        # forcing the forward-shift correction out past 2*min_fix
        # (80 samples) from sac_off
        V[114:300] = np.linspace(7.9, 3.0, 300 - 114)
        V[300:] = 3.5

        result = sd.saccade_detection(V, output='dict')
        assert result['saccades'] == {'1': (99, 110)}
        assert result['glissades'] == {}


# ============================================================================
# SaccadeDetector.detect_events
# ============================================================================

class TestDetectEvents:
    @pytest.mark.unit
    def test_saccade_trial_true_stable_fixation_false(self):
        sd = SaccadeDetector(sfreq=1000, screen_h=30, viewing_dist=60,
                              screen_res=(1920, 1080))
        x0, y0 = fixation_saccade_profile(
            sfreq=1000, pre_ms=300, sac_ms=20, post_ms=300,
            start_xy=(960.0, 540.0), end_xy=(1200.0, 540.0),
        )
        x1 = np.full(x0.size, 960.0)
        y1 = np.full(x0.size, 540.0)

        mask = sd.detect_events(np.vstack([x0, x1]), np.vstack([y0, y1]))
        np.testing.assert_array_equal(mask, [True, False])

    @pytest.mark.unit
    def test_1d_input_auto_reshaped(self):
        sd = SaccadeDetector(sfreq=1000, screen_h=30, viewing_dist=60,
                              screen_res=(1920, 1080))
        x0, y0 = fixation_saccade_profile(
            sfreq=1000, pre_ms=300, sac_ms=20, post_ms=300,
            start_xy=(960.0, 540.0), end_xy=(1200.0, 540.0),
        )
        mask = sd.detect_events(x0, y0)
        assert mask.shape == (1,)
        assert mask[0] == True

    @pytest.mark.unit
    def test_dict_output_not_cast_to_bool(self):
        sd = SaccadeDetector(sfreq=1000, screen_h=30, viewing_dist=60,
                              screen_res=(1920, 1080))
        x0, y0 = fixation_saccade_profile(
            sfreq=1000, pre_ms=300, sac_ms=20, post_ms=300,
            start_xy=(960.0, 540.0), end_xy=(1200.0, 540.0),
        )
        result = sd.detect_events(x0, y0, output='dict')
        assert isinstance(result[0], dict)
        assert 'saccades' in result[0]

    @pytest.mark.unit
    def test_missing_data_trial_preserved_as_nan_not_cast_to_true(self):
        """
        Regression test: detect_events used to blanket-cast its
        per-trial results to bool via np.array(sacc, dtype=bool),
        which silently turned np.nan (missing/undetermined data) into
        True (nonzero float -> truthy), contradicting its own
        docstring ("Trials containing NaN values are marked as np.nan
        in the output") and inflating naive `.sum()` usage.
        """
        sd = SaccadeDetector(sfreq=1000, screen_h=30, viewing_dist=60,
                              screen_res=(1920, 1080))
        n = 200
        x_clean = np.full(n, 960.0) + np.random.RandomState(0).normal(
            0, 0.5, n)
        y_clean = np.full(n, 540.0) + np.random.RandomState(1).normal(
            0, 0.5, n)
        x_missing = np.full(n, np.nan)
        y_missing = np.full(n, np.nan)

        mask = sd.detect_events(np.vstack([x_clean, x_missing]),
                                 np.vstack([y_clean, y_missing]))
        assert mask[0] == False
        assert np.isnan(mask[1])
        # naive docstring-style usage should not be silently inflated
        assert np.nansum(mask) == 0

    @pytest.mark.unit
    def test_inconclusive_threshold_estimation_preserved_as_nan(self):
        sd = SaccadeDetector(sfreq=1000, screen_h=30, viewing_dist=60,
                              screen_res=(1920, 1080))
        # too short/erratic to yield a valid adaptive threshold
        n = 30
        rng = np.random.RandomState(0)
        x = 960.0 + rng.normal(0, 50, n)
        y = 540.0 + rng.normal(0, 50, n)

        mask = sd.detect_events(x, y)
        assert np.isnan(mask[0])
