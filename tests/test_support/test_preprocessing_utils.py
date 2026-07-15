"""
Test suite for open_dvm.support.preprocessing_utils.

Organization
------------
- TestFormatSubjectId: subject/session ID formatting
- TestFindRawFiles: raw file discovery with flexible naming
- TestSelectElectrodes: region-based and explicit electrode selection
- TestBaselineCorrection: baseline subtraction
- TestMatchEpochsTimes: time-axis synchronization across evoked objects
- TestGetDiffPairs: contra-ipsi electrode pairing
- TestTrialExclusion: OR-logic trial exclusion
- TestGetTimeSlice: time-window-to-slice conversion
- TestCreateCndLoop: factorial condition combination generation
- TestLogPreproc: JSON-based preprocessing logging
"""

import json
import warnings

import pytest
import numpy as np
import pandas as pd
import mne

from open_dvm.support.preprocessing_utils import (
    format_subject_id,
    find_raw_files,
    select_electrodes,
    baseline_correction,
    match_epochs_times,
    get_diff_pairs,
    trial_exclusion,
    get_time_slice,
    create_cnd_loop,
    log_preproc,
)

mne.set_log_level('ERROR')


# ============================================================================
# format_subject_id
# ============================================================================

class TestFormatSubjectId:
    @pytest.mark.unit
    @pytest.mark.parametrize('sj_id,expected', [
        (1, '01'), ('1', '01'), ('sub_1', '01'), ('01', '01'),
    ])
    def test_formats_various_inputs(self, sj_id, expected):
        assert format_subject_id(sj_id) == expected

    @pytest.mark.unit
    def test_custom_zero_pad(self):
        assert format_subject_id(5, zero_pad=3) == '005'

    @pytest.mark.unit
    def test_no_digits_raises(self):
        with pytest.raises(ValueError):
            format_subject_id('no_digits_here')


# ============================================================================
# find_raw_files
# ============================================================================

class TestFindRawFiles:
    @pytest.mark.unit
    def test_finds_single_match_any_padding(self, tmp_path):
        (tmp_path / 'sub_2_ses_1.bdf').touch()

        files = find_raw_files(str(tmp_path), sj=2, session=1)

        assert len(files) == 1
        assert 'sub_2_ses_1.bdf' in files[0]

    @pytest.mark.unit
    def test_no_match_returns_empty_list(self, tmp_path):
        (tmp_path / 'sub_2_ses_1.bdf').touch()

        files = find_raw_files(str(tmp_path), sj=3, session=1)

        assert files == []

    @pytest.mark.unit
    def test_flexible_run_accepts_file_without_run_suffix(self, tmp_path):
        (tmp_path / 'sub_2_ses_1.bdf').touch()  # no run suffix

        files = find_raw_files(str(tmp_path), sj=2, session=1, run=1, flexible_run=True)

        assert len(files) == 1

    @pytest.mark.unit
    def test_strict_run_rejects_file_without_run_suffix(self, tmp_path):
        (tmp_path / 'sub_2_ses_1.bdf').touch()

        files = find_raw_files(str(tmp_path), sj=2, session=1, run=1, flexible_run=False)

        assert files == []

    @pytest.mark.unit
    def test_multiple_ambiguous_matches_raise_regression(self, tmp_path):
        """Regression test: the docstring always promised a
        FileNotFoundError on multiple matches, but the check was
        missing -- it used to silently return both files instead."""
        (tmp_path / 'sub_02_ses_01_run_01.bdf').touch()
        (tmp_path / 'sub_002_ses_001_run_001.bdf').touch()

        with pytest.raises(FileNotFoundError):
            find_raw_files(str(tmp_path), sj=2, session=1, run=1)


# ============================================================================
# select_electrodes
# ============================================================================

class TestSelectElectrodes:
    @staticmethod
    def _make_epochs():
        montage = mne.channels.make_standard_montage('biosemi64')
        info = mne.create_info(montage.ch_names, 100, ch_types='eeg')
        return mne.EpochsArray(np.zeros((2, 64, 5)), info, tmin=0)

    @pytest.mark.unit
    def test_all_selects_every_channel(self):
        epochs = self._make_epochs()

        picks = select_electrodes(epochs, 'all')

        assert len(picks) == 64

    @pytest.mark.unit
    def test_posterior_excludes_frontal_channels(self):
        epochs = self._make_epochs()

        picks = select_electrodes(epochs, 'posterior')
        names = [epochs.ch_names[i] for i in picks]

        assert 'Oz' in names and 'Pz' in names
        assert 'Fp1' not in names

    @pytest.mark.unit
    def test_frontal_excludes_posterior_channels(self):
        epochs = self._make_epochs()

        picks = select_electrodes(epochs, 'frontal')
        names = [epochs.ch_names[i] for i in picks]

        assert 'Fp1' in names and 'Fz' in names
        assert 'Oz' not in names

    @pytest.mark.unit
    def test_explicit_list_preserves_only_existing_channels(self):
        epochs = self._make_epochs()

        picks = select_electrodes(epochs, ['Fp1', 'Fp2', 'Oz'])

        assert [epochs.ch_names[i] for i in picks] == ['Fp1', 'Fp2', 'Oz']

    @pytest.mark.unit
    def test_no_matching_electrodes_warns_and_returns_empty(self):
        epochs = self._make_epochs()

        with pytest.warns(UserWarning, match='None of the specified'):
            picks = select_electrodes(epochs, ['NoSuchElectrode'])

        assert len(picks) == 0


# ============================================================================
# baseline_correction
# ============================================================================

class TestBaselineCorrection:
    @pytest.mark.unit
    def test_subtracts_baseline_mean_per_trial_and_channel(self):
        times = np.linspace(-0.2, 0.2, 41)
        X = np.zeros((2, 3, 41))
        # trial0/ch0: baseline mean=5, trial1/ch1: baseline mean=-2;
        # post-baseline samples carry a distinguishing signal so the
        # correction (subtract baseline mean from ALL samples) is
        # actually exercised, not just the trivial all-equal case
        X[0, 0, :21] = 5.0  # baseline window (samples up to t=0 inclusive)
        X[0, 0, 21:] = 8.0  # post-baseline signal
        X[1, 1, :21] = -2.0
        X[1, 1, 21:] = 1.0

        corrected = baseline_correction(X, times, baseline=(-0.2, 0))

        # baseline window mean is exactly zeroed
        assert corrected[0, 0, :21].mean() == pytest.approx(0, abs=1e-10)
        assert corrected[1, 1, :21].mean() == pytest.approx(0, abs=1e-10)
        # post-baseline signal is preserved, shifted by the same offset
        np.testing.assert_allclose(corrected[0, 0, 21:], 8.0 - 5.0)
        np.testing.assert_allclose(corrected[1, 1, 21:], 1.0 - (-2.0))

    @pytest.mark.unit
    def test_non_3d_input_raises(self):
        times = np.linspace(0, 1, 10)
        X = np.zeros((3, 10))  # 2D, not 3D
        with pytest.raises(ValueError):
            baseline_correction(X, times, baseline=(0, 0.5))


# ============================================================================
# match_epochs_times
# ============================================================================

class TestMatchEpochsTimes:
    @pytest.mark.unit
    def test_truncates_longer_evoked_to_match_shortest(self):
        info = mne.create_info(['C1', 'C2'], 100, ch_types='eeg')
        short_data = np.ones((2, 10))
        long_data = np.ones((2, 15)) * 2.0
        ev_short = mne.EvokedArray(short_data, info, tmin=0)
        ev_long = mne.EvokedArray(long_data, info, tmin=0)

        result = match_epochs_times([ev_short, ev_long])

        assert result[0].data.shape == (2, 10)
        assert result[1].data.shape == (2, 10)
        np.testing.assert_allclose(result[1].data, long_data[:, :10])

    @pytest.mark.unit
    def test_shortest_evoked_object_identity_preserved(self):
        info = mne.create_info(['C1'], 100, ch_types='eeg')
        ev_short = mne.EvokedArray(np.ones((1, 10)), info, tmin=0)
        ev_long = mne.EvokedArray(np.ones((1, 15)), info, tmin=0)

        result = match_epochs_times([ev_short, ev_long])

        assert result[0] is ev_short


# ============================================================================
# get_diff_pairs
# ============================================================================

class TestGetDiffPairs:
    @pytest.mark.unit
    def test_bidirectional_lateral_pairing(self):
        ch_names = ['P7', 'P8', 'Fp1', 'Fp2']

        pairs = get_diff_pairs(ch_names)

        assert pairs['P7'] == (0, 1)
        assert pairs['P8'] == (1, 0)
        assert pairs['Fp1'] == (2, 3)
        assert pairs['Fp2'] == (3, 2)

    @pytest.mark.unit
    def test_midline_maps_to_itself(self):
        ch_names = ['Pz', 'Cz']

        pairs = get_diff_pairs(ch_names)

        assert pairs['Pz'] == (0, 0)
        assert pairs['Cz'] == (1, 1)

    @pytest.mark.unit
    def test_only_pairs_present_in_ch_names_included(self):
        ch_names = ['P7']  # P8 missing -> pair incomplete

        pairs = get_diff_pairs(ch_names)

        assert pairs == {}

    @pytest.mark.unit
    def test_no_pairs_found_warns(self):
        with pytest.warns(UserWarning, match='No standard electrode pairs'):
            get_diff_pairs(['NotAnElectrode'])


# ============================================================================
# trial_exclusion
# ============================================================================

class TestTrialExclusion:
    @pytest.mark.unit
    def test_or_logic_across_and_within_keys(self):
        df = pd.DataFrame({'correct': [1, 0, 1, 0, 1], 'cue': ['l', 'l', 'r', 'r', 'l']})
        info = mne.create_info(['C1'], 100, ch_types='eeg')
        epochs = mne.EpochsArray(np.zeros((5, 1, 10)), info, tmin=0)

        out_df, out_epochs, idx = trial_exclusion(
            df.copy(), epochs.copy(), {'correct': [0], 'cue': ['r']}
        )

        # correct==0 -> {1,3}; cue=='r' -> {2,3}; union -> {1,2,3}
        np.testing.assert_array_equal(sorted(idx.tolist()), [1, 2, 3])
        assert len(out_epochs) == 2
        assert len(out_df) == 2

    @pytest.mark.unit
    def test_no_matching_trials_excludes_nothing(self, capsys):
        df = pd.DataFrame({'correct': [1, 1, 1]})
        info = mne.create_info(['C1'], 100, ch_types='eeg')
        epochs = mne.EpochsArray(np.zeros((3, 1, 10)), info, tmin=0)

        out_df, out_epochs, idx = trial_exclusion(df.copy(), epochs.copy(), {'correct': [0]})

        assert len(idx) == 0
        assert len(out_epochs) == 3
        assert 'no trials selected' in capsys.readouterr().out


# ============================================================================
# get_time_slice
# ============================================================================

class TestGetTimeSlice:
    @pytest.mark.unit
    def test_explicit_start_and_end(self):
        times = np.linspace(-0.2, 1.0, 601)
        s = get_time_slice(times, -0.2, 0)
        assert times[s][0] == pytest.approx(-0.2)
        assert times[s][-1] == pytest.approx(0.0)

    @pytest.mark.unit
    def test_both_none_returns_full_range(self):
        times = np.linspace(-0.2, 1.0, 601)
        s = get_time_slice(times, None, None)
        assert times[s][0] == pytest.approx(times[0])
        assert times[s][-1] == pytest.approx(times[-1])

    @pytest.mark.unit
    def test_start_none_uses_first_sample(self):
        times = np.linspace(-0.2, 1.0, 601)
        s = get_time_slice(times, None, 0.5)
        assert times[s][0] == pytest.approx(times[0])
        assert times[s][-1] == pytest.approx(0.5)

    @pytest.mark.unit
    def test_end_none_uses_last_sample(self):
        times = np.linspace(-0.2, 1.0, 601)
        s = get_time_slice(times, 0.2, None)
        assert times[s][0] == pytest.approx(0.2)
        assert times[s][-1] == pytest.approx(times[-1])

    @pytest.mark.unit
    def test_include_final_false_excludes_endpoint(self):
        times = np.linspace(0, 1.0, 101)  # 0.01 spacing
        s = get_time_slice(times, 0, 0.1, include_final=False)
        assert times[s][-1] < 0.1

    @pytest.mark.unit
    def test_include_final_true_includes_endpoint(self):
        times = np.linspace(0, 1.0, 101)
        s = get_time_slice(times, 0, 0.1, include_final=True)
        assert times[s][-1] == pytest.approx(0.1)

    @pytest.mark.unit
    def test_step_applies_to_slice(self):
        times = np.linspace(0, 1.0, 101)
        s = get_time_slice(times, 0, 0.5, step=2)
        assert s.step == 2


# ============================================================================
# create_cnd_loop
# ============================================================================

class TestCreateCndLoop:
    @pytest.mark.unit
    def test_generates_all_factorial_combinations(self):
        cnds = {'cue_side': ['left', 'right'], 'validity': ['valid', 'invalid']}

        filters = create_cnd_loop(cnds)

        names = [name for _, name in filters]
        assert names == ['left_valid', 'left_invalid', 'right_valid', 'right_invalid']

    @pytest.mark.unit
    def test_query_string_selects_correct_rows(self):
        cnds = {'cue_side': ['left', 'right'], 'validity': ['valid', 'invalid']}
        filters = create_cnd_loop(cnds)
        df = pd.DataFrame({
            'cue_side': ['left', 'left', 'right', 'right'],
            'validity': ['valid', 'invalid', 'valid', 'invalid'],
            'row_id': [0, 1, 2, 3],
        })

        for query, name in filters:
            matched = df.query(query)
            assert len(matched) == 1
            expected_row_id = names_to_row = {
                'left_valid': 0, 'left_invalid': 1,
                'right_valid': 2, 'right_invalid': 3,
            }[name]
            assert matched['row_id'].iloc[0] == expected_row_id

    @pytest.mark.unit
    def test_numeric_condition_values(self):
        cnds = {'set_size': [2, 4]}
        filters = create_cnd_loop(cnds)
        assert filters == [('set_size == 2', '2'), ('set_size == 4', '4')]


# ============================================================================
# log_preproc
# ============================================================================

class TestLogPreproc:
    @pytest.mark.unit
    def test_creates_new_file_with_entry(self, tmp_path):
        logfile = str(tmp_path / 'sub' / 'log.json')

        log_preproc((1, 1), logfile, to_update={'high_pass': 0.1})

        with open(logfile) as f:
            data = json.load(f)
        assert data['subject_01_session_01']['high_pass'] == 0.1

    @pytest.mark.unit
    def test_list_values_converted_to_strings(self, tmp_path):
        logfile = str(tmp_path / 'log.json')

        log_preproc((1, 1), logfile, to_update={'bad_chs': ['Fp1', 'Fp2']})

        with open(logfile) as f:
            data = json.load(f)
        assert data['subject_01_session_01']['bad_chs'] == "['Fp1', 'Fp2']"

    @pytest.mark.unit
    def test_preserves_existing_entries_for_other_subjects(self, tmp_path):
        logfile = str(tmp_path / 'log.json')

        log_preproc((1, 1), logfile, to_update={'a': 1})
        log_preproc((2, 1), logfile, to_update={'b': 2})

        with open(logfile) as f:
            data = json.load(f)
        assert set(data.keys()) == {'subject_01_session_01', 'subject_02_session_01'}

    @pytest.mark.unit
    def test_recovers_from_corrupted_file(self, tmp_path):
        logfile = str(tmp_path / 'log.json')
        with open(logfile, 'w') as f:
            f.write('not valid json{{{')

        log_preproc((3, 1), logfile, to_update={'x': 1})

        with open(logfile) as f:
            data = json.load(f)
        assert data == {'subject_03_session_01': {'x': 1}}


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
