"""
Test suite for open_dvm.analysis.ERP.

Organization
------------
- TestSpatialRestrictionANDLogic: select_lateralization_idx AND-logic
- TestFlipTopography: flip_topography electrode/HEOG transformation
- TestSelectErpData: select_erp_data trial exclusion + flip integration
- TestCreateErps: create_erps averaging, baseline, cropping, RT split, I/O
- TestConditionErpsIntegration: condition_erps end-to-end file output
- TestExtractERPFeatures / TestERPDataValidation / TestERPComponentAnalysis:
  extract_erp_features metrics and edge cases
- TestSelectErpWindow: select_erp_window peak detection
- TestLatencyAnalysisConditionNames / TestStatisticalComparisons:
  jackknife_contrast / jack_latency_contrast
- TestCompareLatencies: compare_latencies (list and dict forms)
- TestGroupERPAnalysis / TestSelectWaveformVariations / TestGetERPParams:
  group-level aggregation utilities
- TestExportErpMetricsToCsv: export_erp_metrics_to_csv output
- TestResidualEye: residual_eye HEOG computation
- TestGenerateErpReport: generate_erp_report smoke test
- TestEdgeCases / TestRegressions / TestERPValidationAndErrors /
  TestERPDataInjectionAndValidation: cross-cutting edge cases
"""

import pytest
import numpy as np
import pandas as pd
import mne

from open_dvm.analysis.ERP import ERP
from tests.fixtures.sample_data import (
    create_lateralization_test_data,
    create_multilocation_test_data,
)


# ============================================================================
# select_lateralization_idx: AND-logic spatial restriction
# ============================================================================

class TestSpatialRestrictionANDLogic:
    """Tests for the AND logic in spatial restrictions.

    Multi-key spatial restrictions require ALL constraints to be
    satisfied (intersection), not any (union).
    """

    @pytest.mark.unit
    def test_select_lateralization_idx_simple_and_logic(self):
        """Single constraint filters correctly."""
        data, trial_info = create_lateralization_test_data()

        idx = ERP.select_lateralization_idx(
            trial_info,
            pos_labels={'dist1_loc': [0]}
        )

        expected = np.array([0, 1, 2, 3])
        np.testing.assert_array_equal(idx, expected)

    @pytest.mark.unit
    def test_select_lateralization_idx_multi_key_and_logic(self):
        """Two constraints require both to hold (intersection, not union).

        Bug regression: {'dist1_loc': [0]} AND {'dist2_loc': [4]} must
        return [1], not the union [0,1,2,3,4,6,8].
        """
        data, trial_info = create_lateralization_test_data()

        idx = ERP.select_lateralization_idx(
            trial_info,
            pos_labels={'dist1_loc': [0]},
            spatial_restriction={'dist2_loc': [4]}
        )

        expected = np.array([1])
        np.testing.assert_array_equal(idx, expected)

    @pytest.mark.unit
    def test_select_lateralization_idx_multi_value_and_logic(self):
        """Multiple values per key still combine with AND across keys."""
        data, trial_info = create_multilocation_test_data()

        idx = ERP.select_lateralization_idx(
            trial_info,
            pos_labels={'dist1_loc': [0, 1]},
            spatial_restriction={'dist2_loc': [0, 4]}
        )

        expected = np.array([0, 1, 3, 4])
        np.testing.assert_array_equal(idx, expected)

    @pytest.mark.unit
    def test_select_lateralization_idx_no_matches(self):
        """An impossible constraint combination returns an empty array."""
        data, trial_info = create_lateralization_test_data()

        idx = ERP.select_lateralization_idx(
            trial_info,
            pos_labels={'dist1_loc': [0]},
            spatial_restriction={'dist2_loc': [99]}
        )

        np.testing.assert_array_equal(idx, np.array([], dtype=int))

    @pytest.mark.unit
    def test_select_lateralization_idx_none_restriction(self):
        """None spatial_restriction returns only the pos_labels matches."""
        data, trial_info = create_lateralization_test_data()

        idx = ERP.select_lateralization_idx(
            trial_info,
            pos_labels={'dist1_loc': [0, 1, 2, 3, 4, 5, 6, 9]},
            spatial_restriction=None
        )

        np.testing.assert_array_equal(idx, np.arange(10))

    @pytest.mark.unit
    def test_select_lateralization_idx_single_key_vs_multi_key(self):
        """Adding a spatial_restriction can only narrow the selection."""
        data, trial_info = create_multilocation_test_data()

        idx_single = ERP.select_lateralization_idx(
            trial_info,
            pos_labels={'dist1_loc': [0, 1]},
            spatial_restriction=None
        )
        idx_multi = ERP.select_lateralization_idx(
            trial_info,
            pos_labels={'dist1_loc': [0, 1]},
            spatial_restriction={'dist2_loc': [0]}
        )

        assert len(idx_single) > len(idx_multi)
        assert len(idx_multi) == 2

    @pytest.mark.unit
    def test_select_lateralization_idx_missing_restriction_column_raises(self):
        """A spatial_restriction key absent from trial_info raises KeyError."""
        trial_info = pd.DataFrame({'trial': np.arange(5), 'loc': [0, 1, 2, 3, 4]})

        with pytest.raises(KeyError):
            ERP.select_lateralization_idx(
                trial_info,
                pos_labels={'loc': [0, 1, 2, 3, 4]},
                spatial_restriction={'missing_column': [0]}
            )


# ============================================================================
# flip_topography
# ============================================================================

class TestFlipTopography:
    """Tests for topographic flipping of lateralized trials."""

    @pytest.mark.unit
    def test_flip_swaps_electrode_pairs_for_flagged_trials(
        self, lateralized_flip_epochs
    ):
        """Trials matching topo_flip get O1<->O2 and P7<->P8 swapped."""
        epochs, trial_info = lateralized_flip_epochs

        flipped = ERP.flip_topography(
            epochs.copy(), trial_info, topo_flip={'target_loc': [1]}
        )
        data = flipped.get_data()[:, :, 0]  # constant across time

        # trial 0 (target_loc == 1): flipped
        np.testing.assert_array_equal(data[0], [1, 0, 3, 2, -4])
        # trial 2 (target_loc == 1): flipped, offset by 20
        np.testing.assert_array_equal(data[2], [21, 20, 23, 22, -24])

    @pytest.mark.unit
    def test_flip_leaves_unflagged_trials_unchanged(
        self, lateralized_flip_epochs
    ):
        """Trials not matching topo_flip are untouched."""
        epochs, trial_info = lateralized_flip_epochs

        flipped = ERP.flip_topography(
            epochs.copy(), trial_info, topo_flip={'target_loc': [1]}
        )
        data = flipped.get_data()[:, :, 0]

        # trial 1 (target_loc == 2): unchanged
        np.testing.assert_array_equal(data[1], [10, 11, 12, 13, 14])

    @pytest.mark.unit
    def test_flip_uses_custom_flip_dict(self, lateralized_flip_epochs):
        """An explicit flip_dict overrides the auto-generated odd/even pairing."""
        epochs, trial_info = lateralized_flip_epochs

        flipped = ERP.flip_topography(
            epochs.copy(), trial_info, topo_flip={'target_loc': [1]},
            flip_dict={'O1': 'P8'}
        )
        data = flipped.get_data()[:, :, 0]

        # only O1<->P8 swapped for trial 0; O2, P7 untouched, HEOG untouched
        # (custom flip_dict does not include HEOG in the swap loop, but
        # HEOG polarity is still inverted for flipped trials)
        np.testing.assert_array_equal(data[0], [3, 1, 2, 0, -4])


# ============================================================================
# select_erp_data
# ============================================================================

class TestSelectErpData:
    """Tests for select_erp_data trial exclusion and flip integration."""

    @pytest.mark.unit
    def test_excludes_trials_matching_excl_factor(self, sample_epochs_data,
                                                    sample_trial_dataframe):
        epochs, _ = sample_epochs_data
        df = sample_trial_dataframe
        n_incorrect = (~df['correct']).sum()

        erp = ERP(sj=1, epochs=epochs, df=df, baseline=(-0.2, 0))
        out_df, out_epochs = erp.select_erp_data(excl_factor={'correct': [False]})

        assert len(out_df) == len(df) - n_incorrect
        assert len(out_epochs) == len(df) - n_incorrect
        assert (out_df['correct']).all()

    @pytest.mark.unit
    def test_topo_flip_delegates_to_flip_topography(self, lateralized_flip_epochs):
        epochs, trial_info = lateralized_flip_epochs

        erp = ERP(sj=1, epochs=epochs, df=trial_info, baseline=(-0.05, 0))
        _, out_epochs = erp.select_erp_data(topo_flip={'target_loc': [1]})

        data = out_epochs.get_data()[:, :, 0]
        np.testing.assert_array_equal(data[0], [1, 0, 3, 2, -4])
        np.testing.assert_array_equal(data[1], [10, 11, 12, 13, 14])

    @pytest.mark.unit
    def test_no_flip_when_topo_flip_none(self, lateralized_flip_epochs):
        epochs, trial_info = lateralized_flip_epochs

        erp = ERP(sj=1, epochs=epochs, df=trial_info, baseline=(-0.05, 0))
        _, out_epochs = erp.select_erp_data()

        np.testing.assert_array_equal(
            out_epochs.get_data(), epochs.get_data()
        )


# ============================================================================
# create_erps
# ============================================================================

class TestCreateErps:
    """Tests for create_erps trial averaging, baseline, cropping, RT split."""

    @staticmethod
    def _build_epochs_df():
        ch_names = ['Cz', 'Pz']
        sfreq, n_samples, tmin = 100, 20, -0.1
        rng = np.random.default_rng(0)
        data = rng.normal(0, 1, size=(4, len(ch_names), n_samples))
        for t in range(4):
            data[t] += t  # per-trial offset, present in baseline and post-baseline
        info = mne.create_info(ch_names, sfreq, ch_types='eeg')
        epochs = mne.EpochsArray(data.copy(), info, tmin=tmin)
        df = pd.DataFrame({'RT': [100, 900, 200, 800]})
        return epochs, df, data

    def test_idx_none_raises_typeerror(self):
        """Regression/documentation test: create_erps does NOT support
        idx=None despite the docstring saying "If None, all epochs are
        averaged" -- df.iloc[None] raises. Callers must pass an explicit
        index array (e.g. np.arange(len(df))) to average all trials.
        """
        epochs, df, _ = self._build_epochs_df()
        erp = ERP(sj=1, epochs=epochs, df=df, baseline=(-0.1, 0))

        with pytest.raises(TypeError):
            erp.create_erps(epochs, df, idx=None, erp_name='x', save=False)

    def test_averages_and_baseline_corrects_selected_trials(self):
        epochs, df, data = self._build_epochs_df()
        erp = ERP(sj=1, epochs=epochs, df=df, baseline=(-0.1, 0))
        idx = np.array([0, 1, 2, 3])

        evoked = erp.create_erps(epochs, df, idx=idx, erp_name='x', save=False)

        baseline_idx = epochs.times <= 0
        manual = data.copy()
        for t in range(4):
            manual[t] -= manual[t][:, baseline_idx].mean(axis=1, keepdims=True)
        manual_avg = manual.mean(axis=0)

        np.testing.assert_allclose(evoked.data, manual_avg, atol=1e-8)

    def test_time_oi_crops_after_averaging(self):
        epochs, df, data = self._build_epochs_df()
        erp = ERP(sj=1, epochs=epochs, df=df, baseline=(-0.1, 0))
        idx = np.array([0, 1, 2, 3])

        evoked = erp.create_erps(
            epochs, df, idx=idx, time_oi=(0.0, 0.05), erp_name='x', save=False
        )

        assert evoked.times.min() == pytest.approx(0.0, abs=1e-8)
        assert evoked.times.max() == pytest.approx(0.05, abs=1e-8)

    def test_save_false_does_not_write_files(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        epochs, df, _ = self._build_epochs_df()
        erp = ERP(sj=1, epochs=epochs, df=df, baseline=(-0.1, 0))

        erp.create_erps(epochs, df, idx=np.arange(4), erp_name='x', save=False)

        assert not (tmp_path / 'erp').exists()

    def test_save_true_writes_evoked_file(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        epochs, df, _ = self._build_epochs_df()
        erp = ERP(sj=1, epochs=epochs, df=df, baseline=(-0.1, 0))

        erp.create_erps(epochs, df, idx=np.arange(4), erp_name='x', save=True)

        assert (tmp_path / 'erp' / 'evoked' / 'x-ave.fif').is_file()

    def test_rt_split_writes_fast_and_slow_files(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        epochs, df, _ = self._build_epochs_df()
        erp = ERP(sj=1, epochs=epochs, df=df, baseline=(-0.1, 0))

        erp.create_erps(
            epochs, df, idx=np.arange(4), erp_name='rt', RT_split=True, save=True
        )

        evoked_dir = tmp_path / 'erp' / 'evoked'
        assert (evoked_dir / 'rt-ave.fif').is_file()
        assert (evoked_dir / 'rt_fast-ave.fif').is_file()
        assert (evoked_dir / 'rt_slow-ave.fif').is_file()

        # RT = [100, 900, 200, 800], median = 500 -> trials 0,2 fast; 1,3 slow
        fast = mne.read_evokeds(str(evoked_dir / 'rt_fast-ave.fif'))[0]
        assert fast.nave == 2


# ============================================================================
# condition_erps: end-to-end integration
# ============================================================================

class TestConditionErpsIntegration:
    """Integration tests for the full condition_erps pipeline."""

    @staticmethod
    def _build_epochs_df():
        ch_names = ['Cz', 'Pz']
        sfreq, n_samples, tmin = 100, 20, -0.1
        rng = np.random.default_rng(3)
        data = rng.normal(0, 0.1, size=(8, len(ch_names), n_samples))
        info = mne.create_info(ch_names, sfreq, ch_types='eeg')
        epochs = mne.EpochsArray(data, info, tmin=tmin)
        df = pd.DataFrame({
            'cnd': ['a', 'a', 'a', 'a', 'b', 'b', 'b', 'b'],
        })
        return epochs, df, data

    def test_writes_one_evoked_file_per_condition(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        epochs, df, data = self._build_epochs_df()
        erp = ERP(sj=1, epochs=epochs, df=df, baseline=(-0.1, 0))

        erp.condition_erps(cnds={'cnd': ['a', 'b']}, name='probe')

        evoked_dir = tmp_path / 'erp' / 'evoked'
        assert (evoked_dir / 'sub_01_a_probe-ave.fif').is_file()
        assert (evoked_dir / 'sub_01_b_probe-ave.fif').is_file()

    def test_condition_data_matches_manual_average(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        epochs, df, data = self._build_epochs_df()
        erp = ERP(sj=1, epochs=epochs, df=df, baseline=(-0.1, 0))

        erp.condition_erps(cnds={'cnd': ['a', 'b']}, name='probe')

        ev_a = mne.read_evokeds(
            str(tmp_path / 'erp' / 'evoked' / 'sub_01_a_probe-ave.fif')
        )[0]
        baseline_idx = epochs.times <= 0
        manual_a = data[[0, 1, 2, 3]].copy()
        for t in range(manual_a.shape[0]):
            manual_a[t] -= manual_a[t][:, baseline_idx].mean(axis=1, keepdims=True)
        np.testing.assert_allclose(ev_a.data, manual_a.mean(axis=0), atol=1e-6)

    def test_skips_condition_with_no_matching_trials(self, tmp_path, monkeypatch, capsys):
        monkeypatch.chdir(tmp_path)
        epochs, df, data = self._build_epochs_df()
        erp = ERP(sj=1, epochs=epochs, df=df, baseline=(-0.1, 0))

        erp.condition_erps(cnds={'cnd': ['a', 'missing']}, name='probe')

        evoked_dir = tmp_path / 'erp' / 'evoked'
        assert (evoked_dir / 'sub_01_a_probe-ave.fif').is_file()
        assert not (evoked_dir / 'sub_01_missing_probe-ave.fif').exists()
        assert 'no data found for missing' in capsys.readouterr().out

    def test_pos_labels_restricts_trials(self, tmp_path, monkeypatch):
        """pos_labels selects a subset of trials before condition splitting."""
        monkeypatch.chdir(tmp_path)
        ch_names = ['Cz']
        sfreq, n_samples, tmin = 100, 10, -0.05
        data = np.zeros((4, 1, n_samples))
        info = mne.create_info(ch_names, sfreq, ch_types='eeg')
        epochs = mne.EpochsArray(data, info, tmin=tmin)
        df = pd.DataFrame({'target_loc': [1, 2, 1, 2]})
        erp = ERP(sj=1, epochs=epochs, df=df, baseline=(-0.05, 0))

        erp.condition_erps(pos_labels={'target_loc': [1]}, name='probe')

        ev = mne.read_evokeds(
            str(tmp_path / 'erp' / 'evoked' / 'sub_01_all_data_probe-ave.fif')
        )[0]
        assert ev.nave == 2  # only the two target_loc==1 trials


# ============================================================================
# extract_erp_features
# ============================================================================

class TestExtractERPFeatures:
    """Tests for extract_erp_features method (mean amplitude, AUC, latency)."""

    @pytest.mark.unit
    def test_extract_mean_amplitude(self, sample_waveforms):
        waveforms, times = sample_waveforms
        x = waveforms['absent']

        result = ERP.extract_erp_features(x, times, method='mean_amp')

        assert result.shape == (20,)
        assert np.all(result > -20)
        assert np.all(result < 20)
        assert np.mean(result) < 0

    @pytest.mark.unit
    def test_extract_auc_total(self, sample_waveforms):
        waveforms, times = sample_waveforms
        x = waveforms['absent']

        result = ERP.extract_erp_features(x, times, method='auc')

        assert result.shape == (20,)
        assert np.all(np.isfinite(result))
        assert np.mean(result) < 0

    @pytest.mark.unit
    def test_extract_auc_positive(self, sample_waveforms):
        waveforms, times = sample_waveforms
        x = waveforms['absent']

        result = ERP.extract_erp_features(x, times, method='auc_pos')

        assert result.shape == (20,)
        assert np.all(result >= 0)

    @pytest.mark.unit
    def test_extract_auc_negative(self, sample_waveforms):
        waveforms, times = sample_waveforms
        x = waveforms['absent']

        result = ERP.extract_erp_features(x, times, method='auc_neg')

        assert result.shape == (20,)
        assert np.all(result <= 0)

    @pytest.mark.unit
    def test_extract_onset_latency_pos(self):
        times = np.linspace(0, 0.4, 100)
        x = np.random.randn(5, len(times)) * 0.5
        peak_idx = 50
        x[:, peak_idx - 5:peak_idx + 5] = 5

        result = ERP.extract_erp_features(
            x, times, method='onset_latency', threshold=0.5, polarity='pos'
        )

        assert result.shape == (5,)
        assert np.all(result > 0)
        assert np.all(result < 0.4)
        assert np.all(np.abs(result - 0.2) < 0.1)

    @pytest.mark.unit
    def test_extract_onset_latency_neg(self):
        times = np.linspace(0, 0.4, 100)
        x = np.random.randn(5, len(times)) * 0.5
        peak_idx = 50
        x[:, peak_idx - 5:peak_idx + 5] = -5

        result = ERP.extract_erp_features(
            x, times, method='onset_latency', threshold=0.5, polarity='neg'
        )

        assert result.shape == (5,)
        assert np.all(result > 0)
        assert np.all(result < 0.4)

    @pytest.mark.unit
    def test_extract_features_preserves_input(self, sample_waveforms):
        waveforms, times = sample_waveforms
        x = waveforms['absent'].copy()
        x_orig = x.copy()

        ERP.extract_erp_features(x, times, method='mean_amp')
        ERP.extract_erp_features(x, times, method='auc')
        ERP.extract_erp_features(x, times, method='auc_neg')

        np.testing.assert_array_equal(x, x_orig)

    @pytest.mark.unit
    def test_all_methods_return_one_value_per_trial(self):
        """Every supported method returns shape (n_trials,) given
        appropriate kwargs (onset_latency requires threshold/polarity)."""
        times = np.linspace(0, 0.4, 100)
        x = np.random.randn(5, 100)

        for method in ['mean_amp', 'auc', 'auc_pos', 'auc_neg']:
            result = ERP.extract_erp_features(x, times, method=method)
            assert result.shape == (5,)

        result = ERP.extract_erp_features(
            x, times, method='onset_latency', threshold=0.5, polarity='pos'
        )
        assert result.shape == (5,)


# ============================================================================
# select_erp_window
# ============================================================================

class TestSelectErpWindow:
    """Tests for select_erp_window peak-detection window selection."""

    @pytest.mark.unit
    def test_cnd_avg_centers_on_grand_mean_positive_peak(self, peaked_erps):
        window = ERP.select_erp_window(
            peaked_erps, elec_oi=['Cz', 'Pz'], method='cnd_avg',
            polarity='pos', window_size=0.02
        )

        assert window[2] == 'pos'
        assert window[0] == pytest.approx(0.09, abs=1e-6)
        assert window[1] == pytest.approx(0.11, abs=1e-6)

    @pytest.mark.unit
    def test_cnd_avg_negative_polarity_finds_cond2_dip(self, peaked_erps):
        window = ERP.select_erp_window(
            peaked_erps, elec_oi=['Cz', 'Pz'], method='cnd_avg',
            polarity='neg', window_size=0.02
        )

        assert window[2] == 'neg'
        assert window[0] == pytest.approx(0.19, abs=1e-6)
        assert window[1] == pytest.approx(0.21, abs=1e-6)

    @pytest.mark.unit
    def test_cnd_spc_returns_per_condition_windows(self, peaked_erps):
        windows = ERP.select_erp_window(
            peaked_erps, elec_oi=['Cz', 'Pz'], method='cnd_spc',
            polarity='pos', window_size=0.02
        )

        assert set(windows.keys()) == {'cond1', 'cond2'}
        assert windows['cond1'][0] == pytest.approx(0.09, abs=1e-6)
        # cond2's own positive-polarity peak is noise-driven (its designed
        # spike is negative), so only cond1 is checked precisely here.

    @pytest.mark.unit
    def test_lateralized_elec_oi_computes_difference_wave(self, peaked_erps):
        """Nested elec_oi=[contra, ipsi] computes contra-ipsi before
        peak detection; using the same channel for both should be a
        degenerate all-zero difference wave, so the peak search falls
        back to whatever noise is left (still returns a valid window)."""
        window = ERP.select_erp_window(
            peaked_erps, elec_oi=[['Cz'], ['Pz']], method='cnd_avg',
            polarity='pos', window_size=0.02
        )

        assert len(window) == 3
        assert window[1] - window[0] == pytest.approx(0.02, abs=1e-6)


# ============================================================================
# jackknife_contrast / jack_latency_contrast
# ============================================================================

class TestLatencyAnalysisConditionNames:
    """Tests for jackknife-based latency analysis with condition names."""

    @pytest.mark.unit
    def test_jackknife_contrast_basic(self, sample_waveforms):
        waveforms, times = sample_waveforms
        x1 = waveforms['absent']
        x2 = waveforms['present']

        lat_diff, t_val = ERP.jackknife_contrast(
            x1, x2, times, 75, cnd1_name='absent', cnd2_name='present'
        )

        assert isinstance(lat_diff, (float, np.floating))
        assert isinstance(t_val, (float, np.floating))
        assert -0.1 < lat_diff < 0.1

    @pytest.mark.unit
    def test_jackknife_contrast_condition_names_are_printed(self, sample_waveforms, capsys):
        waveforms, times = sample_waveforms
        ERP.jackknife_contrast(
            waveforms['absent'], waveforms['present'], times, 75,
            cnd1_name='distractor_absent', cnd2_name='distractor_present'
        )

        out = capsys.readouterr().out
        assert 'distractor_absent' in out
        assert 'distractor_present' in out

    @pytest.mark.unit
    def test_jackknife_contrast_default_names(self, sample_waveforms):
        waveforms, times = sample_waveforms
        lat_diff, t_val = ERP.jackknife_contrast(
            waveforms['absent'], waveforms['present'], times, 75
        )

        assert isinstance(lat_diff, (float, np.floating))
        assert isinstance(t_val, (float, np.floating))

    @pytest.mark.unit
    def test_jack_latency_contrast_with_names(self, sample_waveforms):
        waveforms, times = sample_waveforms
        x1 = waveforms['absent'].mean(axis=0)
        x2 = waveforms['present'].mean(axis=0)
        c1 = np.max(x1) * 0.75
        c2 = np.max(x2) * 0.75

        lat_diff = ERP.jack_latency_contrast(
            x1, x2, c1, c2, times, print_output=False,
            cnd1_name='absent', cnd2_name='present'
        )

        assert isinstance(lat_diff, (float, np.floating))


# ============================================================================
# compare_latencies
# ============================================================================

class TestCompareLatencies:
    """Tests for compare_latencies (list form and dict form)."""

    @pytest.mark.unit
    def test_list_form_compares_two_raw_waveforms(self, sample_waveforms):
        waveforms, times = sample_waveforms
        lat_diff, t_val = ERP.compare_latencies(
            [waveforms['absent'], waveforms['present']], times=times, percent_amp=75
        )

        assert isinstance(lat_diff, (float, np.floating))
        assert isinstance(t_val, (float, np.floating))

    @pytest.mark.unit
    def test_dict_form_single_pair_returns_tuple(self, peaked_erps):
        result = ERP.compare_latencies(
            peaked_erps, elec_oi=['Cz', 'Pz'], percent_amp=50, polarity='pos'
        )

        assert isinstance(result, tuple)
        assert len(result) == 2

    @pytest.mark.unit
    def test_dict_form_prints_condition_pair_and_names(self, peaked_erps, capsys):
        ERP.compare_latencies(
            peaked_erps, elec_oi=['Cz', 'Pz'], percent_amp=50, polarity='pos'
        )

        out = capsys.readouterr().out
        assert 'cond1' in out
        assert 'cond2' in out

    @pytest.mark.unit
    def test_dict_form_multi_condition_returns_dict_of_pairs(self, peaked_erps):
        # a 3rd condition with its own peak/noise (not a copy of cond1/cond2 --
        # identical waveforms would trigger the zero-jackknife-variance error)
        from tests.fixtures.sample_data import create_peaked_erps
        cond3 = create_peaked_erps(
            peak_times=(0.15, 0.15), peak_polarities=(1, 1), seed=7
        )['cond1']
        three_cnd = dict(peaked_erps)
        three_cnd['cond3'] = cond3

        result = ERP.compare_latencies(
            three_cnd, elec_oi=['Cz', 'Pz'], percent_amp=50, polarity='pos'
        )

        assert isinstance(result, dict)
        assert set(result.keys()) == {
            'cond1_cond2', 'cond1_cond3', 'cond2_cond3'
        }
        for pair_result in result.values():
            assert len(pair_result) == 2


# ============================================================================
# Group-level ERP analysis
# ============================================================================

class TestGroupERPAnalysis:
    """Tests for group-level ERP analysis methods."""

    @pytest.mark.unit
    def test_lateralized_erp_idx_basic(self, sample_epochs):
        evoked = sample_epochs.average()
        erp_list = [evoked]
        channels = evoked.ch_names
        contra_channels = channels[5:8]
        ipsi_channels = channels[10:13]

        contra_idx, ipsi_idx = ERP.lateralized_erp_idx(
            erp_list, contra_channels, ipsi_channels
        )

        assert isinstance(contra_idx, np.ndarray)
        assert isinstance(ipsi_idx, np.ndarray)
        assert len(contra_idx) == 3
        assert len(ipsi_idx) == 3
        assert np.all(contra_idx >= 0)
        assert np.all(contra_idx < len(channels))
        assert not np.array_equal(contra_idx, ipsi_idx)

    @pytest.mark.unit
    def test_lateralized_erp_idx_mapping(self, sample_epochs):
        evoked = sample_epochs.average()
        erp_list = [evoked]
        channels = evoked.ch_names
        contra_channels = [channels[0], channels[5]]
        ipsi_channels = [channels[10], channels[15]]

        contra_idx, ipsi_idx = ERP.lateralized_erp_idx(
            erp_list, contra_channels, ipsi_channels
        )

        assert channels[contra_idx[0]] == channels[0]
        assert channels[contra_idx[1]] == channels[5]
        assert channels[ipsi_idx[0]] == channels[10]
        assert channels[ipsi_idx[1]] == channels[15]

    @pytest.mark.unit
    def test_group_erp_all_electrodes(self, sample_epochs):
        evoked1 = sample_epochs[:50].average()
        evoked2 = sample_epochs[50:].average()
        erp_list = [evoked1, evoked2]

        evoked_X, group_evoked = ERP.group_erp(erp_list, elec_oi='all')

        assert isinstance(evoked_X, np.ndarray)
        assert isinstance(group_evoked, mne.Evoked)
        assert evoked_X.shape[0] == 2
        assert evoked_X.shape[1] == len(evoked1.times)
        assert len(group_evoked.times) == len(evoked1.times)

    @pytest.mark.unit
    def test_group_erp_specific_electrodes(self, sample_epochs):
        evoked1 = sample_epochs[:50].average()
        evoked2 = sample_epochs[50:].average()
        erp_list = [evoked1, evoked2]
        elec_oi = evoked1.ch_names[:3]

        evoked_X, group_evoked = ERP.group_erp(erp_list, elec_oi=elec_oi)

        assert evoked_X.shape[0] == 2
        assert evoked_X.shape[1] == len(evoked1.times)
        assert isinstance(group_evoked, mne.Evoked)

    @pytest.mark.unit
    def test_select_waveform_basic(self, sample_epochs):
        evoked = sample_epochs.average()
        erp_list = [evoked]
        channels = evoked.ch_names[:5]

        waveform = ERP.select_waveform(erp_list, channels)

        assert isinstance(waveform, np.ndarray)
        assert waveform.shape[0] == 1
        assert waveform.shape[1] == len(evoked.times)

    @pytest.mark.unit
    def test_select_waveform_single_channel(self, sample_epochs):
        evoked = sample_epochs.average()
        erp_list = [evoked]
        channel = [evoked.ch_names[0]]

        waveform = ERP.select_waveform(erp_list, channel)

        assert waveform.shape[0] == 1
        assert waveform.shape[1] == len(evoked.times)

    @pytest.mark.unit
    def test_group_lateralized_erp_difference_and_topography(self, biosemi64_evoked_pair):
        """Uses a real biosemi64 montage so all required electrode pairs
        exist (the previous test using standard_1020 channel names skipped
        every run because it never matched the montage)."""
        diff, evoked_lat = ERP.group_lateralized_erp(
            biosemi64_evoked_pair, ['O1', 'P7'], ['O2', 'P8']
        )

        # O1-O2 = 5-2 = 3, P7-P8 = 3-1 = 2, mean = 2.5
        np.testing.assert_allclose(diff, 2.5)

        ch_names = evoked_lat.ch_names
        assert evoked_lat.data[ch_names.index('O1'), 0] == pytest.approx(3.0)
        assert evoked_lat.data[ch_names.index('O2'), 0] == pytest.approx(-3.0)
        assert evoked_lat.data[ch_names.index('Cz'), 0] == pytest.approx(0.0)


class TestSelectWaveformVariations:
    """Tests for waveform selection with different input formats."""

    @pytest.mark.unit
    def test_select_waveform_multiple_evoked(self, sample_epochs):
        erp1 = sample_epochs[:30].average()
        erp2 = sample_epochs[30:60].average()
        erp3 = sample_epochs[60:].average()
        erp_list = [erp1, erp2, erp3]
        channels = erp1.ch_names[:4]

        waveform = ERP.select_waveform(erp_list, channels)

        assert waveform.shape[0] == 3
        assert waveform.shape[1] == len(erp1.times)


class TestGetERPParams:
    """Tests for get_erp_params utility."""

    @pytest.mark.unit
    def test_get_erp_params_evoked_list_input(self, sample_epochs):
        erp1 = sample_epochs[:50].average()
        erp2 = sample_epochs[50:].average()

        result = ERP.get_erp_params([erp1, erp2])

        assert isinstance(result, tuple)
        assert len(result) == 2

    @pytest.mark.unit
    def test_get_erp_params_dict_input_with_evoked(self, sample_epochs):
        erp1 = sample_epochs[:50].average()
        erp2 = sample_epochs[50:].average()
        erp_dict = {'condition1': [erp1], 'condition2': [erp2]}

        result = ERP.get_erp_params(erp_dict)

        assert isinstance(result, tuple)
        assert len(result) == 2


# ============================================================================
# export_erp_metrics_to_csv
# ============================================================================

class TestExportErpMetricsToCsv:
    """Tests for export_erp_metrics_to_csv output."""

    @pytest.mark.unit
    def test_writes_one_column_per_condition(self, peaked_erps, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)

        ERP.export_erp_metrics_to_csv(
            peaked_erps, window_oi=[0.05, 0.15], elec_oi=['Cz', 'Pz'],
            method='mean_amp', name='probe_metrics'
        )

        path = tmp_path / 'erp' / 'stats' / 'probe_metrics.csv'
        assert path.is_file()

        df = pd.read_csv(path)
        assert list(df.columns) == ['cond1', 'cond2']
        assert len(df) == 5  # n_subjects in the peaked_erps fixture

    @pytest.mark.unit
    def test_lateralized_elec_oi_writes_contra_ipsi_diff_columns(
        self, peaked_erps, tmp_path, monkeypatch
    ):
        monkeypatch.chdir(tmp_path)

        ERP.export_erp_metrics_to_csv(
            peaked_erps, window_oi=[0.05, 0.15], elec_oi=[['Cz'], ['Pz']],
            method='mean_amp', name='probe_metrics_lat'
        )

        path = tmp_path / 'erp' / 'stats' / 'probe_metrics_lat.csv'
        df = pd.read_csv(path)
        assert list(df.columns) == [
            'cond1_contra', 'cond1_ipsi', 'cond1_diff',
            'cond2_contra', 'cond2_ipsi', 'cond2_diff',
        ]

    @pytest.mark.unit
    def test_rejects_invalid_erps_type(self):
        with pytest.raises(ValueError, match="erps must be"):
            ERP.export_erp_metrics_to_csv(
                "not-a-dict-or-list", window_oi=[0, 1], elec_oi=['Cz']
            )

    @pytest.mark.unit
    def test_rejects_invalid_window_oi_type(self, peaked_erps):
        with pytest.raises(ValueError, match="window_oi must be"):
            ERP.export_erp_metrics_to_csv(
                peaked_erps, window_oi="not-a-list-or-dict", elec_oi=['Cz', 'Pz']
            )


# ============================================================================
# residual_eye
# ============================================================================

class TestResidualEye:
    """Tests for residual_eye HEOG computation."""

    @pytest.mark.unit
    def test_computes_documented_residual_formula(
        self, residual_eye_data, tmp_path, monkeypatch
    ):
        """residual = mean(mean(left HEOG), -mean(right HEOG)), per the
        method's own docstring. Note the source carries an unresolved
        `#TODO: check whether function is correct` -- this test only
        confirms the implementation matches its documented formula, not
        that the formula is the right scientific choice."""
        monkeypatch.chdir(tmp_path)
        epochs, trial_info, expected = residual_eye_data

        erp = ERP(sj=1, epochs=epochs, df=trial_info, baseline=(-0.1, 0))
        erp.residual_eye(
            left_info={'target_loc': [1]}, right_info={'target_loc': [2]},
            ch_oi=['HEOG'], name='probe_resid'
        )

        path = tmp_path / 'erp' / 'eog' / 'sub_01_probe_resid.p'
        assert path.is_file()

        import pickle
        with open(path, 'rb') as f:
            result = pickle.load(f)

        assert set(result.keys()) == {'all_data'}
        np.testing.assert_allclose(result['all_data'], expected)


# ============================================================================
# generate_erp_report
# ============================================================================

class TestGenerateErpReport:
    """Smoke test for HTML report generation."""

    @pytest.mark.unit
    def test_writes_html_report_file(self, sample_epochs, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        evoked = sample_epochs.average()
        erp = ERP(
            sj=1, epochs=sample_epochs, df=pd.DataFrame({'x': range(len(sample_epochs))}),
            baseline=(None, 0), report=True
        )

        erp.generate_erp_report({'all_data': evoked}, 'probe_report')

        assert (tmp_path / 'erp' / 'report' / 'probe_report.html').is_file()


# ============================================================================
# ERP component analysis (realistic scenarios)
# ============================================================================

class TestERPComponentAnalysis:
    """Tests for realistic ERP component analysis scenarios."""

    @pytest.mark.unit
    def test_n2pc_like_analysis(self):
        times = np.linspace(-0.2, 0.4, 300)
        n_trials = 30
        x = np.random.randn(n_trials, len(times)) * 0.5
        peak_idx = 200
        x[:, peak_idx - 20:peak_idx + 20] = -3

        mean_amp = ERP.extract_erp_features(x, times, method='mean_amp')
        auc_neg = ERP.extract_erp_features(x, times, method='auc_neg')

        assert np.mean(mean_amp) < 0
        assert np.mean(auc_neg) < 0

    @pytest.mark.unit
    def test_p300_like_analysis(self):
        times = np.linspace(-0.2, 0.8, 600)
        n_trials = 20
        x = np.random.randn(n_trials, len(times)) * 0.5
        peak_idx = 300
        x[:, peak_idx - 40:peak_idx + 40] = 5

        mean_amp = ERP.extract_erp_features(x, times, method='mean_amp')
        auc_pos = ERP.extract_erp_features(x, times, method='auc_pos')

        assert np.mean(mean_amp) > 0
        assert np.mean(auc_pos) > 0

    @pytest.mark.unit
    def test_condition_comparison_workflow(self, sample_waveforms):
        waveforms, times = sample_waveforms

        features = {}
        for condition, waveform in waveforms.items():
            features[condition] = {
                'mean_amp': ERP.extract_erp_features(waveform, times, method='mean_amp'),
                'auc': ERP.extract_erp_features(waveform, times, method='auc'),
            }

        assert 'absent' in features
        assert 'present' in features
        mean_diff = (
            np.mean(features['absent']['mean_amp'])
            - np.mean(features['present']['mean_amp'])
        )
        assert isinstance(mean_diff, (float, np.floating))


class TestWaveformAnalysisCombinations:
    """Tests for common multi-condition feature-comparison workflows."""

    @pytest.mark.unit
    def test_multi_condition_feature_comparison(self, sample_waveforms):
        waveforms, times = sample_waveforms
        conditions = {'absent': waveforms['absent'], 'present': waveforms['present']}

        features = {}
        for cond_name, waveform in conditions.items():
            features[cond_name] = {
                'mean_amp': np.mean(
                    ERP.extract_erp_features(waveform, times, method='mean_amp')
                ),
                'auc': np.mean(
                    ERP.extract_erp_features(waveform, times, method='auc')
                ),
            }

        amp_diff = features['absent']['mean_amp'] - features['present']['mean_amp']
        assert isinstance(amp_diff, (float, np.floating))

    @pytest.mark.unit
    def test_individual_subject_erp_pipeline(self, sample_waveforms):
        waveforms, times = sample_waveforms
        subject_erps = {}
        for cond in ['absent', 'present']:
            waveform = waveforms[cond]
            subject_erps[cond] = {
                'mean_amplitude': np.mean(
                    ERP.extract_erp_features(waveform, times, method='mean_amp')
                ),
                'latency': np.mean(
                    ERP.extract_erp_features(
                        waveform, times, method='onset_latency', polarity='neg'
                    )
                ),
                'auc': np.mean(ERP.extract_erp_features(waveform, times, method='auc')),
            }

        for cond in subject_erps:
            assert all(
                metric in subject_erps[cond]
                for metric in ['mean_amplitude', 'latency', 'auc']
            )


# ============================================================================
# Edge cases and error handling
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases and error conditions."""

    @pytest.mark.unit
    def test_select_lateralization_idx_empty_trial_info(self):
        empty_trial_info = pd.DataFrame({
            'trial': [], 'dist1_loc': [], 'dist2_loc': [],
        })

        idx = ERP.select_lateralization_idx(
            empty_trial_info, pos_labels={'dist1_loc': [0]}
        )

        assert len(idx) == 0

    @pytest.mark.unit
    def test_jackknife_contrast_identical_waveforms_raises(self):
        """Identical waveforms give zero jackknife variance -> ValueError."""
        times = np.linspace(0, 0.4, 100)
        x = np.random.randn(20, len(times))

        with pytest.raises(ValueError, match="Standard deviation is zero"):
            ERP.jackknife_contrast(x, x, times, 75)

    @pytest.mark.unit
    def test_jackknife_contrast_single_trial_raises(self):
        """Jackknife needs >=2 observations per condition to leave one out."""
        times = np.linspace(0, 0.4, 100)
        x1 = np.random.randn(1, len(times))
        x2 = np.random.randn(1, len(times))

        with pytest.raises(ValueError, match="at least 2 observations"):
            ERP.jackknife_contrast(x1, x2, times, 75)


# ============================================================================
# Regression tests
# ============================================================================

class TestRegressions:
    """Regression tests to ensure fixes don't break existing functionality."""

    @pytest.mark.unit
    def test_single_key_restriction_unchanged(self):
        data, trial_info = create_lateralization_test_data()

        idx = ERP.select_lateralization_idx(
            trial_info, pos_labels={'dist1_loc': [0]}
        )

        assert len(idx) == 4
        np.testing.assert_array_equal(idx, [0, 1, 2, 3])

    @pytest.mark.unit
    def test_latency_analysis_returns_correct_types(self, sample_waveforms):
        waveforms, times = sample_waveforms
        lat_diff, t_val = ERP.jackknife_contrast(
            waveforms['absent'], waveforms['present'], times, 75
        )

        assert isinstance(lat_diff, (float, np.floating))
        assert isinstance(t_val, (float, np.floating))


# ============================================================================
# Data validation and error handling
# ============================================================================

class TestERPDataValidation:
    """Tests for data validation and error handling in ERP methods."""

    @pytest.mark.unit
    def test_extract_features_invalid_method_raises(self, sample_waveforms):
        waveforms, times = sample_waveforms
        x = waveforms['absent']

        with pytest.raises(ValueError, match="Unknown method"):
            ERP.extract_erp_features(x, times, method='invalid_method')

    @pytest.mark.unit
    def test_mean_amp_ignores_times_array_entirely(self, sample_waveforms):
        """mean_amp only touches X, so a wrong-length times array is
        silently accepted -- this documents that behavior rather than
        pretending it errors."""
        waveforms, times = sample_waveforms
        x = waveforms['absent']
        wrong_times = np.linspace(0, 0.2, 50)

        result = ERP.extract_erp_features(x, wrong_times, method='mean_amp')

        assert result.shape == (20,)

    @pytest.mark.unit
    def test_auc_raises_on_mismatched_times_length(self, sample_waveforms):
        waveforms, times = sample_waveforms
        x = waveforms['absent']
        wrong_times = np.linspace(0, 0.2, 50)

        with pytest.raises(ValueError, match="inconsistent numbers of samples"):
            ERP.extract_erp_features(x, wrong_times, method='auc')

    @pytest.mark.unit
    def test_mean_amplitude_valid_output_range(self, sample_waveforms):
        waveforms, times = sample_waveforms
        x = waveforms['absent']

        result = ERP.extract_erp_features(x, times, method='mean_amp')

        assert np.all(result > -100)
        assert np.all(result < 100)

    @pytest.mark.unit
    def test_onset_latency_within_time_range(self, sample_waveforms):
        times = np.linspace(0, 0.4, 100)
        x = np.random.randn(5, len(times))
        peak_idx = 50
        x[:, peak_idx - 5:peak_idx + 5] = 5

        result = ERP.extract_erp_features(
            x, times, method='onset_latency', threshold=0.5, polarity='pos'
        )

        assert np.all(result >= times.min())
        assert np.all(result <= times.max())


# ============================================================================
# Jackknife statistical properties
# ============================================================================

class TestStatisticalComparisons:
    """Tests for statistical properties of jackknife_contrast."""

    @pytest.mark.unit
    def test_jackknife_contrast_produces_t_value(self, sample_waveforms):
        waveforms, times = sample_waveforms
        lat_diff, t_val = ERP.jackknife_contrast(
            waveforms['absent'], waveforms['present'], times, 75
        )

        assert np.isfinite(t_val)
        assert t_val != 0

    @pytest.mark.unit
    def test_jackknife_variance_estimation(self, sample_waveforms):
        waveforms, times = sample_waveforms
        x1 = waveforms['absent']
        x2 = waveforms['present'] + 0.01

        lat_diff, t_val = ERP.jackknife_contrast(x1, x2, times, 75)

        assert abs(t_val) < 10


# ============================================================================
# Advanced validation and errors
# ============================================================================

class TestERPValidationAndErrors:
    """Tests for validation and error handling in statistical methods."""

    @pytest.mark.unit
    def test_jackknife_minimum_observations(self):
        times = np.linspace(0, 0.4, 100)
        x1 = np.random.randn(2, len(times))
        x2 = np.random.randn(2, len(times))

        lat_diff, t_val = ERP.jackknife_contrast(x1, x2, times, 75)

        assert np.isfinite(lat_diff)
        assert np.isfinite(t_val)

    @pytest.mark.unit
    def test_feature_extraction_null_signal(self):
        times = np.linspace(0, 0.4, 100)
        x = np.zeros((5, len(times)))

        mean_amp = ERP.extract_erp_features(x, times, method='mean_amp')

        np.testing.assert_array_almost_equal(mean_amp, np.zeros(5))

    @pytest.mark.unit
    def test_feature_extraction_constant_signal(self):
        times = np.linspace(0, 0.4, 100)
        x = np.ones((5, len(times))) * 5

        mean_amp = ERP.extract_erp_features(x, times, method='mean_amp')

        np.testing.assert_array_almost_equal(mean_amp, np.ones(5) * 5)

    @pytest.mark.unit
    def test_auc_symmetry(self):
        times = np.linspace(0, 0.4, 100)
        x_pos = np.ones((1, len(times))) * 5
        x_neg = np.ones((1, len(times))) * -5

        auc_pos = ERP.extract_erp_features(x_pos, times, method='auc_pos')[0]
        auc_neg = ERP.extract_erp_features(x_neg, times, method='auc_neg')[0]

        assert abs(abs(auc_pos) - abs(auc_neg)) < 0.1


class TestERPDataInjectionAndValidation:
    """Tests for robustness to noisy/small-sample input."""

    @pytest.mark.unit
    def test_robust_to_noisy_data(self):
        times = np.linspace(0, 0.4, 100)
        x = np.random.randn(20, len(times)) * 5

        result = ERP.extract_erp_features(x, times, method='mean_amp')

        assert result is not None
        assert len(result) == 20

    @pytest.mark.unit
    def test_small_sample_handling(self):
        times = np.linspace(0, 0.4, 100)
        x_small = np.random.randn(1, len(times))

        result = ERP.extract_erp_features(x_small, times, method='mean_amp')

        assert result is not None
        assert len(result) == 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
