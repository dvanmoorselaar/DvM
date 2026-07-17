"""
Test suite for open_dvm.analysis.CTF.

Organization
------------
- TestCalculateBasisSet: CTF.calculate_basis_set
- TestSelectCtfData / TestCheckCndsInput / TestSelectCtfLabels /
  TestSelectBinsOi / TestSetMaxTrial: data/label selection helpers
- TestExtractPower / TestTfrDecomposition: signal processing
- TestForwardModel / TestForwardModelLoop: the IEM core
- TestTrainTestSplit / TestTrainTestCross: cross-validation partitioning
- TestSetFrequencies: frequency-spec parsing
- TestExtractSlopes / TestExtractCtfParams / TestSummarizeCtfs /
  TestGetCtfTuningParams: parameter-extraction pipeline
- TestFitCosToCtf / TestGaussian / TestFitGaussian: curve fitting
- TestSpatialCtf: end-to-end pipeline (within-cnd, cross-cnd,
  special_loc, permutation testing, GAT)
- TestGenerateCtfReport: HTML report generation
- TestRegressions: regression tests for bugs found & fixed during review

Note: localizer_spatial_ctf is intentionally NOT tested -- it is
deferred (not part of this release, per explicit decision) and its
CTF.__init__ prerequisites (list-based epochs/df) are not supported.
"""

import warnings
from math import pi

import numpy as np
import pandas as pd
import pytest
import mne

from open_dvm.analysis.CTF import CTF
from open_dvm.support.preprocessing_utils import get_time_slice

from tests.fixtures.ctf_sample_data import (
    make_spatial_epochs,
    make_localizer_and_ping,
)


def make_ctf(epochs, df, **kwargs):
    defaults = dict(
        sj=1, to_decode='position', nr_bins=8, nr_chans=8,
        downsample=100, nr_iter=2, nr_folds=3, seed=1,
    )
    defaults.update(kwargs)
    return CTF(epochs=epochs, df=df, **defaults)


# ============================================================================
# CTF.calculate_basis_set
# ============================================================================

class TestCalculateBasisSet:
    @pytest.mark.unit
    def test_shape_and_peak_per_channel(self):
        epochs, df = make_spatial_epochs(nr_bins=8, n_trials_per_bin=2)
        ctf = make_ctf(epochs, df)
        basis = ctf.calculate_basis_set(nr_bins=8, nr_chans=8, sin_power=7,
                                         delta=False)
        assert basis.shape == (8, 8)
        np.testing.assert_array_equal(np.argmax(basis, axis=1), np.arange(8))

    @pytest.mark.unit
    def test_exact_values_even_bins(self):
        epochs, df = make_spatial_epochs(nr_bins=8, n_trials_per_bin=2)
        ctf = make_ctf(epochs, df)
        basis = ctf.calculate_basis_set(nr_bins=8, nr_chans=8, sin_power=7,
                                         delta=False)
        expected_row0 = np.array([1., 0.5745230025, 0.0883883476,
                                   0.0012019257, 0., 0.0012019257,
                                   0.0883883476, 0.5745230025])
        np.testing.assert_allclose(basis[0], expected_row0, atol=1e-9)

    @pytest.mark.unit
    def test_rows_are_circular_shifts_of_each_other(self):
        epochs, df = make_spatial_epochs(nr_bins=8, n_trials_per_bin=2)
        ctf = make_ctf(epochs, df)
        basis = ctf.calculate_basis_set(nr_bins=8, nr_chans=8, sin_power=7,
                                         delta=False)
        for c in range(7):
            np.testing.assert_allclose(basis[c + 1], np.roll(basis[c], 1))

    @pytest.mark.unit
    def test_odd_nr_bins_symmetric_and_peaked(self):
        epochs, df = make_spatial_epochs(nr_bins=7, n_trials_per_bin=2)
        ctf = make_ctf(epochs, df, nr_bins=7, nr_chans=7)
        basis = ctf.calculate_basis_set(nr_bins=7, nr_chans=7, sin_power=7,
                                         delta=False)
        np.testing.assert_array_equal(np.argmax(basis, axis=1), np.arange(7))
        row0 = basis[0]
        for k in range(7):
            assert row0[k] == pytest.approx(row0[(-k) % 7])

    @pytest.mark.unit
    def test_delta_basis_is_identity(self):
        epochs, df = make_spatial_epochs(nr_bins=6, n_trials_per_bin=2)
        ctf = make_ctf(epochs, df, nr_bins=6, nr_chans=6)
        basis = ctf.calculate_basis_set(nr_bins=6, nr_chans=6, delta=True)
        np.testing.assert_array_equal(basis, np.eye(6))


# ============================================================================
# CTF.select_ctf_data
# ============================================================================

class TestSelectCtfData:
    @pytest.mark.unit
    def test_basic_no_exclusion_all_electrodes(self):
        epochs, df = make_spatial_epochs(nr_bins=4, n_trials_per_bin=10, n_ch=6)
        ctf = make_ctf(epochs, df, nr_bins=4, nr_chans=4)
        e2, d2 = ctf.select_ctf_data(epochs.copy(), df.copy(), elec_oi='all',
                                      headers=[None])
        assert len(e2) == 40
        assert len(e2.ch_names) == 6

    @pytest.mark.unit
    def test_explicit_electrode_subset(self):
        epochs, df = make_spatial_epochs(nr_bins=4, n_trials_per_bin=10, n_ch=6)
        ctf = make_ctf(epochs, df, nr_bins=4, nr_chans=4)
        e3, d3 = ctf.select_ctf_data(epochs.copy(), df.copy(),
                                      elec_oi=['Ch1', 'Ch3'], headers=[None])
        assert e3.ch_names == ['Ch1', 'Ch3']

    @pytest.mark.unit
    def test_excl_factor_removes_matching_trials(self):
        epochs, df = make_spatial_epochs(nr_bins=4, n_trials_per_bin=10, n_ch=4)
        ctf = make_ctf(epochs, df, nr_bins=4, nr_chans=4)
        e4, d4 = ctf.select_ctf_data(epochs.copy(), df.copy(), elec_oi='all',
                                      headers=[None],
                                      excl_factor={'position': [0]})
        assert len(e4) == 30
        assert set(d4['position'].unique()) == {1, 2, 3}

    @pytest.mark.unit
    def test_broadband_filter_applied_when_specified(self):
        epochs, df = make_spatial_epochs(nr_bins=2, n_trials_per_bin=5, n_ch=4,
                                          n_samples=500, sfreq=200)
        ctf = make_ctf(epochs, df, nr_bins=2, nr_chans=2, filter=10)
        e5, d5 = ctf.select_ctf_data(epochs.copy(), df.copy(), elec_oi='all',
                                      headers=[None], data_type='broadband')
        assert e5.info['lowpass'] == 10.0

    @pytest.mark.unit
    def test_no_filter_when_data_type_not_broadband(self):
        epochs, df = make_spatial_epochs(nr_bins=2, n_trials_per_bin=5, n_ch=4,
                                          n_samples=500, sfreq=200)
        ctf = make_ctf(epochs, df, nr_bins=2, nr_chans=2, filter=10)
        e6, d6 = ctf.select_ctf_data(epochs.copy(), df.copy(), elec_oi='all',
                                      headers=[None], data_type='power')
        assert e6.info['lowpass'] != 10.0


# ============================================================================
# CTF.check_cnds_input
# ============================================================================

class TestCheckCndsInput:
    @pytest.mark.unit
    def test_within_condition(self):
        epochs, df = make_spatial_epochs(nr_bins=2, n_trials_per_bin=2)
        ctf = make_ctf(epochs, df, nr_bins=2, nr_chans=2)
        train, test = ctf.check_cnds_input({'task': ['a', 'b']})
        assert train == ['a', 'b']
        assert test is None

    @pytest.mark.unit
    def test_cross_condition_single_test(self):
        epochs, df = make_spatial_epochs(nr_bins=2, n_trials_per_bin=2)
        ctf = make_ctf(epochs, df, nr_bins=2, nr_chans=2)
        train, test = ctf.check_cnds_input({'task': [['a'], ['b']]})
        assert train == ['a']
        assert test == ['b']

    @pytest.mark.unit
    def test_cross_condition_multi_test(self):
        epochs, df = make_spatial_epochs(nr_bins=2, n_trials_per_bin=2)
        ctf = make_ctf(epochs, df, nr_bins=2, nr_chans=2)
        train, test = ctf.check_cnds_input({'task': [['a', 'c'], ['b', 'd']]})
        assert train == ['a', 'c']
        assert test == ['b', 'd']

    @pytest.mark.unit
    def test_cross_condition_bare_string_test_still_works(self):
        """
        Regression test: a bare string for a single test condition
        (matching the module's own former docstring example) used to
        be iterated character-by-character downstream instead of
        being treated as one condition name.
        """
        epochs, df = make_spatial_epochs(nr_bins=2, n_trials_per_bin=2)
        ctf = make_ctf(epochs, df, nr_bins=2, nr_chans=2)
        train, test = ctf.check_cnds_input({'task': [['a'], 'b']})
        assert train == ['a']
        assert test == ['b']


# ============================================================================
# CTF.select_ctf_labels / select_bins_oi / set_max_trial
# ============================================================================

class TestSelectCtfLabels:
    @pytest.mark.unit
    def test_all_positions_no_conditions(self):
        epochs, df = make_spatial_epochs(nr_bins=4, n_trials_per_bin=10, n_ch=4)
        ctf = make_ctf(epochs, df, nr_bins=4, nr_chans=4, nr_folds=2)
        pos_bins, cnds, ep, max_tr = ctf.select_ctf_labels(
            epochs.copy(), df.copy(), 'all', None)
        assert len(ep) == 40
        np.testing.assert_array_equal(np.unique(pos_bins), [0, 1, 2, 3])
        assert max_tr == 5  # floor(10 trials-per-bin / 2 folds)

    @pytest.mark.unit
    def test_pos_labels_restricts_subset(self):
        epochs, df = make_spatial_epochs(nr_bins=4, n_trials_per_bin=10, n_ch=4)
        ctf = make_ctf(epochs, df, nr_bins=4, nr_chans=4)
        pos_bins, cnds, ep, max_tr = ctf.select_ctf_labels(
            epochs.copy(), df.copy(), {'position': [0, 1]}, None)
        np.testing.assert_array_equal(np.unique(pos_bins), [0, 1])
        assert len(ep) == 20

    @pytest.mark.unit
    def test_cnds_filters_by_condition_column(self):
        epochs, df = make_spatial_epochs(nr_bins=4, n_trials_per_bin=10, n_ch=4)
        df['task'] = ['a'] * 20 + ['b'] * 20
        ctf = make_ctf(epochs, df, nr_bins=4, nr_chans=4)
        pos_bins, cnds, ep, max_tr = ctf.select_ctf_labels(
            epochs.copy(), df.copy(), 'all', {'task': ['a', 'b']})
        assert len(ep) == 40
        np.testing.assert_array_equal(np.unique(cnds), ['a', 'b'])


class TestSelectBinsOi:
    @pytest.mark.unit
    def test_removes_out_of_range_bins(self):
        epochs, df = make_spatial_epochs(nr_bins=4, n_trials_per_bin=2, n_ch=4)
        ctf = make_ctf(epochs, df, nr_bins=4, nr_chans=4)
        pos_bins = np.array([0, 1, 2, 3, 4, -1, 2])
        cnds = np.array(['x'] * 7)
        ep = epochs.copy()[:7]
        pb, cn, epo = ctf.select_bins_oi(pos_bins, cnds, ep)
        np.testing.assert_array_equal(pb, [0, 1, 2, 3, 2])
        assert len(epo) == 5


class TestSetMaxTrial:
    @pytest.mark.unit
    def test_foster_method_floor_division(self):
        epochs, df = make_spatial_epochs(nr_bins=3, n_trials_per_bin=2, n_ch=4)
        ctf = make_ctf(epochs, df, nr_bins=3, nr_chans=3, nr_folds=3)
        pos_bins = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2])
        cnds = np.array(['x'] * 13)
        # min bin count = 3 (bin 1), nr_folds=3 -> floor(3/3) = 1
        assert ctf.set_max_trial(cnds, pos_bins, 'Foster') == 1

    @pytest.mark.unit
    def test_multiple_conditions_uses_min_across_conditions(self):
        epochs, df = make_spatial_epochs(nr_bins=2, n_trials_per_bin=2, n_ch=4)
        ctf = make_ctf(epochs, df, nr_bins=2, nr_chans=2, nr_folds=2)
        pos_bins = np.array([0, 0, 0, 0, 1, 1, 1, 1,
                              0, 0, 1, 1])
        cnds = np.array(['a'] * 8 + ['b'] * 4)
        # cnd 'a': bin0=4,bin1=4; cnd 'b': bin0=2,bin1=2 -> min=2, floor(2/2)=1
        assert ctf.set_max_trial(cnds, pos_bins, 'Foster') == 1


# ============================================================================
# CTF.extract_power
# ============================================================================

class TestExtractPower:
    @pytest.mark.unit
    def test_filtered_band_returns_squared_magnitude(self):
        epochs, df = make_spatial_epochs(nr_bins=2, n_trials_per_bin=2, n_ch=4)
        ctf = make_ctf(epochs, df, nr_bins=2, nr_chans=2)
        x = np.array([3.0 + 4.0j, 1.0 + 0j])
        np.testing.assert_allclose(ctf.extract_power(x, (8, 12)), [25.0, 1.0])

    @pytest.mark.unit
    def test_none_or_string_band_returns_unchanged(self):
        epochs, df = make_spatial_epochs(nr_bins=2, n_trials_per_bin=2, n_ch=4)
        ctf = make_ctf(epochs, df, nr_bins=2, nr_chans=2)
        x = np.array([3.0 + 4.0j, 1.0 + 0j])
        np.testing.assert_array_equal(ctf.extract_power(x, None), x)
        np.testing.assert_array_equal(ctf.extract_power(x, 'alpha'), x)

    @pytest.mark.unit
    def test_broadband_returns_hilbert_envelope(self):
        epochs, df = make_spatial_epochs(nr_bins=2, n_trials_per_bin=2, n_ch=4)
        ctf = make_ctf(epochs, df, nr_bins=2, nr_chans=2)
        t = np.linspace(0, 1, 200, endpoint=False)
        sig = np.sin(2 * np.pi * 5 * t)
        env = ctf.extract_power(sig, 'broadband')
        # a pure sinusoid's envelope should hover near its amplitude (1),
        # away from the edge-effect region
        np.testing.assert_allclose(env[50:150], 1.0, atol=0.05)


# ============================================================================
# CTF.tfr_decomposition
# ============================================================================

class TestTfrDecomposition:
    @pytest.mark.unit
    def test_broadband_returns_identical_baseline_corrected_voltages(self):
        epochs, df = make_spatial_epochs(nr_bins=2, n_trials_per_bin=5, n_ch=4,
                                          n_samples=100, sfreq=200)
        ctf = make_ctf(epochs, df, nr_bins=2, nr_chans=2)
        tois = get_time_slice(epochs.times, None, None)

        E, T = ctf.tfr_decomposition(epochs.copy(), 'broadband', tois, 1)
        assert np.array_equal(E, T)
        assert E.shape == (10, 4, 100)

    @pytest.mark.unit
    def test_broadband_downsampling_matches_manual_stride(self):
        epochs, df = make_spatial_epochs(nr_bins=2, n_trials_per_bin=5, n_ch=4,
                                          n_samples=100, sfreq=200)
        ctf = make_ctf(epochs, df, nr_bins=2, nr_chans=2)
        tois = get_time_slice(epochs.times, None, None)

        E, T = ctf.tfr_decomposition(epochs.copy(), 'broadband', tois, 4)
        assert E.shape[-1] == 25

        raw = epochs.copy()
        raw.apply_baseline(baseline=None)
        expected = raw._data[:, :, tois][:, :, ::4]
        np.testing.assert_allclose(E, expected)

    @pytest.mark.unit
    def test_filtered_band_returns_complex_evoked_and_real_power(self):
        epochs, df = make_spatial_epochs(nr_bins=2, n_trials_per_bin=5, n_ch=4,
                                          n_samples=500, sfreq=200)
        ctf = make_ctf(epochs, df, nr_bins=2, nr_chans=2)
        tois = get_time_slice(epochs.times, None, None)

        E, T = ctf.tfr_decomposition(epochs.copy(), (4, 8), tois, 1)
        assert np.iscomplexobj(E)
        assert not np.iscomplexobj(T)
        assert (T >= 0).all()
        np.testing.assert_allclose(T, np.abs(E) ** 2)


# ============================================================================
# CTF.forward_model
# ============================================================================

class TestForwardModel:
    @pytest.mark.unit
    def test_identity_system_solves_exactly(self):
        epochs, df = make_spatial_epochs(nr_bins=2, n_trials_per_bin=2, n_ch=2)
        ctf = make_ctf(epochs, df, nr_bins=2, nr_chans=2, pca_cmp=0,
                        shift_bins=0)
        C1 = np.eye(2)
        C2s, W = ctf.forward_model(np.eye(2), np.eye(2), C1)
        np.testing.assert_allclose(W, np.eye(2))
        np.testing.assert_allclose(C2s, [[0., 1.], [0., 1.]])

    @pytest.mark.unit
    def test_3d_input_averages_over_samples_first(self):
        epochs, df = make_spatial_epochs(nr_bins=2, n_trials_per_bin=2, n_ch=2)
        ctf = make_ctf(epochs, df, nr_bins=2, nr_chans=2, pca_cmp=0,
                        shift_bins=0)
        C1 = np.eye(2)
        train_3d = np.stack([np.eye(2), np.eye(2)], axis=-1)
        test_3d = np.stack([np.eye(2), np.eye(2)], axis=-1)
        C2s, W = ctf.forward_model(train_3d, test_3d, C1)
        np.testing.assert_allclose(C2s, [[0., 1.], [0., 1.]])

    @pytest.mark.unit
    def test_shift_bins_rolls_bin_axis_of_both_outputs(self):
        epochs, df = make_spatial_epochs(nr_bins=3, n_trials_per_bin=2, n_ch=3)
        ctf0 = make_ctf(epochs, df, nr_bins=3, nr_chans=3, pca_cmp=0,
                         shift_bins=0)
        ctf1 = make_ctf(epochs, df, nr_bins=3, nr_chans=3, pca_cmp=0,
                         shift_bins=1)
        C1 = np.diag([2.0, 3.0, 5.0])
        C2s_0, W_0 = ctf0.forward_model(C1.copy(), C1.copy(), C1.copy())
        C2s_1, W_1 = ctf1.forward_model(C1.copy(), C1.copy(), C1.copy())
        np.testing.assert_allclose(C2s_1, np.roll(C2s_0, 1, axis=0))
        np.testing.assert_allclose(W_1, np.roll(W_0, 1, axis=0))


# ============================================================================
# CTF.forward_model_loop
# ============================================================================

class TestForwardModelLoop:
    @pytest.mark.unit
    def test_non_gat_shape_and_gat_diagonal_consistency(self):
        epochs, df = make_spatial_epochs(nr_bins=2, n_trials_per_bin=2, n_ch=2)
        ctf = make_ctf(epochs, df, nr_bins=2, nr_chans=2, pca_cmp=0,
                        shift_bins=0, slide_window=0)
        n_samples = 5
        rng = np.random.default_rng(0)
        E_train = rng.normal(0, 1, (2, 4, n_samples))
        E_test = E_train.copy()
        T_train, T_test = E_train.copy(), E_train.copy()
        C1 = np.eye(2)

        C2_E, W_E, C2_T, W_T = ctf.forward_model_loop(
            E_train, E_test, T_train, T_test, C1, GAT=False)
        assert C2_E.shape == (n_samples, 2, 2)

        C2_E_g, W_E_g, C2_T_g, W_T_g = ctf.forward_model_loop(
            E_train, E_test, T_train, T_test, C1, GAT=True)
        assert C2_E_g.shape == (n_samples, n_samples, 2, 2)
        for t in range(n_samples):
            np.testing.assert_allclose(C2_E_g[t, t], C2_E[t])


# ============================================================================
# CTF.train_test_split
# ============================================================================

class TestTrainTestSplit:
    @pytest.mark.unit
    def test_no_train_test_leakage_and_correct_bin_membership(self):
        epochs, df = make_spatial_epochs(nr_bins=3, n_trials_per_bin=12, n_ch=3)
        ctf = make_ctf(epochs, df, nr_bins=3, nr_chans=3, nr_iter=2, nr_folds=3)
        pos_bins = df['position'].values
        cnd_idx = np.ones(pos_bins.size, dtype=bool)
        max_tr = ctf.set_max_trial(np.array(['x'] * pos_bins.size), pos_bins,
                                    'Foster')

        train_idx, test_idx = ctf.train_test_split(pos_bins, cnd_idx, max_tr)
        assert train_idx.shape == (6, 3, 2, max_tr)
        assert test_idx.shape == (6, 3, max_tr)

        for row in range(train_idx.shape[0]):
            train_flat = set(train_idx[row].flatten().tolist())
            test_flat = set(test_idx[row].flatten().tolist())
            assert not (train_flat & test_flat)
            for b in range(3):
                assert (pos_bins[train_idx[row, b].flatten()] == b).all()
                assert (pos_bins[test_idx[row, b]] == b).all()


# ============================================================================
# CTF.train_test_cross
# ============================================================================

class TestTrainTestCross:
    @pytest.mark.unit
    def test_train_and_test_drawn_from_correct_condition_and_bin(self):
        epochs, df = make_spatial_epochs(nr_bins=3, n_trials_per_bin=20, n_ch=3)
        df['task'] = np.tile(['a', 'b'], 30)
        ctf = make_ctf(epochs, df, nr_bins=3, nr_chans=3, nr_iter=2, nr_folds=3)
        pos_bins = df['position'].values
        train_mask = (df['task'] == 'a').values
        test_mask = (df['task'] == 'b').values

        train_idx, test_idx = ctf.train_test_cross(
            pos_bins, train_mask, test_mask, nr_iter=2, trial_limit=None)

        for it in range(train_idx.shape[0]):
            for b in range(train_idx.shape[1]):
                idxs = train_idx[it, b].flatten()
                assert train_mask[idxs].all()
                assert (pos_bins[idxs] == b).all()
        for it in range(test_idx.shape[0]):
            for b in range(test_idx.shape[1]):
                idxs = test_idx[it, b].flatten()
                if idxs.size:
                    assert test_mask[idxs].all()
                    assert (pos_bins[idxs] == b).all()

    @pytest.mark.unit
    def test_trial_limit_prevents_train_test_overlap(self):
        epochs, df = make_spatial_epochs(nr_bins=3, n_trials_per_bin=20, n_ch=3)
        ctf = make_ctf(epochs, df, nr_bins=3, nr_chans=3, nr_iter=2, nr_folds=3)
        pos_bins = df['position'].values
        all_idx = np.ones(pos_bins.size, dtype=bool)

        train_idx, test_idx = ctf.train_test_cross(
            pos_bins, all_idx, all_idx, nr_iter=2, trial_limit=5)

        for it in range(test_idx.shape[0]):
            train_flat = set(train_idx[it].flatten().tolist())
            test_flat = set(test_idx[it].flatten().tolist())
            assert not (train_flat & test_flat)


# ============================================================================
# CTF.set_frequencies
# ============================================================================

class TestSetFrequencies:
    @pytest.mark.unit
    def test_main_param_log_scaling_matches_manual_logspace(self):
        epochs, df = make_spatial_epochs(nr_bins=2, n_trials_per_bin=2, n_ch=2)
        ctf = make_ctf(epochs, df, nr_bins=2, nr_chans=2, min_freq=4,
                        max_freq=32, num_frex=5, freq_scaling='log')
        frex, nr_frex, bands = ctf.set_frequencies('main_param')
        edges = np.logspace(np.log10(4), np.log10(32), 5)
        expected = [(edges[i], edges[i + 1]) for i in range(4)]
        np.testing.assert_allclose(frex, expected)
        assert nr_frex == 4
        assert bands is None

    @pytest.mark.unit
    def test_main_param_linear_scaling_matches_manual_linspace(self):
        epochs, df = make_spatial_epochs(nr_bins=2, n_trials_per_bin=2, n_ch=2)
        ctf = make_ctf(epochs, df, nr_bins=2, nr_chans=2, min_freq=4,
                        max_freq=20, num_frex=5, freq_scaling='linear')
        frex, nr_frex, bands = ctf.set_frequencies('main_param')
        edges = np.linspace(4, 20, 5)
        expected = [(edges[i], edges[i + 1]) for i in range(4)]
        np.testing.assert_allclose(frex, expected)

    @pytest.mark.unit
    def test_broadband(self):
        epochs, df = make_spatial_epochs(nr_bins=2, n_trials_per_bin=2, n_ch=2)
        ctf = make_ctf(epochs, df, nr_bins=2, nr_chans=2)
        frex, nr_frex, bands = ctf.set_frequencies('broadband')
        assert frex == ['broadband']
        assert nr_frex == 1
        assert bands is None

    @pytest.mark.unit
    def test_custom_bands_sorted_by_low_edge(self):
        epochs, df = make_spatial_epochs(nr_bins=2, n_trials_per_bin=2, n_ch=2)
        ctf = make_ctf(epochs, df, nr_bins=2, nr_chans=2)
        frex, nr_frex, bands = ctf.set_frequencies(
            {'alpha': [8, 12], 'theta': [4, 8]})
        assert frex == [(4, 8), (8, 12)]
        assert bands == ['theta', 'alpha']
        assert nr_frex == 2

    @pytest.mark.unit
    def test_invalid_string_raises(self):
        epochs, df = make_spatial_epochs(nr_bins=2, n_trials_per_bin=2, n_ch=2)
        ctf = make_ctf(epochs, df, nr_bins=2, nr_chans=2)
        with pytest.raises(ValueError, match='main_param'):
            ctf.set_frequencies('bogus')

    @pytest.mark.unit
    def test_invalid_type_raises(self):
        epochs, df = make_spatial_epochs(nr_bins=2, n_trials_per_bin=2, n_ch=2)
        ctf = make_ctf(epochs, df, nr_bins=2, nr_chans=2)
        with pytest.raises(ValueError, match='Invalid frequency'):
            ctf.set_frequencies(123)


# ============================================================================
# CTF.extract_slopes
# ============================================================================

class TestExtractSlopes:
    @pytest.mark.unit
    def test_even_nr_chans_symmetric_curve(self):
        epochs, df = make_spatial_epochs(nr_bins=8, n_trials_per_bin=2, n_ch=8)
        ctf = make_ctf(epochs, df, nr_bins=8, nr_chans=8)
        X = np.array([0., 1., 2., 3., 4., 3., 2., 1.])
        assert ctf.extract_slopes(X) == pytest.approx(1.0)

    @pytest.mark.unit
    def test_odd_nr_chans_symmetric_curve(self):
        epochs, df = make_spatial_epochs(nr_bins=7, n_trials_per_bin=2, n_ch=7)
        ctf = make_ctf(epochs, df, nr_bins=7, nr_chans=7)
        X = np.array([0., 1., 2., 3., 2., 1., 0.])
        assert ctf.extract_slopes(X) == pytest.approx(1.0)

    @pytest.mark.unit
    def test_flat_curve_has_zero_slope(self):
        epochs, df = make_spatial_epochs(nr_bins=8, n_trials_per_bin=2, n_ch=8)
        ctf = make_ctf(epochs, df, nr_bins=8, nr_chans=8)
        assert ctf.extract_slopes(np.full(8, 5.0)) == pytest.approx(0.0, abs=1e-9)


# ============================================================================
# CTF.extract_ctf_params
# ============================================================================

class TestExtractCtfParams:
    @pytest.mark.unit
    def test_3d_input_extracts_slope_per_freq_and_sample(self):
        epochs, df = make_spatial_epochs(nr_bins=8, n_trials_per_bin=2, n_ch=8)
        ctf = make_ctf(epochs, df, nr_bins=8, nr_chans=8)
        X = np.array([0., 1., 2., 3., 4., 3., 2., 1.])
        ctfs_3d = np.tile(X, (2, 3, 1))  # (nr_freqs=2, nr_samples=3, nr_chans=8)
        params = {'T_slopes': np.zeros((1, 2, 3, 1))}
        params = ctf.extract_ctf_params(ctfs_3d, params, 'T', perm_idx=0)
        np.testing.assert_allclose(params['T_slopes'][0, :, :, 0], 1.0)

    @pytest.mark.unit
    def test_gat_4d_input_extracts_slope_per_train_test_pair(self):
        epochs, df = make_spatial_epochs(nr_bins=8, n_trials_per_bin=2, n_ch=8)
        ctf = make_ctf(epochs, df, nr_bins=8, nr_chans=8)
        X = np.array([0., 1., 2., 3., 4., 3., 2., 1.])
        ctfs_4d = np.tile(X, (2, 3, 3, 1))  # (freq, train, test, chans)
        params = {'T_slopes': np.zeros((1, 2, 3, 3))}
        params = ctf.extract_ctf_params(ctfs_4d, params, 'T', perm_idx=0)
        np.testing.assert_allclose(params['T_slopes'], 1.0)


# ============================================================================
# CTF.summarize_ctfs
# ============================================================================

class TestSummarizeCtfs:
    @pytest.mark.unit
    def test_test_bins_restriction_avoids_dilution_by_unfitted_bins(self):
        epochs, df = make_spatial_epochs(nr_bins=8, n_trials_per_bin=2, n_ch=8)
        ctf = make_ctf(epochs, df, nr_bins=8, nr_chans=8)
        X = np.array([0., 1., 2., 3., 4., 3., 2., 1.])
        c2 = np.zeros((1, 1, 2, 8, 8))  # (perm, freq, samples, bins, chans)
        c2[0, 0, :, 3, :] = X  # only bin 3 has real (unfitted-bin) data
        ctfs = {'C2_T': c2}

        restricted = ctf.summarize_ctfs(
            ctfs, {'T_slopes': np.zeros((1, 1, 2, 1))}, nr_samples=2,
            nr_freqs=1, test_bins=np.array([3]), nr_perm=1, avg_ch=True)
        np.testing.assert_allclose(restricted['T_slopes'][0, 0], 1.0)

        diluted = ctf.summarize_ctfs(
            ctfs, {'T_slopes': np.zeros((1, 1, 2, 1))}, nr_samples=2,
            nr_freqs=1, test_bins=np.arange(8), nr_perm=1, avg_ch=True)
        # averaging the one real bin against 7 zero-filled bins scales
        # the slope down by 1/nr_bins
        np.testing.assert_allclose(diluted['T_slopes'][0, 0], 1.0 / 8)


# ============================================================================
# CTF.get_ctf_tuning_params
# ============================================================================

class TestGetCtfTuningParams:
    @pytest.mark.unit
    def test_single_frequency_permutation_axis_preserved(self):
        """
        Regression test: the true (index-0) vs. permutation-draw
        (index 1+) split used to protect only "axis 0" from squeezing,
        assuming that axis was always frequency. For permutation draws
        specifically, axis 0 is the permutation-count axis and
        frequency has shifted to axis 1, so with a single frequency
        band, frequency was being silently squeezed out of the
        '_perm' keys only (inconsistent with the true value's shape).
        """
        epochs, df = make_spatial_epochs(nr_bins=8, n_trials_per_bin=2, n_ch=8)
        ctf = make_ctf(epochs, df, nr_bins=8, nr_chans=8)
        X_true = np.array([0., 1., 2., 3., 4., 3., 2., 1.])
        X_null = np.ones(8)
        nr_perm, nr_freqs, nr_samples = 4, 1, 2
        c2 = np.zeros((nr_perm, nr_freqs, nr_samples, 8, 8))
        c2[0, 0] = X_true
        for p in range(1, nr_perm):
            c2[p, 0] = X_null
        ctfs = {'cond_a': {'C2_T': c2, 'C2_E': c2.copy()}}

        ctf_param = ctf.get_ctf_tuning_params(ctfs, ctf_param='slopes',
                                               GAT=False, avg_ch=True)
        assert ctf_param['cond_a']['T_slopes'].shape == (1, 2)
        assert ctf_param['cond_a']['T_slopes_perm'].shape == (3, 1, 2)
        np.testing.assert_allclose(ctf_param['cond_a']['T_slopes'], 1.0)
        np.testing.assert_allclose(ctf_param['cond_a']['T_slopes_perm'], 0.0,
                                    atol=1e-9)

    @pytest.mark.unit
    def test_multi_frequency_permutation_axis_unaffected(self):
        epochs, df = make_spatial_epochs(nr_bins=8, n_trials_per_bin=2, n_ch=8)
        ctf = make_ctf(epochs, df, nr_bins=8, nr_chans=8)
        X_true = np.array([0., 1., 2., 3., 4., 3., 2., 1.])
        X_null = np.ones(8)
        nr_perm, nr_freqs, nr_samples = 4, 3, 2
        c2 = np.zeros((nr_perm, nr_freqs, nr_samples, 8, 8))
        c2[0] = X_true
        for p in range(1, nr_perm):
            c2[p] = X_null
        ctfs = {'cond_a': {'C2_T': c2, 'C2_E': c2.copy()}}

        ctf_param = ctf.get_ctf_tuning_params(ctfs, ctf_param='slopes',
                                               GAT=False, avg_ch=True)
        assert ctf_param['cond_a']['T_slopes'].shape == (3, 2)
        assert ctf_param['cond_a']['T_slopes_perm'].shape == (3, 3, 2)

    @pytest.mark.unit
    def test_no_perm_key_when_nr_perm_is_1(self):
        epochs, df = make_spatial_epochs(nr_bins=8, n_trials_per_bin=2, n_ch=8)
        ctf = make_ctf(epochs, df, nr_bins=8, nr_chans=8)
        X = np.array([0., 1., 2., 3., 4., 3., 2., 1.])
        c2 = np.zeros((1, 1, 2, 8, 8))
        c2[0, 0] = X
        ctfs = {'cond_a': {'C2_T': c2, 'C2_E': c2.copy()}}
        ctf_param = ctf.get_ctf_tuning_params(ctfs, ctf_param='slopes',
                                               GAT=False, avg_ch=True)
        assert 'T_slopes_perm' not in ctf_param['cond_a']


# ============================================================================
# CTF.gaussian / CTF.fit_gaussian
# ============================================================================

class TestGaussian:
    @pytest.mark.unit
    def test_matches_manual_formula(self):
        epochs, df = make_spatial_epochs(nr_bins=2, n_trials_per_bin=2, n_ch=2)
        ctf = make_ctf(epochs, df, nr_bins=2, nr_chans=2)
        x = np.array([0., 1., 2.])
        y = ctf.gaussian(x, amp=2.0, mu=1.0, sig=1.0)
        expected = 2.0 * np.exp(-(x - 1.0) ** 2 / (2 * 1.0 ** 2))
        np.testing.assert_allclose(y, expected)


class TestFitGaussian:
    @pytest.mark.unit
    def test_recovers_true_parameters_from_noisy_data(self):
        epochs, df = make_spatial_epochs(nr_bins=2, n_trials_per_bin=2, n_ch=2)
        ctf = make_ctf(epochs, df, nr_bins=2, nr_chans=2)
        x = np.arange(8)
        true_amp, true_mu, true_sig = 3.0, 4.0, 1.2
        y_clean = true_amp * np.exp(-(x - true_mu) ** 2 / (2 * true_sig ** 2))
        y_noisy = y_clean + np.random.RandomState(0).normal(0, 0.1, 8)

        with warnings.catch_warnings():
            warnings.simplefilter("error")  # OptimizeWarning -> failure
            amp, mu, sig = ctf.fit_gaussian(y_noisy)
        assert amp == pytest.approx(true_amp, abs=0.3)
        assert mu == pytest.approx(true_mu, abs=0.3)
        assert sig == pytest.approx(true_sig, abs=0.3)


# ============================================================================
# CTF.fit_cos_to_ctf
# ============================================================================

class TestFitCosToCtf:
    @pytest.mark.unit
    def test_recovers_exact_parameters_from_noiseless_von_mises(self):
        epochs, df = make_spatial_epochs(nr_bins=8, n_trials_per_bin=2, n_ch=8)
        ctf = make_ctf(epochs, df, nr_bins=8, nr_chans=8)
        num_bins = 8
        spatial_coords = np.linspace(0, pi - pi / num_bins, num_bins)
        true_conc = 10.0
        mean_loc = spatial_coords[num_bins // 2]
        ctf_data = (2.0 * np.exp(true_conc * (np.cos(mean_loc - spatial_coords) - 1))
                    + 0.5)

        amp, base, conc, ml, rmse = ctf.fit_cos_to_ctf(ctf_data, conc_step=0.5)
        assert amp == pytest.approx(2.0, abs=1e-6)
        assert base == pytest.approx(0.5, abs=1e-6)
        assert conc == pytest.approx(true_conc)
        assert ml == pytest.approx(mean_loc)
        assert rmse < 1e-6

    @pytest.mark.unit
    def test_estimate_center_finds_off_center_peak(self):
        epochs, df = make_spatial_epochs(nr_bins=8, n_trials_per_bin=2, n_ch=8)
        ctf = make_ctf(epochs, df, nr_bins=8, nr_chans=8)
        num_bins = 8
        spatial_coords = np.linspace(0, pi - pi / num_bins, num_bins)
        true_conc = 10.0
        off_center_loc = spatial_coords[2]
        ctf_data = (1.5 * np.exp(true_conc * (np.cos(off_center_loc - spatial_coords) - 1))
                    + 0.2)

        amp, base, conc, ml, rmse = ctf.fit_cos_to_ctf(
            ctf_data, conc_step=0.5, estimate_center=True)
        assert ml == pytest.approx(off_center_loc)
        assert rmse < 1e-6


# ============================================================================
# CTF.spatial_ctf (end-to-end)
# ============================================================================

class TestSpatialCtf:
    @pytest.mark.unit
    def test_within_condition_recovers_positive_slope(self):
        epochs, df = make_spatial_epochs(nr_bins=8, n_ch=8, n_trials_per_bin=40,
                                          n_samples=10, signal_amp=3.0, seed=1)
        ctf = make_ctf(epochs, df, nr_iter=3, nr_folds=3)
        ctfs, ctf_param, info = ctf.spatial_ctf(freqs='broadband')
        cnd_key = [k for k in ctfs.keys() if k != 'info'][0]
        assert ctfs[cnd_key]['C2_voltage'].shape == (1, 10, 8)
        assert ctf_param[cnd_key]['voltage_slopes'].mean() > 0.1

    @pytest.mark.unit
    def test_gat_produces_full_train_test_matrix(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        epochs, df = make_spatial_epochs(nr_bins=4, n_ch=4, n_trials_per_bin=20,
                                          n_samples=6, seed=1)
        ctf = make_ctf(epochs, df, nr_bins=4, nr_chans=4, nr_iter=1, nr_folds=2)
        ctfs, ctf_param, info = ctf.spatial_ctf(freqs='broadband', GAT=True)
        cnd_key = [k for k in ctfs.keys() if k != 'info'][0]
        assert ctfs[cnd_key]['C2_voltage'].shape == (1, 6, 6, 4)

    @pytest.mark.unit
    def test_f_name_saves_pickle_files(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        epochs, df = make_spatial_epochs(nr_bins=4, n_ch=4, n_trials_per_bin=20,
                                          n_samples=6, seed=1)
        ctf = make_ctf(epochs, df, nr_bins=4, nr_chans=4, nr_iter=1, nr_folds=2)
        ctf.spatial_ctf(freqs='broadband', f_name='test1')
        saved = list(tmp_path.glob('ctf/**/*.pickle'))
        names = {p.name for p in saved}
        assert names == {'sub_1_test1_ctf.pickle', 'sub_1_test1_info.pickle',
                          'sub_1_test1_param.pickle'}

    @pytest.mark.unit
    def test_cross_condition_localizer_to_ping_with_fixed_special_loc(self):
        epochs, df = make_localizer_and_ping(
            nr_bins=8, n_ch=8, n_trials_per_bin=50, n_ping=200,
            n_samples=15, signal_amp=3.0, ping_special_loc=2, seed=3,
        )
        ctf = make_ctf(epochs, df, nr_iter=3, nr_folds=3)
        ctfs, ctf_param, info = ctf.spatial_ctf(
            freqs='broadband',
            cnds={'task': [['localizer'], ['ping']]},
            special_loc=2,
        )
        cnd_key = [k for k in ctfs.keys() if k != 'info'][0]
        profile = ctfs[cnd_key]['C2_voltage'][0].mean(axis=0)
        # reconstruction should be centered on the common alignment
        # point (ceil(nr_chans/2)) regardless of the ping trials'
        # placeholder to_decode value
        assert np.argmax(profile) == 4

    @pytest.mark.unit
    def test_special_loc_per_trial_column_matches_fixed_int_equivalent(self):
        epochs, df = make_localizer_and_ping(
            nr_bins=8, n_ch=8, n_trials_per_bin=50, n_ping=200,
            n_samples=15, signal_amp=3.0, ping_special_loc=2, seed=3,
        )
        df_col = df.copy()
        df_col['ref_loc'] = df_col['position'].copy()
        df_col.loc[df_col['task'] == 'ping', 'ref_loc'] = 2

        ctf_int = make_ctf(epochs, df.copy(), nr_iter=3, nr_folds=3)
        ctfs_int, _, _ = ctf_int.spatial_ctf(
            freqs='broadband', cnds={'task': [['localizer'], ['ping']]},
            special_loc=2)

        ctf_col = make_ctf(epochs, df_col, nr_iter=3, nr_folds=3)
        ctfs_col, _, _ = ctf_col.spatial_ctf(
            freqs='broadband', cnds={'task': [['localizer'], ['ping']]},
            special_loc='ref_loc')

        cnd_key = [k for k in ctfs_int.keys() if k != 'info'][0]
        np.testing.assert_allclose(ctfs_int[cnd_key]['C2_voltage'],
                                    ctfs_col[cnd_key]['C2_voltage'])

    @pytest.mark.unit
    def test_special_loc_does_not_mutate_original_dataframe(self):
        epochs, df = make_localizer_and_ping(
            nr_bins=8, n_ch=8, n_trials_per_bin=20, n_ping=40,
            n_samples=6, seed=1,
        )
        df_before = df.copy()
        ctf = make_ctf(epochs, df, nr_iter=1, nr_folds=2)
        ctf.spatial_ctf(freqs='broadband',
                         cnds={'task': [['localizer'], ['ping']]},
                         special_loc=2)
        pd.testing.assert_frame_equal(df, df_before)
        pd.testing.assert_frame_equal(ctf.df, df_before)

    @pytest.mark.unit
    def test_special_loc_without_cross_condition_raises(self):
        epochs, df = make_spatial_epochs(nr_bins=4, n_ch=4, n_trials_per_bin=10,
                                          n_samples=6, seed=1)
        ctf = make_ctf(epochs, df, nr_bins=4, nr_chans=4, nr_iter=1, nr_folds=2)
        with pytest.raises(ValueError, match='cross-condition'):
            ctf.spatial_ctf(freqs='broadband', special_loc=1)

    @pytest.mark.unit
    def test_special_loc_bad_column_name_raises(self):
        epochs, df = make_localizer_and_ping(
            nr_bins=4, n_ch=4, n_trials_per_bin=10, n_ping=20,
            n_samples=6, seed=1,
        )
        ctf = make_ctf(epochs, df, nr_bins=4, nr_chans=4, nr_iter=1, nr_folds=2)
        with pytest.raises(ValueError, match='not found'):
            ctf.spatial_ctf(freqs='broadband',
                             cnds={'task': [['localizer'], ['ping']]},
                             special_loc='nonexistent')

    @pytest.mark.unit
    def test_permutation_testing_produces_null_distribution(self):
        # nr_bins=8 (40320 possible bin-to-basis permutations) rather
        # than a smaller count: with very few bins, the permutation
        # space itself is small enough that occasional draws can
        # coincidentally retain some structure, making the null noisy
        # for reasons unrelated to correctness of the implementation
        epochs, df = make_spatial_epochs(nr_bins=8, n_ch=8, n_trials_per_bin=40,
                                          n_samples=8, signal_amp=3.0, seed=0)
        ctf = make_ctf(epochs, df, nr_iter=2, nr_folds=3)
        ctfs, ctf_param, info = ctf.spatial_ctf(freqs='broadband', nr_perm=20)
        cnd_key = [k for k in ctf_param.keys() if k != 'info'][0]
        true_slope = ctf_param[cnd_key]['voltage_slopes']
        perm_slopes = ctf_param[cnd_key]['voltage_slopes_perm']
        assert true_slope.mean() > np.percentile(perm_slopes.mean(axis=-1), 95)

    @pytest.mark.unit
    def test_permutation_true_value_matches_no_permutation_run(self):
        epochs, df = make_spatial_epochs(nr_bins=4, n_ch=4, n_trials_per_bin=30,
                                          n_samples=6, seed=2)
        ctf0 = make_ctf(epochs.copy(), df.copy(), nr_bins=4, nr_chans=4,
                         nr_iter=2, nr_folds=3)
        _, ctf_param0, _ = ctf0.spatial_ctf(freqs='broadband', nr_perm=0)

        ctf3 = make_ctf(epochs.copy(), df.copy(), nr_bins=4, nr_chans=4,
                         nr_iter=2, nr_folds=3)
        _, ctf_param3, _ = ctf3.spatial_ctf(freqs='broadband', nr_perm=3)

        cnd_key = [k for k in ctf_param0.keys() if k != 'info'][0]
        np.testing.assert_allclose(ctf_param0[cnd_key]['voltage_slopes'],
                                    ctf_param3[cnd_key]['voltage_slopes'])

    @pytest.mark.unit
    def test_pos_labels_restricted_subset_not_diluted_by_missing_bins(self):
        """
        Regression test: the raw ctfs['C2_voltage'] reconstruction used
        to be averaged across all nr_bins rows regardless of how many
        bins actually had test data, silently diluting the signal by a
        factor of nr_bins whenever pos_labels restricted analysis to a
        subset of positions (or, equivalently, special_loc assigned all
        test trials to one bin).
        """
        epochs, df = make_localizer_and_ping(
            nr_bins=8, n_ch=8, n_trials_per_bin=50, n_ping=200,
            n_samples=15, signal_amp=3.0, ping_special_loc=2, seed=3,
        )
        ctf = make_ctf(epochs, df, nr_iter=3, nr_folds=3)
        ctfs, _, _ = ctf.spatial_ctf(
            freqs='broadband', cnds={'task': [['localizer'], ['ping']]},
            special_loc=2)
        cnd_key = [k for k in ctfs.keys() if k != 'info'][0]
        profile = ctfs[cnd_key]['C2_voltage'][0].mean(axis=0)
        # with a real injected signal, the peak should be clearly above
        # a small threshold -- a diluted (1/nr_bins-scaled) result
        # would fall far below this
        assert profile.max() > 0.5

    @pytest.mark.unit
    def test_shift_bins_warns_with_avg_ch_true(self):
        epochs, df = make_spatial_epochs(nr_bins=4, n_ch=4, n_trials_per_bin=10,
                                          n_samples=6, seed=1)
        ctf = make_ctf(epochs, df, nr_bins=4, nr_chans=4, nr_iter=1, nr_folds=2,
                        shift_bins=2, avg_ch=True)
        with pytest.warns(UserWarning, match='shift_bins'):
            ctf.spatial_ctf(freqs='broadband')

    @pytest.mark.unit
    def test_shift_bins_no_warning_with_avg_ch_false(self):
        epochs, df = make_spatial_epochs(nr_bins=4, n_ch=4, n_trials_per_bin=10,
                                          n_samples=6, seed=1)
        ctf = make_ctf(epochs, df, nr_bins=4, nr_chans=4, nr_iter=1, nr_folds=2,
                        shift_bins=2, avg_ch=False)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            ctf.spatial_ctf(freqs='broadband')
        assert not any('shift_bins' in str(w.message) for w in caught)


# ============================================================================
# CTF.generate_ctf_report
# ============================================================================

class TestGenerateCtfReport:
    @pytest.mark.unit
    def test_report_html_generated_for_single_band(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        epochs, df = make_spatial_epochs(nr_bins=4, n_ch=4, n_trials_per_bin=10,
                                          n_samples=6, seed=1)
        ctf = make_ctf(epochs, df, nr_bins=4, nr_chans=4, nr_iter=1, nr_folds=2,
                        report=True)
        ctf.spatial_ctf(freqs='broadband', f_name='report_test')
        report_path = tmp_path / 'ctf' / 'report' / 'sub_1_report_test.html'
        assert report_path.exists()
        html = report_path.read_text()
        assert 'CTF slope over time' in html
        assert 'Channel response' in html

    @pytest.mark.unit
    def test_report_html_generated_for_multi_band(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        epochs, df = make_spatial_epochs(nr_bins=4, n_ch=4, n_trials_per_bin=10,
                                          n_samples=200, sfreq=200, seed=1)
        ctf = make_ctf(epochs, df, nr_bins=4, nr_chans=4, nr_iter=1, nr_folds=2,
                        report=True, downsample=50)
        ctf.spatial_ctf(freqs={'theta': [4, 8], 'alpha': [8, 12]},
                         f_name='report_multi')
        report_path = tmp_path / 'ctf' / 'report' / 'sub_1_report_multi.html'
        assert report_path.exists()


# ============================================================================
# Additional coverage: init warnings, cross_cv overlap, k-fold stub,
# slide_window, PCA, laplacian, von_mises end-to-end, avg_ch=False
# ============================================================================

class TestAdditionalPaths:
    @pytest.mark.unit
    def test_init_warns_when_downsample_does_not_evenly_divide_sfreq(self):
        epochs, df = make_spatial_epochs(nr_bins=2, n_ch=2, n_trials_per_bin=4,
                                          sfreq=100)
        with pytest.warns(UserWarning, match='not evenly divisible'):
            make_ctf(epochs, df, nr_bins=2, nr_chans=2, downsample=30)

    @pytest.mark.unit
    def test_forward_model_pca_branch_runs(self):
        epochs, df = make_spatial_epochs(nr_bins=2, n_ch=2, n_trials_per_bin=4)
        ctf = make_ctf(epochs, df, nr_bins=2, nr_chans=2, pca_cmp=1)
        C1 = np.eye(2)
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        C2s, W = ctf.forward_model(X, X, C1)
        assert C2s.shape == (2, 2)

    @pytest.mark.unit
    def test_overlapping_train_test_conditions_uses_cross_cv(self):
        epochs, df = make_spatial_epochs(nr_bins=4, n_ch=4, n_trials_per_bin=20,
                                          n_samples=6)
        df['task'] = ['a'] * 40 + ['b'] * 40
        ctf = make_ctf(epochs, df, nr_bins=4, nr_chans=4)
        with pytest.warns(UserWarning, match='overlapping conditions'):
            ctfs, ctf_param, info = ctf.spatial_ctf(
                freqs='broadband', cnds={'task': [['a', 'b'], ['a']]})
        assert 'a_a' in ctfs

    @pytest.mark.unit
    def test_within_condition_cnds_produces_one_result_per_condition(self):
        epochs, df = make_spatial_epochs(nr_bins=4, n_ch=4, n_trials_per_bin=20,
                                          n_samples=6)
        df['task'] = ['a'] * 40 + ['b'] * 40
        ctf = make_ctf(epochs, df, nr_bins=4, nr_chans=4)
        ctfs, ctf_param, info = ctf.spatial_ctf(freqs='broadband',
                                                 cnds={'task': ['a', 'b']})
        assert set(ctfs.keys()) == {'a', 'b', 'info'}

    @pytest.mark.unit
    def test_slide_window_shortens_output_times(self):
        epochs, df = make_spatial_epochs(nr_bins=4, n_ch=4, n_trials_per_bin=20,
                                          n_samples=6)
        ctf_base = make_ctf(epochs, df, nr_bins=4, nr_chans=4)
        _, ctf_param0, _ = ctf_base.spatial_ctf(freqs='broadband')
        ctf_slide = make_ctf(epochs, df, nr_bins=4, nr_chans=4, slide_window=2)
        _, ctf_param2, _ = ctf_slide.spatial_ctf(freqs='broadband')
        assert (ctf_param2['info']['times'].size
                == ctf_param0['info']['times'].size - 2)

    @pytest.mark.unit
    def test_laplacian_applied_when_montage_available(self):
        epochs, df = make_spatial_epochs(nr_bins=2, n_ch=4, n_trials_per_bin=4,
                                          n_samples=6, seed=1)
        montage = mne.channels.make_standard_montage('standard_1020')
        epochs.rename_channels({'Ch1': 'Fp1', 'Ch2': 'Fp2', 'Ch3': 'F3',
                                 'Ch4': 'F4'})
        epochs.set_montage(montage)
        ctf = make_ctf(epochs, df, nr_bins=2, nr_chans=2, laplacian=True)
        e2, d2 = ctf.select_ctf_data(
            epochs.copy(), df.copy(),
            elec_oi=['Fp1', 'Fp2', 'F3', 'F4'], headers=[None],
            data_type='power')
        assert len(e2) == 8
        assert e2.ch_names == ['Fp1', 'Fp2', 'F3', 'F4']

    @pytest.mark.unit
    def test_von_mises_fitting_end_to_end(self):
        epochs, df = make_spatial_epochs(nr_bins=6, n_ch=6, n_trials_per_bin=10,
                                          n_samples=6, seed=1)
        ctf = make_ctf(epochs, df, nr_bins=6, nr_chans=6, ctf_param='von_mises')
        ctfs, ctf_param, info = ctf.spatial_ctf(freqs='broadband')
        cnd_key = [k for k in ctf_param.keys() if k != 'info'][0]
        for stat in ['slopes', 'amps', 'base', 'conc', 'means']:
            assert f'voltage_{stat}' in ctf_param[cnd_key]
            assert f'envelope_{stat}' in ctf_param[cnd_key]

    @pytest.mark.unit
    def test_avg_ch_false_skips_all_zero_bin(self):
        epochs, df = make_spatial_epochs(nr_bins=6, n_ch=6, n_trials_per_bin=4,
                                          n_samples=6, seed=1)
        ctf = make_ctf(epochs, df, nr_bins=6, nr_chans=6)
        nr_bins = 6
        c2 = np.zeros((1, 1, 2, nr_bins, nr_bins))
        X = np.array([0., 1., 2., 3., 2., 1.])
        c2[0, 0, :, 0, :] = X  # bin 0 has real data; bin 1 stays all-zero
        ctfs_d = {'C2_T': c2}
        params = {'T_slopes': np.zeros((1, 1, 2, 1, nr_bins))}
        out = ctf.summarize_ctfs(ctfs_d, params, nr_samples=2, nr_freqs=1,
                                  test_bins=np.arange(nr_bins), nr_perm=1,
                                  avg_ch=False)
        np.testing.assert_allclose(out['T_slopes'][0, 0, :, 0, 0], 1.0)
        np.testing.assert_allclose(out['T_slopes'][0, 0, :, 0, 1], 0.0)

    @pytest.mark.unit
    def test_actual_sfreq_mismatch_prints_correct_value(self, capsys):
        """
        Regression test: the mismatch-warning print split its message
        across two string literals but only the first had an f-prefix,
        so the second half printed the literal text
        '{self.downsample}' instead of substituting the actual value.
        """
        epochs, df = make_spatial_epochs(nr_bins=2, n_ch=2, n_trials_per_bin=4,
                                          sfreq=100, n_samples=10)
        ctf = make_ctf(epochs, df, nr_bins=2, nr_chans=2, downsample=30)
        ctf.spatial_ctf(freqs='broadband')
        out = capsys.readouterr().out
        assert 'desired downsample (30)' in out
        assert '{self.downsample}' not in out

    @pytest.mark.unit
    def test_extract_ctf_params_gaussian_fitting_branch(self):
        epochs, df = make_spatial_epochs(nr_bins=6, n_ch=6, n_trials_per_bin=2,
                                          n_samples=6)
        ctf = make_ctf(epochs, df, nr_bins=6, nr_chans=6)
        X = np.array([0., 1., 2., 3., 2., 1.])
        ctfs_3d = np.tile(X, (1, 2, 1))
        params = {f'T_{k}': np.zeros((1, 1, 2, 1))
                  for k in ['slopes', 'amps', 'base', 'conc', 'means']}
        params = ctf.extract_ctf_params(ctfs_3d, params, 'T', perm_idx=0,
                                         fitting_method='gaussian')
        assert params['T_amps'][0, :, :, 0].max() > 0

    @pytest.mark.unit
    def test_summarize_ctfs_gat_avg_ch_false(self):
        epochs, df = make_spatial_epochs(nr_bins=6, n_ch=6, n_trials_per_bin=2,
                                          n_samples=6)
        ctf = make_ctf(epochs, df, nr_bins=6, nr_chans=6)
        X = np.array([0., 1., 2., 3., 2., 1.])
        c2 = np.zeros((1, 1, 2, 2, 6, 6))  # (perm,freq,train,test,bins,chans)
        c2[0, 0, :, :, 0, :] = X
        params = {'T_slopes': np.zeros((1, 1, 2, 2, 1, 6))}
        out = ctf.summarize_ctfs({'C2_T': c2}, params, nr_samples=(2, 2),
                                  nr_freqs=1, test_bins=np.arange(6),
                                  nr_perm=1, avg_ch=False)
        assert out['T_slopes'].shape == (1, 1, 2, 2, 1, 6)
        np.testing.assert_allclose(out['T_slopes'][0, 0, :, :, 0, 0], 1.0)

    @pytest.mark.unit
    def test_extract_ctf_params_unknown_fitting_method_raises(self):
        epochs, df = make_spatial_epochs(nr_bins=6, n_ch=6, n_trials_per_bin=2,
                                          n_samples=6)
        ctf = make_ctf(epochs, df, nr_bins=6, nr_chans=6)
        X = np.array([0., 1., 2., 3., 2., 1.])
        ctfs_3d = np.tile(X, (1, 2, 1))
        params = {'T_slopes': np.zeros((1, 1, 2, 1)),
                  'T_amps': np.zeros((1, 1, 2, 1))}
        with pytest.raises(ValueError, match='Unknown fitting method'):
            ctf.extract_ctf_params(ctfs_3d, params, 'T', perm_idx=0,
                                    fitting_method='bogus')

    @pytest.mark.unit
    def test_extract_ctf_params_per_channel_fitting(self):
        epochs, df = make_spatial_epochs(nr_bins=6, n_ch=6, n_trials_per_bin=2,
                                          n_samples=6)
        ctf = make_ctf(epochs, df, nr_bins=6, nr_chans=6)
        X = np.array([0., 1., 2., 3., 2., 1.])
        ctfs_3d = np.tile(X, (1, 2, 1))
        params = {f'T_{k}': np.zeros((1, 1, 2, 1, 6))
                  for k in ['slopes', 'amps', 'base', 'conc', 'means']}
        out = ctf.extract_ctf_params(ctfs_3d, params, 'T', perm_idx=0,
                                      ch_idx=3, fitting_method='von_mises')
        assert out['T_amps'][0, :, :, 0, 3].max() > 0
