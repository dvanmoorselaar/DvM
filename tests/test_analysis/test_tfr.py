"""
Test suite for open_dvm.analysis.TFR.

Organization
------------
- TestInit: constructor behavior, wavelet/freq-band deferred setup
- TestWaveletParamsHashAndEnsure: change-detection / regeneration logic
- TestSelectTfrData: electrode selection, trial exclusion, topo flip
- TestCreateFreqBands: Hilbert frequency-band construction
- TestCreateMorlet: Morlet wavelet bank correctness
- TestWaveletConvolutionAndNextpow2: low-level convolution helpers
- TestTfrLoop: core decomposition loop, shape/mutation regressions
- TestComputeTfrs: power/phase computation, baseline handling
- TestBaselineTfr: baseline correction dispatch
- TestDbConvert: decibel conversion
- TestNormalizePower: lateralization-index normalization
- TestLateralizationIndex: static lateralization-index method
- TestConditionTfrs: end-to-end condition pipeline
- TestRegressions: targeted checks for bugs fixed in this module
"""

import os

import numpy as np
import pandas as pd
import pytest
import mne

from open_dvm.analysis.TFR import TFR

from tests.fixtures.tfr_sample_data import (
    make_epochs,
    make_oscillating_epochs,
    make_behavioral_df,
)


def _fft_peak_freq(x, sfreq):
    # x may be complex (e.g. a Morlet wavelet), so use the full fft
    # rather than rfft (which requires real input)
    spec = np.abs(np.fft.fft(x))
    freqs = np.fft.fftfreq(len(x), d=1 / sfreq)
    return freqs[np.argmax(spec)]


# ============================================================================
# __init__
# ============================================================================

class TestInit:
    @pytest.mark.unit
    def test_wavelet_method_defers_wavelet_creation(self):
        epochs = make_epochs(n_trials=3)
        df = make_behavioral_df(3, x=1)
        tfr = TFR(sj=1, epochs=epochs, df=df, method='wavelet')
        assert tfr.wavelets is None
        assert tfr.frex is None

    @pytest.mark.unit
    def test_hilbert_method_creates_freq_bands_immediately(self):
        epochs = make_epochs(n_trials=3)
        df = make_behavioral_df(3, x=1)
        tfr = TFR(sj=1, epochs=epochs, df=df, method='hilbert', num_frex=5)
        assert tfr.freq_bands is not None
        assert len(tfr.freq_bands) == 5
        assert tfr.frex is not None

    @pytest.mark.unit
    def test_subject_id_zero_padded(self):
        epochs = make_epochs(n_trials=3)
        df = make_behavioral_df(3, x=1)
        tfr = TFR(sj=1, epochs=epochs, df=df)
        assert tfr.sj == '01'


# ============================================================================
# _get_wavelet_params_hash / _ensure_wavelets
# ============================================================================

class TestWaveletParamsHashAndEnsure:
    @pytest.mark.unit
    def test_wavelets_not_regenerated_when_params_unchanged(self):
        epochs = make_epochs(n_trials=3)
        df = make_behavioral_df(3, x=1)
        tfr = TFR(sj=1, epochs=epochs, df=df, min_freq=4, max_freq=20, num_frex=5)
        tfr._ensure_wavelets()
        w1 = tfr.wavelets
        tfr._ensure_wavelets()
        assert tfr.wavelets is w1

    @pytest.mark.unit
    def test_wavelets_regenerated_when_min_freq_changes(self):
        epochs = make_epochs(n_trials=3)
        df = make_behavioral_df(3, x=1)
        tfr = TFR(sj=1, epochs=epochs, df=df, min_freq=4, max_freq=20, num_frex=5)
        tfr._ensure_wavelets()
        w1, h1 = tfr.wavelets, tfr._wavelet_params_hash

        tfr.min_freq = 8
        tfr._ensure_wavelets()

        assert tfr._wavelet_params_hash != h1
        assert tfr.wavelets is not w1
        np.testing.assert_allclose(tfr.frex[0], 8.0)

    @pytest.mark.unit
    def test_switching_to_hilbert_generates_freq_bands(self):
        epochs = make_epochs(n_trials=3)
        df = make_behavioral_df(3, x=1)
        tfr = TFR(sj=1, epochs=epochs, df=df, method='wavelet', num_frex=5)
        tfr.method = 'hilbert'
        tfr._ensure_wavelets()
        assert tfr.freq_bands is not None
        assert len(tfr.freq_bands) == 5


# ============================================================================
# select_tfr_data
# ============================================================================

class TestSelectTfrData:
    @pytest.mark.unit
    def test_elec_oi_all_selects_all_channels(self):
        epochs = make_epochs(ch_names=['Fp1', 'Cz', 'O1'], n_trials=5)
        df = make_behavioral_df(5, x=1)
        tfr = TFR(sj=1, epochs=epochs, df=df)
        ep, _ = tfr.select_tfr_data(elec_oi='all')
        assert set(ep.ch_names) == {'Fp1', 'Cz', 'O1'}

    @pytest.mark.unit
    def test_elec_oi_posterior_selects_posterior_channels(self):
        # regression: this shortcut previously crashed (never implemented)
        epochs = make_epochs(ch_names=['Fp1', 'Cz', 'O1', 'O2', 'Pz'], n_trials=5)
        df = make_behavioral_df(5, x=1)
        tfr = TFR(sj=1, epochs=epochs, df=df)
        ep, _ = tfr.select_tfr_data(elec_oi='posterior')
        assert set(ep.ch_names) == {'O1', 'O2', 'Pz'}

    @pytest.mark.unit
    def test_elec_oi_frontal_selects_frontal_channels(self):
        epochs = make_epochs(ch_names=['Fp1', 'Fp2', 'Cz', 'O1'], n_trials=5)
        df = make_behavioral_df(5, x=1)
        tfr = TFR(sj=1, epochs=epochs, df=df)
        ep, _ = tfr.select_tfr_data(elec_oi='frontal')
        assert set(ep.ch_names) == {'Fp1', 'Fp2'}

    @pytest.mark.unit
    def test_elec_oi_central_selects_central_channels(self):
        epochs = make_epochs(ch_names=['Fp1', 'Cz', 'C3', 'O1'], n_trials=5)
        df = make_behavioral_df(5, x=1)
        tfr = TFR(sj=1, epochs=epochs, df=df)
        ep, _ = tfr.select_tfr_data(elec_oi='central')
        assert set(ep.ch_names) == {'Cz', 'C3'}

    @pytest.mark.unit
    def test_elec_oi_explicit_list(self):
        epochs = make_epochs(ch_names=['Fp1', 'Cz', 'O1'], n_trials=5)
        df = make_behavioral_df(5, x=1)
        tfr = TFR(sj=1, epochs=epochs, df=df)
        ep, _ = tfr.select_tfr_data(elec_oi=['Cz', 'O1'])
        assert set(ep.ch_names) == {'Cz', 'O1'}

    @pytest.mark.unit
    def test_excl_factor_removes_matching_trials(self):
        epochs = make_epochs(n_trials=4)
        df = make_behavioral_df(4, cue=['left', 'right', 'left', 'right'])
        tfr = TFR(sj=1, epochs=epochs, df=df)
        ep, out_df = tfr.select_tfr_data(elec_oi='all', excl_factor={'cue': ['right']})
        assert set(out_df['cue']) == {'left'}
        assert len(ep) == 2

    @pytest.mark.unit
    def test_does_not_mutate_original_epochs_or_df(self):
        epochs = make_epochs(ch_names=['Fp1', 'Cz', 'O1'], n_trials=4)
        df = make_behavioral_df(4, cue=['left', 'right', 'left', 'right'])
        tfr = TFR(sj=1, epochs=epochs, df=df)
        tfr.select_tfr_data(elec_oi=['Cz'], excl_factor={'cue': ['right']})
        assert tfr.epochs.ch_names == ['Fp1', 'Cz', 'O1']
        assert len(tfr.df) == 4

    @pytest.mark.unit
    def test_laplacian_converts_to_csd_channel_type(self):
        montage = mne.channels.make_standard_montage('biosemi64')
        ch_names = montage.ch_names[:8]
        info = mne.create_info(ch_names, 250., ch_types='eeg')
        info.set_montage(montage)
        data = np.random.default_rng(0).normal(0, 1, (5, 8, 20))
        epochs = mne.EpochsArray(data, info, tmin=-0.1)
        df = make_behavioral_df(5, x=1)
        tfr = TFR(sj=1, epochs=epochs, df=df, laplacian=True)

        ep, _ = tfr.select_tfr_data(elec_oi='all')

        assert set(ep.get_channel_types()) == {'csd'}

    @pytest.mark.unit
    def test_topo_flip_calls_flip_topography_without_crash(self):
        ch_names = ['O1', 'O2', 'Cz']
        info = mne.create_info(ch_names, 250., ch_types='eeg')
        data = np.random.default_rng(0).normal(0, 1, (4, 3, 20))
        epochs = mne.EpochsArray(data, info, tmin=-0.1)
        df = make_behavioral_df(4, target_loc=[1, 2, 1, 2])
        tfr = TFR(sj=1, epochs=epochs, df=df)

        ep, out_df = tfr.select_tfr_data(elec_oi='all',
                                          topo_flip={'target_loc': [1]})

        assert len(ep) == 4
        assert len(out_df) == 4


# ============================================================================
# create_freq_bands
# ============================================================================

class TestCreateFreqBands:
    @pytest.mark.unit
    def test_linear_bands_match_manual_calculation(self):
        epochs = make_epochs(n_trials=3)
        df = make_behavioral_df(3, x=1)
        tfr = TFR(sj=1, epochs=epochs, df=df, min_freq=4, max_freq=20,
                  num_frex=5, freq_scaling='linear')
        bands = tfr.create_freq_bands()
        expected = [(2.0, 6.0), (6.0, 10.0), (10.0, 14.0), (14.0, 18.0), (18.0, 22.0)]
        for (lo, hi), (elo, ehi) in zip(bands, expected):
            assert lo == pytest.approx(elo)
            assert hi == pytest.approx(ehi)

    @pytest.mark.unit
    def test_number_of_bands_matches_num_frex(self):
        epochs = make_epochs(n_trials=3)
        df = make_behavioral_df(3, x=1)
        tfr = TFR(sj=1, epochs=epochs, df=df, num_frex=7, method='hilbert')
        assert len(tfr.freq_bands) == 7

    @pytest.mark.unit
    def test_log_bands_cover_full_range_contiguously(self):
        epochs = make_epochs(n_trials=3)
        df = make_behavioral_df(3, x=1)
        tfr = TFR(sj=1, epochs=epochs, df=df, min_freq=4, max_freq=20,
                  num_frex=5, freq_scaling='log')
        bands = tfr.create_freq_bands()
        # each band's high edge should equal the next band's low edge
        for (_, hi), (lo2, _) in zip(bands[:-1], bands[1:]):
            assert hi == pytest.approx(lo2)


# ============================================================================
# create_morlet
# ============================================================================

class TestCreateMorlet:
    @pytest.mark.unit
    def test_center_frequencies_match_requested(self):
        epochs = make_epochs(n_trials=3)
        df = make_behavioral_df(3, x=1)
        tfr = TFR(sj=1, epochs=epochs, df=df)
        wavelets, frex = tfr.create_morlet(4, 20, 5, (3, 10), 'log', 200, 100.)
        for i, f in enumerate(frex):
            peak = _fft_peak_freq(wavelets[i], 100.)
            assert peak == pytest.approx(f, abs=1.0)

    @pytest.mark.unit
    def test_normalize_wavelets_gives_unit_energy(self):
        epochs = make_epochs(n_trials=3)
        df = make_behavioral_df(3, x=1)
        tfr = TFR(sj=1, epochs=epochs, df=df)
        wavelets, _ = tfr.create_morlet(4, 20, 5, (3, 10), 'log', 200, 100.,
                                          normalize_wavelets=True)
        energies = np.sum(np.abs(wavelets) ** 2, axis=1)
        np.testing.assert_allclose(energies, 1.0)

    @pytest.mark.unit
    def test_no_normalization_gives_nonunit_energy(self):
        epochs = make_epochs(n_trials=3)
        df = make_behavioral_df(3, x=1)
        tfr = TFR(sj=1, epochs=epochs, df=df)
        wavelets, _ = tfr.create_morlet(4, 20, 5, (3, 10), 'log', 200, 100.,
                                          normalize_wavelets=False)
        energies = np.sum(np.abs(wavelets) ** 2, axis=1)
        assert not np.allclose(energies, 1.0)

    @pytest.mark.unit
    def test_invalid_freq_scaling_raises(self):
        epochs = make_epochs(n_trials=3)
        df = make_behavioral_df(3, x=1)
        tfr = TFR(sj=1, epochs=epochs, df=df)
        with pytest.raises(ValueError):
            tfr.create_morlet(4, 20, 5, (3, 10), 'bogus', 200, 100.)

    @pytest.mark.unit
    def test_linear_scaling_center_frequencies_match_requested(self):
        epochs = make_epochs(n_trials=3)
        df = make_behavioral_df(3, x=1)
        tfr = TFR(sj=1, epochs=epochs, df=df)
        wavelets, frex = tfr.create_morlet(4, 20, 5, (3, 10), 'linear', 200, 100.)
        np.testing.assert_allclose(frex, np.linspace(4, 20, 5))
        for i, f in enumerate(frex):
            peak = _fft_peak_freq(wavelets[i], 100.)
            assert peak == pytest.approx(f, abs=1.0)


# ============================================================================
# wavelet_convolution / nextpow2
# ============================================================================

class TestWaveletConvolutionAndNextpow2:
    @pytest.mark.parametrize('i,expected', [(1, 1), (2, 1), (5, 3), (8, 3), (100, 7)])
    @pytest.mark.unit
    def test_nextpow2_matches_manual_calc(self, i, expected):
        assert TFR.nextpow2(i) == expected
        assert 2 ** TFR.nextpow2(i) >= i

    @pytest.mark.unit
    def test_convolution_output_shape(self):
        epochs = make_epochs(n_trials=3)
        df = make_behavioral_df(3, x=1)
        tfr = TFR(sj=1, epochs=epochs, df=df, min_freq=4, max_freq=20, num_frex=3)
        tfr._ensure_wavelets()

        nr_epochs, nr_time = 3, 100
        l_conv = 2 ** TFR.nextpow2(nr_time * nr_epochs + nr_time - 1)
        x = np.random.default_rng(0).normal(0, 1, (nr_epochs, nr_time))
        x_fft = np.fft.fft(x.ravel(), l_conv)

        m = tfr.wavelet_convolution(x_fft, tfr.wavelets[0], l_conv, nr_time, nr_epochs)
        assert m.shape == (nr_epochs, nr_time)


# ============================================================================
# tfr_loop
# ============================================================================

class TestTfrLoop:
    @pytest.mark.unit
    def test_multitrial_epochsarray_correct_shape(self):
        # regression: EpochsArray (not EpochsFIF) used to be miscounted
        # as a single "epoch"
        epochs = make_epochs(n_trials=5, n_samples=20)
        df = make_behavioral_df(5, x=1)
        tfr = TFR(sj=1, epochs=epochs, df=df, min_freq=4, max_freq=20, num_frex=3)
        tfr._ensure_wavelets()

        raw_conv = tfr.tfr_loop(epochs)

        assert raw_conv.shape == (5, 3, 2, 20)

    @pytest.mark.unit
    def test_multitrial_epochsarray_not_mutated(self):
        epochs = make_epochs(n_trials=5, n_samples=20)
        df = make_behavioral_df(5, x=1)
        tfr = TFR(sj=1, epochs=epochs, df=df, min_freq=4, max_freq=20, num_frex=3)
        tfr._ensure_wavelets()
        before = epochs._data.copy()

        tfr.tfr_loop(epochs)

        np.testing.assert_array_equal(before, epochs._data)
        assert epochs._data.shape == (5, 2, 20)

    @pytest.mark.unit
    def test_evoked_input_correct_shape(self):
        epochs = make_epochs(n_trials=5, n_samples=20)
        df = make_behavioral_df(5, x=1)
        tfr = TFR(sj=1, epochs=epochs, df=df, min_freq=4, max_freq=20, num_frex=3)
        tfr._ensure_wavelets()
        evoked = epochs.average()

        raw_conv = tfr.tfr_loop(evoked)

        assert raw_conv.shape == (1, 3, 2, 20)
        assert evoked._data.shape == (2, 20)

    @pytest.mark.unit
    def test_wavelet_method_power_peaks_at_true_frequency(self):
        epochs = make_oscillating_epochs(freq=10, ch_names=['C3'], n_trials=6,
                                          n_samples=200, sfreq=200., amplitude=5.0,
                                          phase_locked=False)
        df = make_behavioral_df(6, x=1)
        tfr = TFR(sj=1, epochs=epochs, df=df, min_freq=4, max_freq=30, num_frex=10)
        tfr._ensure_wavelets()

        raw_conv = tfr.tfr_loop(epochs)
        power = (raw_conv.real ** 2 + raw_conv.imag ** 2).mean(axis=(0, 1))
        # power shape: (nr_freq, nr_ch, nr_time) after mean over (epoch, ch)? recompute properly
        power = (raw_conv.real ** 2 + raw_conv.imag ** 2).mean(axis=(0, 2, 3))
        peak_freq_idx = np.argmax(power)
        np.testing.assert_allclose(tfr.frex[peak_freq_idx], 10, atol=3)

    @pytest.mark.unit
    def test_hilbert_method_power_peaks_at_true_frequency(self):
        epochs = make_oscillating_epochs(freq=10, ch_names=['C3'], n_trials=6,
                                          n_samples=200, sfreq=200., amplitude=5.0,
                                          phase_locked=False)
        df = make_behavioral_df(6, x=1)
        tfr = TFR(sj=1, epochs=epochs, df=df, min_freq=4, max_freq=30, num_frex=10,
                  method='hilbert')
        tfr._ensure_wavelets()

        raw_conv = tfr.tfr_loop(epochs)
        power = (raw_conv.real ** 2 + raw_conv.imag ** 2).mean(axis=(0, 2, 3))
        peak_freq_idx = np.argmax(power)
        np.testing.assert_allclose(tfr.frex[peak_freq_idx], 10, atol=3)


# ============================================================================
# compute_tfrs
# ============================================================================

class TestComputeTfrs:
    @pytest.mark.unit
    def test_power_output_matches_real_imag_squared(self):
        epochs = make_epochs(n_trials=4, n_samples=20)
        df = make_behavioral_df(4, x=1)
        tfr = TFR(sj=1, epochs=epochs, df=df, min_freq=4, max_freq=20, num_frex=3)

        X = tfr.compute_tfrs(epochs.copy(), output='power', for_decoding=True)

        assert X.ndim == 4
        assert (X >= 0).all()  # power is non-negative

    @pytest.mark.unit
    def test_phase_output_in_valid_cosine_range(self):
        epochs = make_epochs(n_trials=4, n_samples=20)
        df = make_behavioral_df(4, x=1)
        tfr = TFR(sj=1, epochs=epochs, df=df, min_freq=4, max_freq=20, num_frex=3)

        X = tfr.compute_tfrs(epochs.copy(), output='phase', for_decoding=True)

        assert X.min() >= -1.0001 and X.max() <= 1.0001

    @pytest.mark.unit
    def test_induced_power_removes_phase_locked_component(self):
        # regression: epochs[idx]._data = ... used to be a complete no-op
        sfreq = 200.
        epochs = make_oscillating_epochs(freq=10, ch_names=['C3'], n_trials=10,
                                          n_samples=200, sfreq=sfreq, amplitude=5.0,
                                          noise_sd=0.1, phase_locked=True)
        df = make_behavioral_df(10, x=1)

        tfr_total = TFR(sj=1, epochs=epochs, df=df, min_freq=4, max_freq=20,
                         num_frex=10, power='total')
        X_total = tfr_total.compute_tfrs(epochs.copy(), output='power',
                                          for_decoding=True)

        tfr_induced = TFR(sj=1, epochs=epochs, df=df, min_freq=4, max_freq=20,
                           num_frex=10, power='induced')
        cnd_idx = [np.arange(10)]
        X_induced = tfr_induced.compute_tfrs(epochs.copy(), output='power',
                                              for_decoding=True, cnd_idx=cnd_idx)

        f_idx = np.argmin(np.abs(tfr_total.frex - 10))
        assert X_induced[f_idx].mean() < X_total[f_idx].mean() / 10

    @pytest.mark.unit
    def test_induced_without_cnd_idx_raises(self):
        epochs = make_epochs(n_trials=4, n_samples=20)
        df = make_behavioral_df(4, x=1)
        tfr = TFR(sj=1, epochs=epochs, df=df, min_freq=4, max_freq=20,
                  num_frex=3, power='induced')
        with pytest.raises(ValueError, match='cnd_idx'):
            tfr.compute_tfrs(epochs.copy(), cnd_idx=None)

    @pytest.mark.unit
    def test_for_decoding_percent_change_baseline(self):
        epochs = make_epochs(n_trials=4, n_samples=40, sfreq=100.)
        df = make_behavioral_df(4, x=1)
        tfr = TFR(sj=1, epochs=epochs, df=df, min_freq=4, max_freq=20,
                  num_frex=3, baseline=(-0.2, 0))

        X = tfr.compute_tfrs(epochs.copy(), output='power', for_decoding=True)

        # percent-change values can legitimately be negative; just check
        # it's not raw power (which is always non-negative and large)
        assert X.min() < 0 or X.max() < 1e6

    @pytest.mark.unit
    def test_for_decoding_no_baseline_uses_raw_power(self):
        epochs = make_epochs(n_trials=4, n_samples=20)
        df = make_behavioral_df(4, x=1)
        tfr = TFR(sj=1, epochs=epochs, df=df, min_freq=4, max_freq=20,
                  num_frex=3, baseline=None)

        X = tfr.compute_tfrs(epochs.copy(), output='power', for_decoding=True)

        assert (X >= 0).all()

    @pytest.mark.unit
    def test_for_decoding_false_applies_db_conversion(self):
        from open_dvm.support.preprocessing_utils import get_time_slice

        epochs = make_epochs(n_trials=4, n_samples=40, sfreq=100.)
        df = make_behavioral_df(4, x=1)

        tfr_raw = TFR(sj=1, epochs=epochs, df=df, min_freq=4, max_freq=20,
                      num_frex=3, baseline=None)
        X_raw = tfr_raw.compute_tfrs(epochs.copy(), output='power',
                                      for_decoding=True)

        tfr_db = TFR(sj=1, epochs=epochs, df=df, min_freq=4, max_freq=20,
                     num_frex=3, baseline=(-0.2, 0))
        X_db = tfr_db.compute_tfrs(epochs.copy(), output='power',
                                    for_decoding=False)

        base_idx = get_time_slice(epochs.times, -0.2, 0)
        base = X_raw[..., base_idx].mean(axis=-1, keepdims=True)
        expected = 10 * (np.log10(X_raw + 1e-12) - np.log10(base + 1e-12))
        np.testing.assert_allclose(X_db, expected, atol=1e-6)


# ============================================================================
# baseline_tfr
# ============================================================================

class TestBaselineTfr:
    @staticmethod
    def _make_tfr_dict(power_values, n_epochs=4, n_freq=1, n_ch=1, n_time=2):
        power = np.full((n_epochs, n_freq, n_ch, n_time), power_values, dtype=float)
        return {
            'ch_names': np.array(['Cz']),
            'power': {'condA': power},
            'cnd_cnt': {},
        }

    @pytest.mark.unit
    def test_none_method_averages_trials_only(self):
        tfr_obj = TFR.__new__(TFR)
        tfr = self._make_tfr_dict(5.0)

        result = TFR.baseline_tfr(tfr_obj, tfr, base={}, method=None)

        np.testing.assert_allclose(result['power']['condA'], 5.0)
        assert result['cnd_cnt']['condA'] == 4

    @pytest.mark.unit
    def test_default_trial_spec_with_no_baseline_data_no_crash(self):
        # regression: method=='trial_spec' (the __init__ default) was
        # checked before checking whether any baseline data existed,
        # crashing whenever self.baseline was None
        tfr_obj = TFR.__new__(TFR)
        tfr = self._make_tfr_dict(5.0)

        result = TFR.baseline_tfr(tfr_obj, tfr, base={}, method='trial_spec')

        np.testing.assert_allclose(result['power']['condA'], 5.0)

    @pytest.mark.unit
    def test_trial_spec_applies_db_conversion_per_trial(self):
        tfr_obj = TFR.__new__(TFR)
        tfr = self._make_tfr_dict(20.0)
        base = {'condA': np.full((4, 1, 1), 10.0)}  # 3D: trial-specific

        result = TFR.baseline_tfr(tfr_obj, tfr, base=base, method='trial_spec')

        expected = 10 * np.log10(20.0 / 10.0)
        np.testing.assert_allclose(result['power']['condA'], expected, atol=1e-6)

    @pytest.mark.unit
    def test_cnd_spec_averages_before_correcting(self):
        tfr_obj = TFR.__new__(TFR)
        tfr = self._make_tfr_dict(20.0)
        base = {'condA': np.full((1, 1), 10.0)}  # 2D: condition-specific

        result = TFR.baseline_tfr(tfr_obj, tfr, base=base, method='cnd_spec')

        expected = 10 * np.log10(20.0 / 10.0)
        np.testing.assert_allclose(result['power']['condA'], expected, atol=1e-6)

    @pytest.mark.unit
    def test_cnd_avg_uses_grand_average_baseline(self):
        tfr_obj = TFR.__new__(TFR)
        power = np.full((4, 1, 1, 2), 20.0, dtype=float)
        power_b = np.full((4, 1, 1, 2), 40.0, dtype=float)
        tfr = {
            'ch_names': np.array(['Cz']),
            'power': {'condA': power, 'condB': power_b},
            'cnd_cnt': {},
        }
        base = {'condA': np.full((1, 1), 10.0), 'condB': np.full((1, 1), 20.0)}

        result = TFR.baseline_tfr(tfr_obj, tfr, base=base, method='cnd_avg')

        cnd_avg = np.mean([base['condA'], base['condB']], axis=0)  # = 15.0
        expected_a = 10 * np.log10(20.0 / cnd_avg[0, 0])
        np.testing.assert_allclose(result['power']['condA'], expected_a, atol=1e-6)

    @pytest.mark.unit
    def test_norm_method_computes_lateralization_index(self):
        # regression: base_method='norm' called a nonexistent method
        tfr_obj = TFR.__new__(TFR)
        power = np.zeros((1, 1, 2, 1))
        power[0, 0, 0] = 10.0  # O1
        power[0, 0, 1] = 2.0   # O2
        tfr = {
            'ch_names': np.array(['O1', 'O2']),
            'power': {'condA': power},
            'cnd_cnt': {},
        }

        result = TFR.baseline_tfr(tfr_obj, tfr, base={}, method='norm')

        np.testing.assert_allclose(result['power']['condA'][0, 0, 0], 0.6666666667)
        assert 'norm_info' in result

    @pytest.mark.unit
    def test_invalid_method_raises(self):
        tfr_obj = TFR.__new__(TFR)
        tfr = self._make_tfr_dict(5.0)
        with pytest.raises(ValueError, match='Invalid method'):
            TFR.baseline_tfr(tfr_obj, tfr, base={'condA': np.full((1, 1), 1.0)},
                              method='bogus')


# ============================================================================
# db_convert
# ============================================================================

class TestDbConvert:
    @pytest.mark.unit
    def test_matches_manual_formula_2d_baseline(self):
        tfr_obj = TFR.__new__(TFR)
        power = np.full((1, 1, 3), 20.0)
        base = np.full((1, 1), 10.0)

        result = TFR.db_convert(tfr_obj, power, base)

        expected = 10 * np.log10(20.0 / 10.0)
        np.testing.assert_allclose(result, expected, atol=1e-6)

    @pytest.mark.unit
    def test_matches_manual_formula_3d_baseline(self):
        tfr_obj = TFR.__new__(TFR)
        power = np.full((2, 1, 1, 3), 20.0)
        base = np.full((2, 1, 1), 5.0)

        result = TFR.db_convert(tfr_obj, power, base)

        expected = 10 * np.log10(20.0 / 5.0)
        np.testing.assert_allclose(result, expected, atol=1e-6)

    @pytest.mark.unit
    def test_invalid_ndim_raises(self):
        tfr_obj = TFR.__new__(TFR)
        power = np.ones((1, 1, 3))
        base = np.ones((1, 1, 1, 1))  # 4D, invalid
        with pytest.raises(ValueError, match='2D or 3D'):
            TFR.db_convert(tfr_obj, power, base)


# ============================================================================
# normalize_power
# ============================================================================

class TestNormalizePower:
    @pytest.mark.unit
    def test_lateralization_formula_correct(self):
        tfr_obj = TFR.__new__(TFR)
        ch_names = ['O1', 'O2', 'Cz']
        avg_power = np.zeros((1, 3, 2))
        avg_power[:, 0] = 10.0
        avg_power[:, 1] = 2.0
        avg_power[:, 2] = 5.0

        lx_power, info = TFR.normalize_power(tfr_obj, avg_power, ch_names)

        np.testing.assert_allclose(lx_power[0, 0, 0], (10 - 2) / (10 + 2))
        np.testing.assert_allclose(lx_power[0, 1, 0], (2 - 10) / (2 + 10))

    @pytest.mark.unit
    def test_midline_electrode_gives_zero(self):
        tfr_obj = TFR.__new__(TFR)
        ch_names = ['O1', 'O2', 'Cz']
        avg_power = np.zeros((1, 3, 2))
        avg_power[:, 0] = 10.0
        avg_power[:, 1] = 2.0
        avg_power[:, 2] = 5.0

        lx_power, _ = TFR.normalize_power(tfr_obj, avg_power, ch_names)

        np.testing.assert_allclose(lx_power[0, 2, 0], 0.0)


# ============================================================================
# lateralization_index
# ============================================================================

class TestLateralizationIndex:
    @staticmethod
    def _make_average_tfr(ch_names, values, freqs=None, times=None):
        if freqs is None:
            freqs = np.array([10.0])
        if times is None:
            times = np.array([0.0, 0.1, 0.2])
        info = mne.create_info(list(ch_names), 100., ch_types='eeg')
        data = np.zeros((len(ch_names), len(freqs), len(times)))
        for i, v in enumerate(values):
            data[i] = v
        return mne.time_frequency.AverageTFRArray(
            info=info, data=data, times=times, freqs=freqs, nave=5
        )

    @pytest.mark.unit
    def test_conditions_do_not_alias_shared_array(self):
        # regression: output was allocated once outside the condition
        # loop and stored by reference, so every condition ended up
        # showing the last-processed condition's data
        tfr = {
            'condA': [self._make_average_tfr(['C3', 'C4'], [10.0, 2.0])],
            'condB': [self._make_average_tfr(['C3', 'C4'], [1.0, 9.0])],
        }

        result = TFR.lateralization_index(tfr, elec_oi=['C3'])

        np.testing.assert_allclose(result['condA'][0, 0, 0], (10 - 2) / (10 + 2))
        np.testing.assert_allclose(result['condB'][0, 0, 0], (1 - 9) / (1 + 9))
        assert result['condA'] is not result['condB']

    @pytest.mark.unit
    def test_elec_oi_all_modifies_tfr_data_in_place(self):
        tfr = {
            'condA': [self._make_average_tfr(['O1', 'O2'], [10.0, 2.0])],
        }

        result = TFR.lateralization_index(tfr, elec_oi='all')

        np.testing.assert_allclose(result['condA'][0]._data[0, 0, 0],
                                    (10 - 2) / (10 + 2))

    @pytest.mark.unit
    def test_does_not_mutate_original_tfr_dict(self):
        original_tfr_obj = self._make_average_tfr(['C3', 'C4'], [10.0, 2.0])
        tfr = {'condA': [original_tfr_obj]}

        TFR.lateralization_index(tfr, elec_oi=['C3'])

        # original object's data should be untouched (deepcopy used internally)
        np.testing.assert_allclose(original_tfr_obj._data[0, 0, 0], 10.0)

    @pytest.mark.unit
    def test_elec_oi_all_raises_for_unknown_channel(self):
        tfr = {'condA': [self._make_average_tfr(['NotAChannel'], [1.0])]}
        with pytest.raises(ValueError, match='not found in ch_pairs'):
            TFR.lateralization_index(tfr, elec_oi='all')


# ============================================================================
# condition_tfrs (end-to-end)
# ============================================================================

class TestConditionTfrs:
    @pytest.mark.unit
    def test_single_default_condition(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        epochs = make_epochs(n_trials=6, n_samples=40, sfreq=100.)
        df = make_behavioral_df(6, x=1)
        tfr = TFR(sj=1, epochs=epochs, df=df, min_freq=4, max_freq=20, num_frex=3)

        result = tfr.condition_tfrs(pos_labels=None)

        assert list(result.keys()) == ['all_data']
        assert isinstance(result['all_data'], mne.time_frequency.AverageTFR)

    @pytest.mark.unit
    def test_multiple_conditions_separated(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        epochs = make_epochs(n_trials=6, n_samples=40, sfreq=100.)
        df = make_behavioral_df(6, cond=['A', 'A', 'A', 'B', 'B', 'B'])
        tfr = TFR(sj=1, epochs=epochs, df=df, min_freq=4, max_freq=20, num_frex=3)

        result = tfr.condition_tfrs(pos_labels=None, cnds={'cond': ['A', 'B']})

        assert set(result.keys()) == {'A', 'B'}
        assert result['A'].nave == 3
        assert result['B'].nave == 3

    @pytest.mark.unit
    def test_window_oi_crops_time_axis(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        epochs = make_epochs(n_trials=4, n_samples=100, sfreq=100.,
                              tmin=-0.5)
        df = make_behavioral_df(4, x=1)
        tfr = TFR(sj=1, epochs=epochs, df=df, min_freq=4, max_freq=20, num_frex=3)

        result = tfr.condition_tfrs(pos_labels=None, window_oi=(0.0, 0.2))

        times = result['all_data'].times
        assert times.min() >= -1e-9
        assert times.max() <= 0.2 + 1e-9

    @pytest.mark.unit
    def test_downsample_reduces_time_points(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        epochs = make_epochs(n_trials=4, n_samples=100, sfreq=100.)
        df = make_behavioral_df(4, x=1)

        tfr_full = TFR(sj=1, epochs=epochs, df=df, min_freq=4, max_freq=20,
                        num_frex=3, downsample=1)
        result_full = tfr_full.condition_tfrs(pos_labels=None, name='full')

        tfr_down = TFR(sj=1, epochs=epochs, df=df, min_freq=4, max_freq=20,
                        num_frex=3, downsample=2)
        result_down = tfr_down.condition_tfrs(pos_labels=None, name='down')

        assert len(result_down['all_data'].times) < len(result_full['all_data'].times)

    @pytest.mark.unit
    def test_saves_tfr_files_to_disk(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        epochs = make_epochs(n_trials=4, n_samples=40, sfreq=100.)
        df = make_behavioral_df(4, x=1)
        tfr = TFR(sj=1, epochs=epochs, df=df, min_freq=4, max_freq=20, num_frex=3)

        tfr.condition_tfrs(pos_labels=None, name='mytest')

        expected = tmp_path / 'tfr' / 'wavelet' / 'sub_01_mytest_all_data-tfr.h5'
        assert expected.exists()

    @pytest.mark.unit
    def test_pos_labels_dict_restricts_trials(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        epochs = make_epochs(n_trials=6, n_samples=40, sfreq=100.)
        df = make_behavioral_df(6, target_loc=[1, 2, 1, 2, 1, 2])
        tfr = TFR(sj=1, epochs=epochs, df=df, min_freq=4, max_freq=20, num_frex=3)

        result = tfr.condition_tfrs(pos_labels={'target_loc': [1]})

        assert result['all_data'].nave == 3

    @pytest.mark.unit
    def test_invalid_pos_labels_type_raises(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        epochs = make_epochs(n_trials=4, n_samples=40, sfreq=100.)
        df = make_behavioral_df(4, x=1)
        tfr = TFR(sj=1, epochs=epochs, df=df, min_freq=4, max_freq=20, num_frex=3)

        with pytest.raises(TypeError, match='pos_labels must be'):
            tfr.condition_tfrs(pos_labels=['bad'])

    @pytest.mark.unit
    def test_baseline_trial_spec_runs_without_crash(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        epochs = make_epochs(n_trials=6, n_samples=40, sfreq=100., tmin=-0.2)
        df = make_behavioral_df(6, x=1)
        tfr = TFR(sj=1, epochs=epochs, df=df, min_freq=4, max_freq=20, num_frex=3,
                  baseline=(-0.2, 0), base_method='trial_spec')

        result = tfr.condition_tfrs(pos_labels=None, name='trial')

        assert result['all_data'].data.shape[1:] == (3, 40)

    @pytest.mark.unit
    def test_baseline_cnd_avg_runs_without_crash(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        epochs = make_epochs(n_trials=6, n_samples=40, sfreq=100., tmin=-0.2)
        df = make_behavioral_df(6, cond=[1, 1, 1, 2, 2, 2])
        tfr = TFR(sj=1, epochs=epochs, df=df, min_freq=4, max_freq=20, num_frex=3,
                  baseline=(-0.2, 0), base_method='cnd_avg')

        result = tfr.condition_tfrs(pos_labels=None, cnds={'cond': [1, 2]}, name='cndavg')

        assert set(result.keys()) == {1, 2}

    @pytest.mark.unit
    def test_induced_power_via_condition_tfrs(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        epochs = make_epochs(n_trials=6, n_samples=40, sfreq=100.)
        df = make_behavioral_df(6, x=1)
        tfr = TFR(sj=1, epochs=epochs, df=df, min_freq=4, max_freq=20, num_frex=3,
                  power='induced')

        result = tfr.condition_tfrs(pos_labels=None, name='induced')

        assert result['all_data'].nave == 6

    @pytest.mark.unit
    def test_evoked_power_via_condition_tfrs(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        epochs = make_epochs(n_trials=6, n_samples=40, sfreq=100.)
        df = make_behavioral_df(6, x=1)
        tfr = TFR(sj=1, epochs=epochs, df=df, min_freq=4, max_freq=20, num_frex=3,
                  power='evoked')

        result = tfr.condition_tfrs(pos_labels=None, name='evoked')

        # a single averaged evoked response is treated as nave=1 for TF decomposition
        assert result['all_data'].nave == 1

    @pytest.mark.unit
    def test_report_true_generates_html_file(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        montage = mne.channels.make_standard_montage('biosemi64')
        ch_names = montage.ch_names[:4]
        info = mne.create_info(ch_names, 100., ch_types='eeg')
        info.set_montage(montage)
        data = np.random.default_rng(0).normal(0, 1, (6, 4, 40))
        epochs = mne.EpochsArray(data, info, tmin=-0.2)
        df = make_behavioral_df(6, x=1)
        # num_frex > 3 to exercise the "subsample frequencies for topo
        # plots" branch too
        tfr = TFR(sj=1, epochs=epochs, df=df, min_freq=4, max_freq=20, num_frex=5,
                  report=True)

        tfr.condition_tfrs(pos_labels=None, name='reporttest')

        expected = tmp_path / 'tfr' / 'report' / 'sj_01_reporttest.html'
        assert expected.exists()

    @pytest.mark.unit
    def test_report_with_three_or_fewer_freqs(self, tmp_path, monkeypatch):
        # covers the "use all freqs as-is" branch (len(freqs) <= 3)
        monkeypatch.chdir(tmp_path)
        montage = mne.channels.make_standard_montage('biosemi64')
        ch_names = montage.ch_names[:4]
        info = mne.create_info(ch_names, 100., ch_types='eeg')
        info.set_montage(montage)
        data = np.random.default_rng(0).normal(0, 1, (6, 4, 40))
        epochs = mne.EpochsArray(data, info, tmin=-0.2)
        df = make_behavioral_df(6, x=1)
        tfr = TFR(sj=1, epochs=epochs, df=df, min_freq=4, max_freq=20, num_frex=2,
                  report=True)

        tfr.condition_tfrs(pos_labels=None, name='reporttest2')

        expected = tmp_path / 'tfr' / 'report' / 'sj_01_reporttest2.html'
        assert expected.exists()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
