"""
Test suite for open_dvm.support.synthetic_data.

Organization
------------
- TestMakeConditionEvokeds: synthetic ERP data (list of mne.EvokedArray)
- TestMakeAverageTfr: synthetic TFR data (list of AverageTFRArray)
- TestMakeBdmResult: synthetic BDM decoding results (1d/gat/tfr shapes)
- TestMakeCtfResult: synthetic CTF reconstruction results (1d/gat/tfr shapes)
"""

import mne
import numpy as np
import pytest

from open_dvm.support.synthetic_data import (
    make_average_tfr,
    make_bdm_result,
    make_condition_evokeds,
    make_ctf_result,
)

mne.set_log_level("ERROR")


class TestMakeConditionEvokeds:
    @pytest.mark.unit
    def test_returns_one_evoked_per_subject(self):
        evokeds = make_condition_evokeds(n_subjects=7)
        assert len(evokeds) == 7
        assert all(isinstance(e, mne.EvokedArray) for e in evokeds)

    @pytest.mark.unit
    def test_effect_confined_to_window_contra_gt_ipsi(self):
        evokeds = make_condition_evokeds(
            ch_names=["C3", "C4"],
            contra_ch=["C3"],
            ipsi_ch=["C4"],
            contra_amp=6e-6,
            ipsi_amp=1e-6,
            baseline=0.0,
            noise_sd=1e-8,
            effect_window=(20, 30),
            n_samples=50,
            n_subjects=10,
        )
        data = np.stack([e.data for e in evokeds])  # (n_sub, n_ch, n_times)
        mean_data = data.mean(axis=0)

        # inside the window: contra (ch 0) clearly above ipsi (ch 1)
        assert mean_data[0, 20:30].mean() > mean_data[1, 20:30].mean()
        # outside the window: both near baseline
        assert abs(mean_data[0, :20].mean()) < 1e-6
        assert abs(mean_data[1, 40:].mean()) < 1e-6

    @pytest.mark.unit
    def test_seed_reproducible(self):
        a = make_condition_evokeds(seed=42, n_subjects=3)
        b = make_condition_evokeds(seed=42, n_subjects=3)
        for ea, eb in zip(a, b):
            np.testing.assert_array_equal(ea.data, eb.data)


class TestMakeAverageTfr:
    @pytest.mark.unit
    def test_returns_one_tfr_per_subject(self):
        tfrs = make_average_tfr(n_subjects=6)
        assert len(tfrs) == 6
        assert all(isinstance(t, mne.time_frequency.AverageTFRArray) for t in tfrs)

    @pytest.mark.unit
    def test_effect_confined_to_time_and_freq_window(self):
        freqs = np.linspace(4, 30, 10)
        tfrs = make_average_tfr(
            ch_names=["C3", "C4"],
            contra_ch=["C3"],
            ipsi_ch=["C4"],
            freqs=freqs,
            contra_amp=3.0,
            ipsi_amp=0.3,
            baseline=0.0,
            noise_sd=1e-6,
            effect_window=(20, 30),
            effect_freq_range=(8, 12),
            n_samples=50,
            n_subjects=8,
        )
        data = np.stack([t.data for t in tfrs])  # (n_sub, n_ch, n_freq, n_times)
        mean_data = data.mean(axis=0)
        freq_mask = (freqs >= 8) & (freqs <= 12)

        # inside the time/freq box: contra > ipsi
        contra_in_box = mean_data[0][freq_mask][:, 20:30].mean()
        ipsi_in_box = mean_data[1][freq_mask][:, 20:30].mean()
        assert contra_in_box > ipsi_in_box
        # outside the freq range (same time window): near baseline
        assert abs(mean_data[0][~freq_mask][:, 20:30].mean()) < 1e-3

    @pytest.mark.unit
    def test_seed_reproducible(self):
        a = make_average_tfr(seed=1, n_subjects=3)
        b = make_average_tfr(seed=1, n_subjects=3)
        for ta, tb in zip(a, b):
            np.testing.assert_array_equal(ta.data, tb.data)


class TestMakeBdmResult:
    @pytest.mark.unit
    def test_1d_shape_and_effect_window(self):
        bdms = make_bdm_result(
            effect_window=(10, 20),
            peak_auc=0.9,
            chance_level=0.5,
            noise_sd=1e-6,
            n_subjects=10,
            n_samples=40,
            cnd="A",
        )
        assert len(bdms) == 10
        scores = np.stack([b["A"]["dec_scores"] for b in bdms])
        assert scores.shape == (10, 40)
        # core plateau of the ramped window (envelope == 1), not the
        # ramped edges -- see TestMakeBdmResult module docstring note
        assert scores[:, 13:17].mean() == pytest.approx(0.9, abs=0.01)
        assert scores[:, :10].mean() == pytest.approx(0.5, abs=0.01)
        assert "times" in bdms[0]["info"]

    @pytest.mark.unit
    def test_gat_shape_confines_effect_to_block(self):
        bdms = make_bdm_result(
            effect_window=(10, 20),
            peak_auc=0.9,
            chance_level=0.5,
            noise_sd=1e-6,
            n_subjects=6,
            n_samples=30,
            shape="gat",
        )
        scores = np.stack([b["A"]["dec_scores"] for b in bdms])
        assert scores.shape == (6, 30, 30)
        mean_scores = scores.mean(axis=0)
        assert mean_scores[13:17, 13:17].mean() == pytest.approx(0.9, abs=0.01)
        assert mean_scores[0:10, 0:10].mean() == pytest.approx(0.5, abs=0.01)
        assert "test_times" in bdms[0]["info"]

    @pytest.mark.unit
    def test_tfr_shape_confines_effect_to_freq_and_time(self):
        bdms = make_bdm_result(
            effect_window=(10, 20),
            peak_auc=0.9,
            chance_level=0.5,
            noise_sd=1e-6,
            n_subjects=6,
            n_samples=30,
            shape="tfr",
            freqs=np.linspace(4, 30, 5),
            effect_freq_idx=2,
        )
        scores = np.stack([b["A"]["dec_scores"] for b in bdms])
        assert scores.shape == (6, 5, 30)
        mean_scores = scores.mean(axis=0)
        assert mean_scores[2, 13:17].mean() == pytest.approx(0.9, abs=0.01)
        assert mean_scores[0, 10:20].mean() == pytest.approx(0.5, abs=0.01)
        assert "freqs" in bdms[0]["info"]

    @pytest.mark.unit
    def test_invalid_shape_raises(self):
        with pytest.raises(ValueError, match="shape"):
            make_bdm_result(shape="bogus")

    @pytest.mark.unit
    def test_seed_reproducible(self):
        a = make_bdm_result(seed=5, n_subjects=3)
        b = make_bdm_result(seed=5, n_subjects=3)
        for ba, bb in zip(a, b):
            np.testing.assert_array_equal(ba["A"]["dec_scores"], bb["A"]["dec_scores"])

    @pytest.mark.unit
    def test_explicit_times_overrides_n_samples(self):
        times = np.linspace(-0.1, 0.9, 25)
        bdms = make_bdm_result(times=times, n_subjects=2)
        np.testing.assert_array_equal(bdms[0]["info"]["times"], times)
        assert bdms[0]["A"]["dec_scores"].shape == (25,)

    @pytest.mark.unit
    def test_tfr_shape_default_freqs_and_effect_idx(self):
        bdms = make_bdm_result(shape="tfr", n_subjects=2, n_samples=20)
        assert bdms[0]["info"]["freqs"].shape == (15,)
        assert bdms[0]["A"]["dec_scores"].shape == (15, 20)


class TestMakeCtfResult:
    @pytest.mark.unit
    def test_1d_shape_and_effect_window(self):
        ctfs = make_ctf_result(
            effect_window=(10, 20),
            peak_slope=0.02,
            baseline_slope=0.0,
            noise_sd=1e-5,
            n_subjects=10,
            n_samples=40,
            cnd="A",
        )
        assert len(ctfs) == 10
        slopes = np.stack([c["A"]["raw_slopes"] for c in ctfs])
        assert slopes.shape == (10, 1, 40)  # single 'all' band
        # core plateau of the ramped window (envelope == 1), not the
        # ramped edges
        assert slopes[:, 0, 13:17].mean() == pytest.approx(0.02, abs=0.001)
        assert slopes[:, 0, :10].mean() == pytest.approx(0.0, abs=0.001)
        assert ctfs[0]["info"]["bands"] == ["all"]

    @pytest.mark.unit
    def test_gat_shape_confines_effect_to_block(self):
        ctfs = make_ctf_result(
            effect_window=(10, 20),
            peak_slope=0.02,
            baseline_slope=0.0,
            noise_sd=1e-5,
            n_subjects=6,
            n_samples=30,
            shape="gat",
        )
        slopes = np.stack([c["A"]["raw_slopes"] for c in ctfs])
        assert slopes.shape == (6, 1, 30, 30)
        mean_slopes = slopes.mean(axis=0)[0]
        assert mean_slopes[13:17, 13:17].mean() == pytest.approx(0.02, abs=0.001)
        assert mean_slopes[0:10, 0:10].mean() == pytest.approx(0.0, abs=0.001)

    @pytest.mark.unit
    def test_gat_shape_rejects_multiple_bands(self):
        with pytest.raises(ValueError, match="single band"):
            make_ctf_result(shape="gat", bands=["theta", "alpha"])

    @pytest.mark.unit
    def test_tfr_shape_confines_effect_to_band_and_time(self):
        ctfs = make_ctf_result(
            effect_window=(10, 20),
            peak_slope=0.02,
            baseline_slope=0.0,
            noise_sd=1e-5,
            n_subjects=6,
            n_samples=30,
            shape="tfr",
            effect_band_idx=1,
        )
        slopes = np.stack([c["A"]["raw_slopes"] for c in ctfs])
        assert slopes.shape == (6, 4, 30)  # default 4 bands
        mean_slopes = slopes.mean(axis=0)
        assert mean_slopes[1, 13:17].mean() == pytest.approx(0.02, abs=0.001)
        assert mean_slopes[0, 10:20].mean() == pytest.approx(0.0, abs=0.001)
        assert "freqs" in ctfs[0]["info"]

    @pytest.mark.unit
    def test_invalid_shape_raises(self):
        with pytest.raises(ValueError, match="shape"):
            make_ctf_result(shape="bogus")

    @pytest.mark.unit
    def test_seed_reproducible(self):
        a = make_ctf_result(seed=7, n_subjects=3)
        b = make_ctf_result(seed=7, n_subjects=3)
        for ca, cb in zip(a, b):
            np.testing.assert_array_equal(ca["A"]["raw_slopes"], cb["A"]["raw_slopes"])

    @pytest.mark.unit
    def test_explicit_times_overrides_n_samples(self):
        times = np.linspace(-0.1, 0.9, 25)
        ctfs = make_ctf_result(times=times, n_subjects=2)
        np.testing.assert_array_equal(ctfs[0]["info"]["times"], times)
        assert ctfs[0]["A"]["raw_slopes"].shape == (1, 25)

    @pytest.mark.unit
    def test_tfr_shape_default_bands_and_effect_idx(self):
        ctfs = make_ctf_result(shape="tfr", n_subjects=2, n_samples=20)
        assert ctfs[0]["info"]["bands"] == [[4, 8], [8, 12], [12, 20], [20, 30]]
        assert ctfs[0]["A"]["raw_slopes"].shape == (4, 20)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
