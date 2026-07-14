"""
Test suite for open_dvm.analysis.EEG.

Organization
------------
- TestRawFileDispatch: __init__ file-type detection and reader dispatch
- TestRawReplaceChannel: replace_channel electrode substitution
- TestRawRereference: rereference voltage scaling + reference removal
- TestRawConfigureMontage: configure_montage renaming + montage application
- TestRawSelectEvents: select_events trigger detection
- TestRawReportRaw: report_raw smoke test
- TestEpochsInit: filter-padding math, sj/session formatting
- TestEpochsAlignMetaData: behavior/EEG trial-count alignment (all branches)
- TestEpochsAlignEyeData: EOG bipolar-derivation half (no eye-tracker files)
- TestEpochsAddChannelData: channel append + resample + time placement
- TestEpochsReportEpochs: smoke test
- TestEpochsSavePreprocessed: single/multi-session save + combine
- TestEpochsBaselineByCondition: per-condition baseline correction math
- TestEpochsShiftByCondition: per-condition circular time-shift math
- TestArtefactRejectInit: attribute storage
- TestFitICA: n_components math, invalid method, fit_params per method
- TestApplyICA: exclude=[] vs exclude=[k]
- TestAutomatedIcaBlinkSelection: zero-EOG regression + has-EOG path
- TestFiltPad: edge-mean padding math
- TestBoxSmoothing: boxcar smoothing + in-place mutation
- TestMarkBads: artifact-segment boundary detection (all edge cases)
- TestZScoreData: z-score/threshold math, mask + filter_z branches
- TestPreprocessEpochs: mutable-default and z_thresh-forwarding regressions
- TestApplyHilbert: envelope extraction shape/non-mutation
- TestAutoRepairNoise: full pipeline, z_thresh forwarding, metadata-None regression
- TestUpdateHeatMap: heat_map/cleaned_info state mutation
- TestPlottingSmokeTests: plot_heat_map/plot_hist_auto_repair/plot_z_score_epochs/plot_auto_repair
- TestRunBlinkICA: end-to-end with mocked input()/time.sleep()
"""

import tempfile
import os
import pickle
import time

import pytest
import numpy as np
import pandas as pd
import mne
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from unittest.mock import patch

from open_dvm.analysis.EEG import RAW, Epochs, ArtefactReject
from tests.fixtures.eeg_sample_data import (
    make_synthetic_raw,
    make_synthetic_raw_with_stim,
    make_artefact_epochs,
    make_eog_correlated_raw,
    make_synthetic_epochs,
    write_behavioral_csv,
)


# ============================================================================
# RAW.__init__ / from_* classmethods: file-type detection and dispatch
# ============================================================================

class TestRawFileDispatch:
    """Tests for file-type auto-detection and reader dispatch."""

    @pytest.mark.unit
    def test_fif_round_trip_preserves_data(self, tmp_path):
        raw = make_synthetic_raw(['F1', 'F2', 'Cz'], 'eeg', sfreq=250, n_samples=500)
        path = str(tmp_path / 'test_raw.fif')
        mne.io.RawArray(raw.get_data(), raw.info).save(path)

        loaded = RAW(path)

        assert isinstance(loaded, RAW)
        assert loaded.ch_names == raw.ch_names
        np.testing.assert_allclose(loaded.get_data(), raw.get_data())

    @pytest.mark.unit
    def test_from_fif_classmethod(self, tmp_path):
        raw = make_synthetic_raw(['F1', 'F2'], 'eeg', sfreq=250, n_samples=200)
        path = str(tmp_path / 'test_raw.fif')
        mne.io.RawArray(raw.get_data(), raw.info).save(path)

        loaded = RAW.from_fif(path)

        assert loaded.ch_names == raw.ch_names

    @pytest.mark.unit
    def test_unknown_extension_raises_value_error(self):
        with pytest.raises(ValueError, match="Cannot determine file type"):
            RAW('/tmp/does_not_exist.xyz')

    @staticmethod
    def _stub_raw():
        info = mne.create_info(['A1', 'A2'], 250, ch_types='eeg')
        return mne.io.RawArray(np.zeros((2, 100)), info)

    @pytest.mark.unit
    @pytest.mark.parametrize('ext,reader_target', [
        ('.bdf', 'mne.io.read_raw_bdf'),
        ('.edf', 'mne.io.read_raw_edf'),
        ('.vhdr', 'mne.io.read_raw_brainvision'),
        ('.cnt', 'mne.io.read_raw_cnt'),
        ('.set', 'mne.io.read_raw_eeglab'),
    ])
    def test_extension_dispatches_to_correct_reader(self, ext, reader_target):
        """Each supported extension must route to its corresponding MNE
        reader function -- verified via mocking rather than fabricating
        real binary files in each proprietary format."""
        with patch(reader_target) as mock_reader:
            mock_reader.return_value = self._stub_raw()
            RAW(f'/tmp/fake{ext}')
            assert mock_reader.called

    @pytest.mark.unit
    def test_eog_kwarg_forwarded_to_reader(self):
        with patch('mne.io.read_raw_bdf') as mock_reader:
            mock_reader.return_value = self._stub_raw()
            RAW('/tmp/fake.bdf', eog=['EOG1', 'EOG2'])
            assert mock_reader.call_args.kwargs['eog'] == ['EOG1', 'EOG2']

    @pytest.mark.unit
    def test_explicit_file_type_overrides_extension(self):
        """file_type can override extension-based auto-detection."""
        with patch('mne.io.read_raw_fif') as mock_reader:
            mock_reader.return_value = self._stub_raw()
            RAW('/tmp/fake.bdf', file_type='fif')
            assert mock_reader.called


# ============================================================================
# replace_channel
# ============================================================================

class TestRawReplaceChannel:
    """Tests for replace_channel electrode substitution."""

    @pytest.mark.unit
    def test_replaces_data_and_drops_replacement_channel(self):
        raw = make_synthetic_raw(['F1', 'EXG7', 'F2'], 'eeg', n_samples=50)
        exg7_data = raw.get_data(picks='EXG7').copy()

        result = raw.replace_channel({'F1': 'EXG7'})

        np.testing.assert_allclose(raw.get_data(picks='F1'), exg7_data)
        assert 'EXG7' not in raw.ch_names
        assert result is raw

    @pytest.mark.unit
    def test_missing_channel_warns_but_does_not_raise(self, capsys):
        raw = make_synthetic_raw(['F1', 'F2'], 'eeg', n_samples=50)

        raw.replace_channel({'F1': 'NONEXISTENT'})

        assert raw.ch_names == ['F1', 'F2']
        assert 'Warning' in capsys.readouterr().out

    @pytest.mark.unit
    def test_empty_dict_returns_immediately_unchanged(self):
        raw = make_synthetic_raw(['F1'], 'eeg', n_samples=50)
        before = raw.get_data().copy()

        result = raw.replace_channel({})

        assert result is raw
        np.testing.assert_array_equal(raw.get_data(), before)


# ============================================================================
# rereference
# ============================================================================

class TestRawRereference:
    """Tests for rereference voltage scaling and reference-channel removal."""

    @pytest.mark.unit
    def test_change_voltage_scales_by_1e6(self):
        # a zero-valued reference channel isolates the voltage-scaling
        # effect from the (data-altering) referencing effect
        rng = np.random.default_rng(1)
        f1_data = rng.normal(0, 1e-5, size=(1, 100))
        data = np.vstack([f1_data, np.zeros((1, 100))])
        raw = make_synthetic_raw(['F1', 'RefZero'], ['eeg', 'eeg'], data=data)

        raw.rereference(ref_channels=['RefZero'], change_voltage=True)

        np.testing.assert_allclose(
            raw.get_data(picks=['F1']), f1_data * 1e6, atol=1e-12
        )

    @pytest.mark.unit
    def test_change_voltage_false_leaves_scale_untouched(self):
        rng = np.random.default_rng(1)
        f1_data = rng.normal(0, 1e-5, size=(1, 100))
        data = np.vstack([f1_data, np.zeros((1, 100))])
        raw = make_synthetic_raw(['F1', 'RefZero'], ['eeg', 'eeg'], data=data)

        raw.rereference(ref_channels=['RefZero'], change_voltage=False)

        np.testing.assert_allclose(raw.get_data(picks=['F1']), f1_data, atol=1e-12)

    @pytest.mark.unit
    def test_specific_reference_channels_are_removed(self):
        raw = make_synthetic_raw(['F1', 'F2', 'Ref1', 'Ref2'], 'eeg', n_samples=50)

        raw.rereference(ref_channels=['Ref1', 'Ref2'], change_voltage=False)

        assert raw.ch_names == ['F1', 'F2']

    @pytest.mark.unit
    def test_average_reference_does_not_remove_channels(self):
        raw = make_synthetic_raw(['F1', 'F2', 'F3'], 'eeg', n_samples=50)

        raw.rereference(ref_channels='average', change_voltage=False)

        assert raw.ch_names == ['F1', 'F2', 'F3']

    @pytest.mark.unit
    def test_to_remove_missing_channel_silently_skipped(self):
        raw = make_synthetic_raw(['F1', 'F2'], 'eeg', n_samples=50)

        raw.rereference(
            ref_channels='average', change_voltage=False,
            to_remove=['DOES_NOT_EXIST']
        )

        assert raw.ch_names == ['F1', 'F2']

    @pytest.mark.unit
    def test_to_remove_combined_with_ref_channels(self):
        raw = make_synthetic_raw(
            ['F1', 'F2', 'Ref1', 'EXG7'], 'eeg', n_samples=50
        )

        raw.rereference(
            ref_channels=['Ref1'], change_voltage=False, to_remove=['EXG7']
        )

        assert raw.ch_names == ['F1', 'F2']


# ============================================================================
# configure_montage
# ============================================================================

class TestRawConfigureMontage:
    """Tests for configure_montage BioSemi renaming and montage application."""

    @pytest.mark.unit
    def test_biosemi_channels_renamed_to_montage_names(self):
        montage = mne.channels.make_standard_montage('biosemi32')
        biosemi_names = [f'A{i}' for i in range(1, 17)] + [f'B{i}' for i in range(1, 17)]
        raw = make_synthetic_raw(biosemi_names, 'eeg', n_samples=50)

        raw.configure_montage(montage='biosemi32')

        assert raw.ch_names == montage.ch_names
        assert raw.get_montage() is not None

    @pytest.mark.unit
    def test_ch_remove_applied_before_montage(self):
        biosemi_names = [f'A{i}' for i in range(1, 17)] + [f'B{i}' for i in range(1, 17)]
        raw = make_synthetic_raw(biosemi_names + ['EXG7', 'EXG8'], 'eeg', n_samples=50)

        raw.configure_montage(montage='biosemi32', ch_remove=['EXG7', 'EXG8'])

        assert 'EXG7' not in raw.ch_names
        assert 'EXG8' not in raw.ch_names

    @pytest.mark.unit
    def test_non_biosemi_names_unchanged(self):
        raw = make_synthetic_raw(['Fp1', 'Fp2', 'Cz'], 'eeg', n_samples=50)

        raw.configure_montage(montage='standard_1020')

        assert raw.ch_names == ['Fp1', 'Fp2', 'Cz']
        assert raw.get_montage() is not None

    @pytest.mark.unit
    def test_accepts_dig_montage_object_directly(self):
        raw = make_synthetic_raw(['Fp1', 'Fp2', 'Cz'], 'eeg', n_samples=50)
        montage_obj = mne.channels.make_standard_montage('standard_1020')

        raw.configure_montage(montage=montage_obj)

        assert raw.get_montage() is not None


# ============================================================================
# select_events
# ============================================================================

class TestRawSelectEvents:
    """Tests for select_events trigger detection."""

    @pytest.mark.unit
    def test_detects_events_at_expected_samples(self):
        raw = make_synthetic_raw_with_stim([(100, 1), (300, 2), (500, 1)])

        events = raw.select_events(event_id=[1, 2])

        np.testing.assert_array_equal(
            events, [[100, 0, 1], [300, 0, 2], [500, 0, 1]]
        )

    @pytest.mark.unit
    def test_event_id_filters_unwanted_triggers(self):
        raw = make_synthetic_raw_with_stim([(100, 1), (300, 99)])

        events = raw.select_events(event_id=[1])

        np.testing.assert_array_equal(events, [[100, 0, 1]])

    @pytest.mark.unit
    def test_consecutive_false_removes_duplicate_events(self):
        """Two same-valued triggers with no other event between them are
        collapsed to the later one when consecutive=False (the default
        semantics of mne.find_events)."""
        raw = make_synthetic_raw_with_stim([(100, 1), (120, 1), (300, 2)])

        events_dup = raw.select_events(event_id=[1, 2], consecutive=True)
        events_nodup = raw.select_events(event_id=[1, 2], consecutive=False)

        np.testing.assert_array_equal(
            events_dup, [[100, 0, 1], [120, 0, 1], [300, 0, 2]]
        )
        np.testing.assert_array_equal(events_nodup, [[120, 0, 1], [300, 0, 2]])

    @pytest.mark.unit
    def test_binary_offset_correction_is_temporary(self):
        """binary subtracts a constant offset to recover clean trigger
        codes (e.g. BioSemi's status-channel convention, where the stim
        channel rests at a nonzero baseline equal to the offset itself
        rather than at 0) -- and must restore the original data after
        detection."""
        binary_offset = 3840
        raw = make_synthetic_raw_with_stim(
            [(100, binary_offset + 1), (300, binary_offset + 2)],
            baseline_value=binary_offset,
        )
        stim_before = raw.get_data(picks='STI').copy()

        events = raw.select_events(event_id=[1, 2], binary=binary_offset)

        np.testing.assert_array_equal(events, [[100, 0, 1], [300, 0, 2]])
        np.testing.assert_array_equal(raw.get_data(picks='STI'), stim_before)

    @pytest.mark.unit
    def test_no_stim_channel_raises_value_error(self):
        raw = make_synthetic_raw(['F1', 'F2'], 'eeg', n_samples=50)

        with pytest.raises(ValueError):
            raw.select_events()

    @pytest.mark.unit
    def test_explicit_stim_channel_not_found_raises_value_error(self):
        raw = make_synthetic_raw_with_stim([(100, 1)])

        with pytest.raises(ValueError, match="not found in data"):
            raw.select_events(stim_channel='NOPE')


# ============================================================================
# report_raw
# ============================================================================

class TestRawReportRaw:
    """Smoke test for report_raw."""

    @pytest.mark.unit
    def test_adds_content_and_returns_same_report(self):
        raw = make_synthetic_raw_with_stim([(100, 1), (300, 2)])
        events = np.array([[100, 0, 1], [300, 0, 2]])
        report = mne.Report(title='test')
        n_before = len(report._content)

        out = raw.report_raw(report, events, event_id=[1, 2])

        assert out is report
        assert len(report._content) > n_before


# ============================================================================
# Epochs.__init__: filter-padding math
# ============================================================================

class TestEpochsInit:
    """Tests for filter-padding math and subject/session formatting."""

    @pytest.mark.unit
    def test_float_flt_pad_expands_symmetrically(self):
        epochs = make_synthetic_epochs(
            [1, 1], event_id=1, tmin=-0.2, tmax=0.5, flt_pad=0.3
        )

        assert epochs.tmin == pytest.approx(-0.5)
        assert epochs.tmax == pytest.approx(0.8)
        assert epochs.flt_pad == 0.3

    @pytest.mark.unit
    def test_tuple_flt_pad_expands_asymmetrically(self):
        epochs = make_synthetic_epochs(
            [1, 1], event_id=1, tmin=-0.2, tmax=0.5, flt_pad=(0.1, 0.4)
        )

        assert epochs.tmin == pytest.approx(-0.3)
        assert epochs.tmax == pytest.approx(0.9)

    @pytest.mark.unit
    def test_no_flt_pad_leaves_window_unchanged(self):
        epochs = make_synthetic_epochs(
            [1, 1], event_id=1, tmin=-0.2, tmax=0.5, flt_pad=None
        )

        assert epochs.tmin == pytest.approx(-0.2)
        assert epochs.tmax == pytest.approx(0.5)
        assert epochs.flt_pad is None

    @pytest.mark.unit
    def test_subject_and_session_are_zero_padded(self):
        epochs = make_synthetic_epochs([1, 1], event_id=1, sj=3, session=7)

        assert epochs.sj == '03'
        assert epochs.session == '07'


# ============================================================================
# align_meta_data
# ============================================================================

class TestEpochsAlignMetaData:
    """Tests for behavioral/EEG trial-count alignment."""

    @pytest.mark.unit
    def test_perfect_match(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        epochs = make_synthetic_epochs([1, 2, 1, 2], event_id=[1, 2])
        events = epochs.events
        write_behavioral_csv(1, 1, pd.DataFrame({
            'trigger': [1, 2, 1, 2], 'nr_trials': [1, 2, 3, 4],
        }))

        missing, report = epochs.align_meta_data(
            events, trigger_header='trigger', beh_oi=['trigger', 'nr_trials']
        )

        assert len(missing) == 0
        assert len(epochs) == 4
        np.testing.assert_array_equal(epochs.metadata['nr_trials'], [1, 2, 3, 4])

    @pytest.mark.unit
    def test_behavior_has_extra_trailing_trials(self, tmp_path, monkeypatch):
        """nr_miss > 0: behavior has more trials than EEG events."""
        monkeypatch.chdir(tmp_path)
        epochs = make_synthetic_epochs([1, 2, 1], event_id=[1, 2])
        events = epochs.events
        write_behavioral_csv(1, 1, pd.DataFrame({
            'trigger': [1, 2, 1, 2], 'nr_trials': [1, 2, 3, 4],
        }))

        missing, report = epochs.align_meta_data(
            events, trigger_header='trigger', beh_oi=['trigger', 'nr_trials']
        )

        assert list(missing) == [4]
        assert len(epochs) == 3
        np.testing.assert_array_equal(epochs.metadata['nr_trials'], [1, 2, 3])

    @pytest.mark.unit
    def test_eeg_has_extra_trailing_events(self, tmp_path, monkeypatch):
        """nr_miss < 0, fast path: excess EEG events are simply trailing."""
        monkeypatch.chdir(tmp_path)
        epochs = make_synthetic_epochs([1, 2, 1, 2, 1], event_id=[1, 2])
        events = epochs.events
        write_behavioral_csv(1, 1, pd.DataFrame({
            'trigger': [1, 2, 1, 2], 'nr_trials': [1, 2, 3, 4],
        }))

        missing, report = epochs.align_meta_data(
            events, trigger_header='trigger', beh_oi=['trigger', 'nr_trials']
        )

        assert len(epochs) == 4
        np.testing.assert_array_equal(epochs.metadata['nr_trials'], [1, 2, 3, 4])

    @pytest.mark.unit
    def test_eeg_has_extra_interleaved_events(self, tmp_path, monkeypatch):
        """Regression test: nr_miss < 0 with interleaved (non-trailing)
        excess EEG events previously never actually dropped anything
        (idx_remove stayed [] due to dead removal logic), which then hit
        a numpy broadcasting error comparing differently-sized arrays.
        EEG triggers [1,2,1,2,1] vs behavior [1,1,2,1]: the excess event
        at original EEG index 1 (value 2, unmatched) must be dropped,
        leaving [1,1,2,1] aligned with behavior."""
        monkeypatch.chdir(tmp_path)
        epochs = make_synthetic_epochs([1, 2, 1, 2, 1], event_id=[1, 2])
        events = epochs.events
        write_behavioral_csv(1, 1, pd.DataFrame({
            'trigger': [1, 1, 2, 1], 'nr_trials': [1, 2, 3, 4],
        }))

        missing, report = epochs.align_meta_data(
            events, trigger_header='trigger', beh_oi=['trigger', 'nr_trials']
        )

        assert len(epochs) == 4
        np.testing.assert_array_equal(epochs.metadata['nr_trials'], [1, 2, 3, 4])
        np.testing.assert_array_equal(
            epochs.metadata['trigger'].values, [1, 1, 2, 1]
        )

    @pytest.mark.unit
    def test_no_behavior_file_found(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        epochs = make_synthetic_epochs([1, 1], event_id=1)
        events = epochs.events

        missing, report = epochs.align_meta_data(events, trigger_header='trigger')

        assert missing.size == 0
        assert report == 'No behavior file found'

    @pytest.mark.unit
    def test_missing_trigger_header_raises(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        epochs = make_synthetic_epochs([1, 1], event_id=1)
        events = epochs.events
        write_behavioral_csv(1, 1, pd.DataFrame({'nr_trials': [1, 2]}))

        with pytest.raises(ValueError, match="not found in behavioral"):
            epochs.align_meta_data(events, trigger_header='trigger')

    @pytest.mark.unit
    def test_missing_beh_oi_column_raises(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        epochs = make_synthetic_epochs([1, 1], event_id=1)
        events = epochs.events
        write_behavioral_csv(1, 1, pd.DataFrame({
            'trigger': [1, 1], 'nr_trials': [1, 2],
        }))

        with pytest.raises(ValueError, match="not found in behavioral"):
            epochs.align_meta_data(
                events, trigger_header='trigger',
                beh_oi=['trigger', 'nonexistent_column']
            )


# ============================================================================
# align_eye_data (EOG-building half only; eye-tracker file parsing is
# out of scope for this pass)
# ============================================================================

class TestEpochsAlignEyeData:
    """Tests for the EOG bipolar-derivation half of align_eye_data."""

    @pytest.mark.unit
    def test_mixed_eeg_and_eog_channels(self, tmp_path, monkeypatch):
        """Regression test: vEOG/hEOG previously required BOTH channels
        to already be type 'eog' (self.copy().pick('eog') excluded any
        'eeg'-typed channel), which broke the method's own documented
        example of using a frontal EEG electrode (e.g. Fp1) as half of
        a bipolar VEOG derivation -- a standard real-world technique."""
        monkeypatch.chdir(tmp_path)
        epochs = make_synthetic_epochs(
            [1, 1], event_id=1,
            ch_names=['Fp1', 'VEOG_lower', 'HEOG_L', 'HEOG_R', 'Cz'],
            ch_types=['eeg', 'eog', 'eog', 'eog', 'eeg'],
        )

        tracker, report = epochs.align_eye_data(
            eye_info=None, missing=np.array([]), nr_epochs=len(epochs),
            vEOG=['Fp1', 'VEOG_lower'], hEOG=['HEOG_L', 'HEOG_R'],
        )

        assert tracker is False
        assert 'VEOG' in epochs.ch_names
        assert 'HEOG' in epochs.ch_names
        fp1 = epochs.get_data(picks=['Fp1'])
        veog_lower = epochs.get_data(picks=['VEOG_lower'])
        veog = epochs.get_data(picks=['VEOG'])
        np.testing.assert_allclose(veog, fp1 - veog_lower, atol=1e-15)

    @pytest.mark.unit
    def test_all_eog_typed_channels_still_works(self, tmp_path, monkeypatch):
        """Regression guard: the fix (broadening the pick to
        ['eeg', 'eog']) must not break the previously-working case where
        both channels are already eog-typed."""
        monkeypatch.chdir(tmp_path)
        epochs = make_synthetic_epochs(
            [1, 1], event_id=1,
            ch_names=['VEOG_u', 'VEOG_l', 'HEOG_L', 'HEOG_R', 'Cz'],
            ch_types=['eog', 'eog', 'eog', 'eog', 'eeg'],
        )

        tracker, report = epochs.align_eye_data(
            eye_info=None, missing=np.array([]), nr_epochs=len(epochs),
            vEOG=['VEOG_u', 'VEOG_l'], hEOG=['HEOG_L', 'HEOG_R'],
        )

        assert 'VEOG' in epochs.ch_names
        assert 'HEOG' in epochs.ch_names

    @pytest.mark.unit
    def test_no_vEOG_or_hEOG_requested(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        epochs = make_synthetic_epochs([1, 1], event_id=1)
        n_ch_before = len(epochs.ch_names)

        tracker, report = epochs.align_eye_data(
            eye_info=None, missing=np.array([]), nr_epochs=len(epochs),
            vEOG=None, hEOG=None,
        )

        assert tracker is False
        assert 'No eye tracker data linked' in report
        assert len(epochs.ch_names) == n_ch_before


# ============================================================================
# add_channel_data
# ============================================================================

class TestEpochsAddChannelData:
    """Tests for channel append + resample + time-placement."""

    @pytest.mark.unit
    def test_places_data_at_correct_time_offset_same_sfreq(self):
        epochs = make_synthetic_epochs(
            [1, 1, 1], event_id=1, sfreq=250, tmin=-0.2, tmax=0.2
        )
        n_new_samples = 50  # 0.2s at 250Hz
        new_data = np.zeros((len(epochs), 1, n_new_samples))
        new_data[:, 0, :] = np.arange(n_new_samples)

        epochs.add_channel_data(new_data, ['TESTCH'], 250, 'misc', t_min=-0.1)

        assert 'TESTCH' in epochs.ch_names
        placed = epochs.get_data(picks=['TESTCH'])[0, 0]
        expected_start = np.argmin(np.abs(epochs.times - (-0.1)))
        assert placed[expected_start] == pytest.approx(0.0)
        assert placed[expected_start + n_new_samples - 1] == pytest.approx(
            n_new_samples - 1
        )

    @pytest.mark.unit
    def test_resamples_to_match_epochs_sfreq(self):
        epochs = make_synthetic_epochs(
            [1, 1, 1], event_id=1, sfreq=250, tmin=-0.2, tmax=0.2
        )
        new_data = np.ones((len(epochs), 1, 100))  # 100 samples @ 500Hz = 0.2s

        epochs.add_channel_data(new_data, ['TESTCH2'], 500, 'misc', t_min=-0.1)

        placed = epochs.get_data(picks=['TESTCH2'])[0, 0]
        # 0.2s at the epochs' own 250Hz sfreq is ~50 samples
        assert np.sum(placed != 0) == pytest.approx(50, abs=1)


# ============================================================================
# report_epochs
# ============================================================================

class TestEpochsReportEpochs:
    """Smoke test for report_epochs."""

    @pytest.mark.unit
    def test_adds_content_and_returns_same_report(self):
        epochs = make_synthetic_epochs([1, 1], event_id=1)
        report = mne.Report(title='test')
        n_before = len(report._content)

        out = epochs.report_epochs(report, title='test title')

        assert out is report
        assert len(report._content) > n_before

    @pytest.mark.unit
    def test_missing_adds_html_block(self):
        epochs = make_synthetic_epochs([1, 1], event_id=1)
        report = mne.Report(title='test')

        out = epochs.report_epochs(report, title='x', missing=np.array([1, 2]))

        assert out is report


# ============================================================================
# save_preprocessed
# ============================================================================

class TestEpochsSavePreprocessed:
    """Tests for save/combine-sessions behavior."""

    @pytest.mark.unit
    def test_session_one_does_not_create_combined_file(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        epochs = make_synthetic_epochs([1, 1, 1], event_id=1, sj=1, session=1)

        epochs.save_preprocessed('clean')

        files = set(os.listdir(tmp_path / 'eeg' / 'processed'))
        assert files == {'sub_01_ses_01_clean-epo.fif'}

    @pytest.mark.unit
    def test_multi_session_combines_into_all_file(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        epochs1 = make_synthetic_epochs(
            [1, 1], event_id=1, sj=1, session=1, seed=1
        )
        epochs1.save_preprocessed('clean', combine_sessions=False)
        epochs2 = make_synthetic_epochs(
            [1, 1, 1], event_id=1, sj=1, session=2, seed=2
        )

        epochs2.save_preprocessed('clean')

        proc_dir = tmp_path / 'eeg' / 'processed'
        combined = mne.read_epochs(str(proc_dir / 'sub_01_all_clean-epo.fif'))
        assert len(combined) == 2 + 3

    @pytest.mark.unit
    def test_eye_data_combined_only_if_all_sessions_have_it(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.chdir(tmp_path)
        epochs1 = make_synthetic_epochs(
            [1, 1], event_id=1, sj=1, session=1, seed=1
        )
        epochs1.save_preprocessed('clean', combine_sessions=False)
        eye_dir = tmp_path / 'eye' / 'processed'
        eye_dir.mkdir(parents=True, exist_ok=True)
        np.savez(
            str(eye_dir / 'sub_01_ses_01_clean.npz'),
            times=np.arange(10), x=np.zeros((2, 10)), y=np.zeros((2, 10)),
            sfreq=1000,
        )
        # session 2 deliberately has no matching eye file
        epochs2 = make_synthetic_epochs(
            [1, 1, 1], event_id=1, sj=1, session=2, seed=2
        )

        epochs2.save_preprocessed('clean')

        assert not (eye_dir / 'sub_01_all_clean.npz').exists()


# ============================================================================
# baseline_by_condition
# ============================================================================

class TestEpochsBaselineByCondition:
    """Tests for per-condition baseline correction."""

    @pytest.mark.unit
    def test_baselines_only_specified_condition(self):
        epochs = make_synthetic_epochs(
            [1, 2, 1, 2], event_id=[1, 2], sfreq=100, tmin=-0.2, tmax=0.2
        )
        n_ch = len(epochs.ch_names)
        # constant-per-trial-per-channel data: trial*10 + channel_idx
        for t in range(len(epochs)):
            for c in range(n_ch):
                epochs._data[t, c, :] = t * 10 + c
        before = epochs._data.copy()
        df = pd.DataFrame({'cnd': ['a', 'b', 'a', 'b']})

        epochs.baseline_by_condition(
            df, cnds=['a'], cnd_header='cnd', base_period=(-0.2, 0), nr_elec=n_ch
        )

        # condition 'a' trials are 0 and 2 (values 0+c, 20+c); baseline
        # mean per electrode = 10+c -> trial0 becomes -10, trial2 becomes +10
        np.testing.assert_allclose(epochs._data[0, :, 0], -10)
        np.testing.assert_allclose(epochs._data[2, :, 0], 10)
        # condition 'b' trials (1, 3) are untouched
        np.testing.assert_array_equal(epochs._data[1], before[1])
        np.testing.assert_array_equal(epochs._data[3], before[3])

    @pytest.mark.unit
    def test_returns_self_for_chaining(self):
        epochs = make_synthetic_epochs([1, 1], event_id=1, sfreq=100)
        df = pd.DataFrame({'cnd': ['a', 'a']})

        result = epochs.baseline_by_condition(df, cnds=['a'], cnd_header='cnd')

        assert result is epochs


# ============================================================================
# shift_by_condition
# ============================================================================

class TestEpochsShiftByCondition:
    """Tests for per-condition circular time-shift."""

    @pytest.mark.unit
    def test_shifts_only_specified_condition_forward(self):
        epochs = make_synthetic_epochs(
            [1, 2], event_id=[1, 2], sfreq=100, tmin=-0.2, tmax=0.2
        )
        epochs._data[:] = 0
        impulse_idx = 20
        epochs._data[:, 0, impulse_idx] = 1.0
        df = pd.DataFrame({'cnd': ['shiftme', 'noshift']})

        epochs.shift_by_condition(df, cnd_info={'shiftme': 0.1}, cnd_header='cnd')

        assert np.argmax(epochs._data[0, 0, :]) == impulse_idx + 10  # +0.1s @ 100Hz
        assert np.argmax(epochs._data[1, 0, :]) == impulse_idx  # untouched

    @pytest.mark.unit
    def test_shifts_backward_for_negative_value(self):
        epochs = make_synthetic_epochs(
            [1, 1], event_id=1, sfreq=100, tmin=-0.2, tmax=0.2
        )
        epochs._data[:] = 0
        impulse_idx = 20
        epochs._data[:, 0, impulse_idx] = 1.0
        df = pd.DataFrame({'cnd': ['fast', 'fast']})

        epochs.shift_by_condition(df, cnd_info={'fast': -0.1}, cnd_header='cnd')

        assert np.argmax(epochs._data[0, 0, :]) == impulse_idx - 10

    @pytest.mark.unit
    def test_returns_self_for_chaining(self):
        epochs = make_synthetic_epochs([1, 1], event_id=1, sfreq=100)
        df = pd.DataFrame({'cnd': ['a', 'a']})

        result = epochs.shift_by_condition(df, cnd_info={'a': 0.05}, cnd_header='cnd')

        assert result is epochs


# ============================================================================
# ArtefactReject.__init__
# ============================================================================

class TestArtefactRejectInit:
    """Tests for attribute storage."""

    @pytest.mark.unit
    def test_stores_all_constructor_args(self):
        ar = ArtefactReject(z_thresh=3.5, max_bad=7, flt_pad=0.2, filter_z=False)

        assert ar.z_thresh == 3.5
        assert ar.max_bad == 7
        assert ar.flt_pad == 0.2
        assert ar.filter_z is False

    @pytest.mark.unit
    def test_accepts_tuple_flt_pad(self):
        ar = ArtefactReject(flt_pad=(0.1, 0.2))

        assert ar.flt_pad == (0.1, 0.2)


# ============================================================================
# fit_ICA
# ============================================================================

class TestFitICA:
    """Tests for ICA fitting."""

    @pytest.mark.unit
    @pytest.mark.slow
    def test_n_components_is_n_good_channels_minus_one(self):
        epochs = make_artefact_epochs(n_ch=6, n_epochs=10)
        ar = ArtefactReject()

        ica = ar.fit_ICA(epochs, method='picard')

        assert ica.n_components == 5

    @pytest.mark.unit
    def test_invalid_method_raises(self):
        epochs = make_artefact_epochs(n_ch=6, n_epochs=10)
        ar = ArtefactReject()

        with pytest.raises(ValueError, match="Unknown ICA method"):
            ar.fit_ICA(epochs, method='bogus')


# ============================================================================
# apply_ICA
# ============================================================================

class TestApplyICA:
    """Tests for applying ICA component removal."""

    @pytest.mark.unit
    @pytest.mark.slow
    def test_empty_exclude_leaves_data_unchanged(self):
        epochs = make_artefact_epochs(n_ch=6, n_epochs=10, seed=1)
        ar = ArtefactReject()
        ica = ar.fit_ICA(epochs, method='picard')
        before = epochs.get_data().copy()

        ica.exclude = []
        result = ar.apply_ICA(ica, epochs.copy())

        np.testing.assert_allclose(result.get_data(), before)

    @pytest.mark.unit
    @pytest.mark.slow
    def test_nonempty_exclude_changes_data(self):
        epochs = make_artefact_epochs(n_ch=6, n_epochs=10, seed=1)
        ar = ArtefactReject()
        ica = ar.fit_ICA(epochs, method='picard')
        before = epochs.get_data().copy()

        ica.exclude = [0]
        result = ar.apply_ICA(ica, epochs.copy())

        assert not np.allclose(result.get_data(), before)


# ============================================================================
# automated_ica_blink_selection
# ============================================================================

class TestAutomatedIcaBlinkSelection:
    """Tests for automatic blink-component detection."""

    @pytest.mark.unit
    def test_zero_eog_channels_returns_gracefully(self):
        """Regression test: raw.copy().pick('eog') used to raise ValueError
        immediately on zero EOG channels, before the method's own
        defensive `if pick_eog.size > 0` check could ever run."""
        info = mne.create_info(['F1', 'F2', 'F3'], 250, ch_types='eeg')
        data = np.random.default_rng(0).normal(0, 1e-5, size=(3, 5000))
        raw = mne.io.RawArray(data, info)
        ica = mne.preprocessing.ICA(n_components=2, random_state=97)
        ica.fit(raw)
        ar = ArtefactReject()

        eog_epochs, eog_inds, eog_scores = ar.automated_ica_blink_selection(ica, raw)

        assert eog_epochs is None
        assert eog_inds == []
        assert eog_scores is None

    @pytest.mark.unit
    @pytest.mark.slow
    def test_with_eog_channels_returns_scores(self):
        raw = make_eog_correlated_raw()
        ar = ArtefactReject()
        ica = ar.fit_ICA(raw, method='picard')

        eog_epochs, eog_inds, eog_scores = ar.automated_ica_blink_selection(
            ica, raw, threshold=0.5
        )

        assert eog_epochs is not None
        assert eog_scores is not None
        assert eog_scores.shape[0] == ica.n_components


# ============================================================================
# filt_pad
# ============================================================================

class TestFiltPad:
    """Tests for edge-mean padding."""

    @pytest.mark.unit
    def test_pads_with_edge_means(self):
        ar = ArtefactReject()
        X = np.array([[1., 2., 3., 4., 5., 6.], [10., 20., 30., 40., 50., 60.]])

        padded = ar.filt_pad(X.copy(), pad_length=2)

        assert padded.shape == (2, 10)
        np.testing.assert_allclose(padded[:, :2], [[1.5, 1.5], [15., 15.]])
        np.testing.assert_allclose(padded[:, -2:], [[5.5, 5.5], [55., 55.]])
        np.testing.assert_array_equal(padded[:, 2:8], X)

    @pytest.mark.unit
    def test_clamps_pad_length_to_half_signal(self):
        ar = ArtefactReject()
        X = np.array([[1., 2., 3., 4.]])  # 4 samples, half = 2

        padded = ar.filt_pad(X.copy(), pad_length=100)

        assert padded.shape == (1, 8)  # 4 + 2*2, not 4 + 2*100


# ============================================================================
# box_smoothing
# ============================================================================

class TestBoxSmoothing:
    """Tests for boxcar smoothing."""

    @pytest.mark.unit
    def test_constant_signal_stays_constant(self):
        ar = ArtefactReject()
        X = np.ones((1, 1, 20)) * 5.0

        result = ar.box_smoothing(X, sfreq=100, boxcar=0.1)

        np.testing.assert_allclose(result[0, 0, 5:15], 5.0, atol=1e-6)

    @pytest.mark.unit
    def test_mutates_and_returns_same_array(self):
        ar = ArtefactReject()
        X = np.ones((1, 1, 20)) * 5.0
        original_id = id(X)

        result = ar.box_smoothing(X, sfreq=100, boxcar=0.1)

        assert id(result) == original_id


# ============================================================================
# mark_bads
# ============================================================================

class TestMarkBads:
    """Tests for artifact-segment boundary detection, including the
    fixed off-by-one bugs (general segment start was 1 sample too
    early; end-of-epoch fallback was 1 sample too short)."""

    @pytest.mark.unit
    def test_middle_segment_exact_boundaries(self):
        ar = ArtefactReject()
        times = np.linspace(0, 1, 10)
        Z = np.zeros((1, 10))
        Z[0, 3:6] = 10  # noisy samples 3, 4, 5

        events = ar.mark_bads(Z, z_thresh=5, times=times)

        assert len(events) == 1
        ep, (seg_slice, t_start, t_end, duration) = events[0][0], events[0][1]
        assert ep == 0
        assert seg_slice == slice(3, 6)
        assert t_start == pytest.approx(times[3])
        assert t_end == pytest.approx(times[6])

    @pytest.mark.unit
    def test_no_segments_returns_empty(self):
        ar = ArtefactReject()
        times = np.linspace(0, 1, 10)
        Z = np.zeros((1, 10))

        events = ar.mark_bads(Z, z_thresh=5, times=times)

        assert events == []

    @pytest.mark.unit
    def test_segment_touching_epoch_start(self):
        ar = ArtefactReject()
        times = np.linspace(0, 1, 10)
        Z = np.zeros((1, 10))
        Z[0, 0:3] = 10  # noisy samples 0, 1, 2

        events = ar.mark_bads(Z, z_thresh=5, times=times)

        seg_slice = events[0][1][0]
        assert seg_slice == slice(0, 3)

    @pytest.mark.unit
    def test_segment_touching_epoch_end(self):
        """Regression test for the end-of-epoch fallback fix: the
        segment must include the true last noisy sample (index 9),
        not stop one sample short."""
        ar = ArtefactReject()
        times = np.linspace(0, 1, 10)
        Z = np.zeros((1, 10))
        Z[0, 7:10] = 10  # noisy samples 7, 8, 9 (through the epoch's end)

        events = ar.mark_bads(Z, z_thresh=5, times=times)

        seg_slice = events[0][1][0]
        assert seg_slice == slice(7, 10)

    @pytest.mark.unit
    def test_multiple_segments_in_one_epoch(self):
        ar = ArtefactReject()
        times = np.linspace(0, 1, 10)
        Z = np.zeros((1, 10))
        Z[0, 1:3] = 10
        Z[0, 6:8] = 10

        events = ar.mark_bads(Z, z_thresh=5, times=times)

        slices = [seg[0] for seg in events[0][1:]]
        assert slices == [slice(1, 3), slice(6, 8)]


# ============================================================================
# z_score_data
# ============================================================================

class TestZScoreData:
    """Tests for z-scoring and data-driven threshold adjustment."""

    @pytest.mark.unit
    def test_matches_manual_computation_no_mask(self):
        from scipy.stats import zscore
        ar = ArtefactReject()
        rng = np.random.default_rng(0)
        X = rng.normal(0, 1, size=(2, 3, 10))

        Z_n, elecs_z, thresh = ar.z_score_data(X, z_thresh=4.0)

        Xr = X.swapaxes(0, 1).reshape(3, -1)
        Xz_manual = zscore(Xr, axis=1)
        Zn_manual = Xz_manual.sum(axis=0) / np.sqrt(3)
        thresh_manual = (
            4.0 + np.median(Zn_manual) + abs(Zn_manual.min() - np.median(Zn_manual))
        )

        np.testing.assert_allclose(Z_n.reshape(-1), Zn_manual)
        assert thresh == pytest.approx(thresh_manual)

    @pytest.mark.unit
    def test_mask_recomputes_zscore_on_masked_samples_only(self):
        from scipy.stats import zscore
        ar = ArtefactReject()
        rng = np.random.default_rng(0)
        X = rng.normal(0, 1, size=(2, 3, 20))
        mask = np.zeros(20, dtype=bool)
        mask[5:15] = True

        Z_n, elecs_z, thresh = ar.z_score_data(X, z_thresh=4.0, mask=mask)

        Xr = X.swapaxes(0, 1).reshape(3, -1)
        mask_tiled = np.tile(mask, 2)
        Xz = zscore(Xr, axis=1)
        Xz[:, mask_tiled] = zscore(Xr[:, mask_tiled], axis=1)
        Zn_manual = Xz.sum(axis=0) / np.sqrt(3)
        Zn_manual[mask_tiled] = Xz[:, mask_tiled].sum(axis=0) / np.sqrt(3)
        thresh_manual = 4.0 + np.median(Zn_manual[mask_tiled]) + abs(
            Zn_manual[mask_tiled].min() - np.median(Zn_manual[mask_tiled])
        )

        np.testing.assert_allclose(Z_n.reshape(-1), Zn_manual)
        assert thresh == pytest.approx(thresh_manual)

    @pytest.mark.unit
    def test_filter_z_branch_runs_without_error(self):
        ar = ArtefactReject()
        rng = np.random.default_rng(0)
        X = rng.normal(0, 1, size=(2, 3, 500))

        Z_n, elecs_z, thresh = ar.z_score_data(X, z_thresh=4.0, filter_z=(True, 250))

        assert Z_n.shape == (2, 500)


# ============================================================================
# preprocess_epochs
# ============================================================================

class TestPreprocessEpochs:
    """Regression tests for the mutable-default-argument and
    z_thresh-forwarding bugs."""

    @pytest.mark.unit
    def test_band_pass_default_not_mutated_across_calls(self):
        """A low-sfreq call (triggering the Nyquist clamp) must not
        corrupt the shared default for a later high-sfreq call that
        also omits band_pass."""
        ar = ArtefactReject(z_thresh=4.0, max_bad=2, flt_pad=0.1, filter_z=True)
        epochs_low = make_artefact_epochs(n_ch=4, n_epochs=5, sfreq=250, n_samples=500)

        with pytest.warns(UserWarning, match="Nyquist"):
            ar.preprocess_epochs(epochs_low)

        epochs_high = make_artefact_epochs(n_ch=4, n_epochs=5, sfreq=1000, n_samples=500)
        with warnings_as_errors():
            ar.preprocess_epochs(epochs_high)  # must NOT warn about Nyquist

    @pytest.mark.unit
    def test_z_thresh_forwarding(self):
        """Regression test: auto_repair_noise's z_thresh parameter was
        previously silently ignored (preprocess_epochs always used
        self.z_thresh); different explicit z_thresh values must now
        produce correspondingly different final thresholds."""
        ar = ArtefactReject(z_thresh=4.0, max_bad=2, flt_pad=0.1, filter_z=True)
        epochs_a = make_artefact_epochs(n_ch=4, n_epochs=5, sfreq=500, seed=1)
        epochs_b = make_artefact_epochs(n_ch=4, n_epochs=5, sfreq=500, seed=1)

        _, _, thresh_low, _ = ar.preprocess_epochs(epochs_a, z_thresh=1.0)
        _, _, thresh_high, _ = ar.preprocess_epochs(epochs_b, z_thresh=50.0)

        assert thresh_high > thresh_low
        assert thresh_high - thresh_low == pytest.approx(49.0)


class _WarningsAsErrors:
    def __enter__(self):
        import warnings
        self._cm = warnings.catch_warnings()
        self._cm.__enter__()
        warnings.simplefilter("error", UserWarning)
        return self

    def __exit__(self, *exc):
        self._cm.__exit__(*exc)
        return False


def warnings_as_errors():
    return _WarningsAsErrors()


# ============================================================================
# apply_hilbert
# ============================================================================

class TestApplyHilbert:
    """Tests for Hilbert envelope extraction."""

    @pytest.mark.unit
    def test_returns_positive_envelope_and_does_not_mutate_input(self):
        ar = ArtefactReject()
        epochs = make_artefact_epochs(n_ch=4, n_epochs=3, sfreq=500, n_samples=500)
        before = epochs.get_data().copy()

        envelope = ar.apply_hilbert(epochs, 110, 140)

        assert np.all(envelope >= 0)
        np.testing.assert_array_equal(epochs.get_data(), before)


# ============================================================================
# auto_repair_noise (full pipeline)
# ============================================================================

class TestAutoRepairNoise:
    """Integration tests for the full detect-and-repair pipeline."""

    @pytest.mark.unit
    @pytest.mark.slow
    def test_detects_and_repairs_injected_noise(self):
        epochs = make_artefact_epochs(inject_noise=True)
        ar = ArtefactReject(z_thresh=4.0, max_bad=3, flt_pad=0.1, filter_z=True)

        cleaned, z_thresh_used, report = ar.auto_repair_noise(
            epochs.copy(), sj=1, session=1, drop_bads=True
        )

        # the injected epoch is either successfully interpolated (kept)
        # or dropped -- either way, the heat map must show an interpolation
        # attempt was made
        assert np.any(ar.heat_map != 0)

    @pytest.mark.unit
    @pytest.mark.slow
    def test_z_thresh_forwarding_at_pipeline_level(self):
        """An impossibly high z_thresh must flag nothing, confirming the
        parameter actually reaches the detection step end-to-end."""
        epochs = make_artefact_epochs(inject_noise=True)
        ar = ArtefactReject(z_thresh=4.0, max_bad=3, flt_pad=0.1, filter_z=True)

        cleaned, z_thresh_used, _ = ar.auto_repair_noise(
            epochs.copy(), sj=1, session=1, drop_bads=True, z_thresh=1e6
        )

        assert len(cleaned) == len(epochs)
        assert z_thresh_used > 1e6

    @pytest.mark.unit
    @pytest.mark.slow
    def test_drop_bads_false_marks_metadata_when_present(self):
        epochs = make_artefact_epochs(inject_noise=True)
        epochs.metadata = pd.DataFrame({'trial': np.arange(len(epochs))})
        ar = ArtefactReject(z_thresh=4.0, max_bad=3, flt_pad=0.1, filter_z=True)

        cleaned, _, _ = ar.auto_repair_noise(
            epochs, sj=1, session=1, drop_bads=False
        )

        assert len(cleaned) == len(epochs)
        assert 'bad_epochs' in cleaned.metadata.columns

    @pytest.mark.unit
    @pytest.mark.slow
    def test_drop_bads_false_without_metadata_regression(self):
        """Regression test: previously crashed with AttributeError
        ('NoneType' object has no attribute 'reset_index') whenever
        epochs.metadata was None, which is the default unless metadata
        was explicitly attached (e.g. via align_meta_data)."""
        epochs = make_artefact_epochs(inject_noise=True)
        assert epochs.metadata is None
        ar = ArtefactReject(z_thresh=4.0, max_bad=3, flt_pad=0.1, filter_z=True)

        cleaned, _, _ = ar.auto_repair_noise(
            epochs, sj=1, session=1, drop_bads=False
        )

        assert 'bad_epochs' in cleaned.metadata.columns


# ============================================================================
# update_heat_map
# ============================================================================

class TestUpdateHeatMap:
    """Tests for heat_map/cleaned_info/not_cleaned_info state mutation."""

    @pytest.mark.unit
    def test_cleaned_updates_heat_map_and_cleaned_info(self):
        ar = ArtefactReject()
        channels = np.array(['Ch1', 'Ch2', 'Ch3'])
        ar.heat_map = np.zeros((5, 3))
        ar.cleaned_info = {ch: 0 for ch in channels}
        ar.not_cleaned_info = {ch: 0 for ch in channels}

        ar.update_heat_map(channels, ch_idx=np.array([0, 2]), tr_idx=1, upd_value=1)

        np.testing.assert_array_equal(ar.heat_map[1], [1, 0, 1])
        assert ar.cleaned_info['Ch1'] == 1
        assert ar.cleaned_info['Ch3'] == 1
        assert ar.cleaned_info['Ch2'] == 0

    @pytest.mark.unit
    def test_bad_updates_heat_map_and_not_cleaned_info(self):
        ar = ArtefactReject()
        channels = np.array(['Ch1', 'Ch2', 'Ch3'])
        ar.heat_map = np.zeros((5, 3))
        ar.cleaned_info = {ch: 0 for ch in channels}
        ar.not_cleaned_info = {ch: 0 for ch in channels}

        ar.update_heat_map(channels, ch_idx=np.array([1]), tr_idx=2, upd_value=-1)

        np.testing.assert_array_equal(ar.heat_map[2], [0, -1, 0])
        assert ar.not_cleaned_info['Ch2'] == 1


# ============================================================================
# Plotting smoke tests (no plt.show() blocks; safe to run headless)
# ============================================================================

class TestPlottingSmokeTests:
    """Smoke tests for the plotting methods used in QC reports."""

    def teardown_method(self):
        plt.close('all')

    @pytest.mark.unit
    def test_plot_hist_auto_repair_returns_false_when_all_zero(self):
        ar = ArtefactReject()
        ar.cleaned_info = {'Ch1': 0, 'Ch2': 0}

        result = ar.plot_hist_auto_repair('cleaned')

        assert result is False

    @pytest.mark.unit
    def test_plot_hist_auto_repair_returns_figure_when_nonzero(self):
        ar = ArtefactReject()
        ar.cleaned_info = {'Ch1': 3, 'Ch2': 0}

        result = ar.plot_hist_auto_repair('cleaned')

        assert isinstance(result, plt.Figure)

    @pytest.mark.unit
    def test_plot_heat_map_returns_figure(self):
        ar = ArtefactReject()
        ar.heat_map = np.array([[0, 1, -1], [1, 0, 0]])

        fig = ar.plot_heat_map(np.array(['A', 'B', 'C']))

        assert isinstance(fig, plt.Figure)

    @pytest.mark.unit
    def test_plot_z_score_epochs_returns_figure(self):
        ar = ArtefactReject()

        fig = ar.plot_z_score_epochs(np.random.randn(2, 20), z_thresh=1.5)

        assert isinstance(fig, plt.Figure)

    @pytest.mark.unit
    def test_plot_auto_repair_composes_all_figures(self):
        ar = ArtefactReject()
        ar.heat_map = np.array([[0, 1, -1], [1, 0, 0]])
        ar.cleaned_info = {'A': 1, 'B': 0, 'C': 0}
        ar.not_cleaned_info = {'A': 0, 'B': 0, 'C': 0}

        figs = ar.plot_auto_repair(
            np.array(['A', 'B', 'C']), np.random.randn(2, 20), 1.5
        )

        # zscore plot + heatmap + cleaned-histogram (bad-histogram is
        # skipped since not_cleaned_info is all zero)
        assert len(figs) == 3


# ============================================================================
# run_blink_ICA (end-to-end with mocked interactive prompts)
# ============================================================================

class TestRunBlinkICA:
    """End-to-end test with input()/time.sleep() mocked out, since the
    real method blocks on terminal input for interactive confirmation."""

    @pytest.mark.unit
    @pytest.mark.slow
    def test_auto_confirm_applies_ica(self, monkeypatch):
        monkeypatch.setattr(time, 'sleep', lambda *a, **k: None)
        monkeypatch.setattr('builtins.input', lambda *a, **k: 'y')

        raw = make_eog_correlated_raw(n_ch=6, n_samples=15000)
        ar = ArtefactReject()

        # ica_inst uses the same raw (same channel names as the fit) --
        # applying the fitted ICA back to a Raw/Epochs with different
        # channel names would fail regardless of run_blink_ICA's own
        # correctness, so this keeps the test focused on the confirm
        # loop + apply behavior.
        cleaned = ar.run_blink_ICA(
            fit_inst=raw, raw=raw, ica_inst=raw.copy(),
            sj=1, session=1, method='picard', threshold=0.5,
        )

        assert cleaned is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
