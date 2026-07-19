"""
Test suite for open_dvm.support.FolderStructure.

Organization
------------
- TestExtractSubjectNumber: filename -> subject number parsing
- TestFolderTracker: path generation, folder creation, overwrite logic
- TestReadRawBeh: raw behavioral CSV loading
- TestReadErps: evoked file loading
- TestReadTfr: time-frequency file loading
- TestReadBdm: decoding result loading/merging
- TestReadCtfs: CTF result loading
- TestLoadProcessedEpochs: epochs + behavioral + eye/trial exclusion
- TestBlockPrinting: stdout-suppression decorator
- TestRegressions: targeted checks for bugs fixed in this module
"""

import importlib
import os
import pickle
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import mne

from open_dvm.support.FolderStructure import FolderStructure, blockPrinting

# open_dvm/support/__init__.py does `from ...FolderStructure import FolderStructure`,
# which shadows the FolderStructure *module* attribute on the package with the
# *class* of the same name. `import ... as fs_module` chases that shadowed
# attribute, so importlib.import_module is used here to get the real module.
fs_module = importlib.import_module('open_dvm.support.FolderStructure')

from tests.fixtures.folder_structure_sample_data import (
    write_epochs,
    write_processed_beh,
    write_raw_beh,
    write_evoked,
    make_evoked,
    write_tfr,
    make_tfr,
    write_bdm_pickle,
    write_ctf_pickle,
)


# ============================================================================
# _extract_subject_number
# ============================================================================

class TestExtractSubjectNumber:
    @pytest.mark.unit
    def test_extracts_number_no_padding(self):
        assert FolderStructure._extract_subject_number('sub_1_main-epo.fif') == 1

    @pytest.mark.unit
    def test_extracts_number_with_padding(self):
        assert FolderStructure._extract_subject_number('sub_01_main-epo.fif') == 1

    @pytest.mark.unit
    def test_extracts_multidigit_number(self):
        assert FolderStructure._extract_subject_number('sub_123_main.pickle') == 123

    @pytest.mark.unit
    def test_no_match_raises(self):
        with pytest.raises(ValueError, match='Could not extract'):
            FolderStructure._extract_subject_number('no_subject_here.csv')


# ============================================================================
# folder_tracker
# ============================================================================

class TestFolderTracker:
    @pytest.mark.unit
    def test_creates_nested_folders(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        path = FolderStructure.folder_tracker(ext=['a', 'b'])
        assert os.path.isdir(path)
        assert path == os.path.join(str(tmp_path), 'a', 'b')

    @pytest.mark.unit
    def test_ext_none_uses_cwd_only(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        path = FolderStructure.folder_tracker(ext=None, fname='')
        assert path == str(tmp_path)

    @pytest.mark.unit
    def test_fname_none_returns_folder_only(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        path = FolderStructure.folder_tracker(ext=['processed'])
        assert path == os.path.join(str(tmp_path), 'processed')

    @pytest.mark.unit
    def test_fname_empty_string_returns_folder_only(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        path = FolderStructure.folder_tracker(ext=['processed'], fname='')
        assert path == os.path.join(str(tmp_path), 'processed')

    @pytest.mark.unit
    def test_fname_appended_to_path(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        path = FolderStructure.folder_tracker(ext=['processed'], fname='data.fif')
        assert path == os.path.join(str(tmp_path), 'processed', 'data.fif')

    @pytest.mark.unit
    def test_overwrite_true_returns_unchanged_path_even_if_exists(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.chdir(tmp_path)
        (tmp_path / 'data.fif').write_text('x')
        path = FolderStructure.folder_tracker(fname='data.fif', overwrite=True)
        assert path == os.path.join(str(tmp_path), 'data.fif')

    @pytest.mark.unit
    def test_overwrite_false_no_change_if_not_exists(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        path = FolderStructure.folder_tracker(fname='new.fif', overwrite=False)
        assert path == os.path.join(str(tmp_path), 'new.fif')

    @pytest.mark.unit
    def test_overwrite_false_appends_plus_before_extension(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.chdir(tmp_path)
        (tmp_path / 'data.fif').write_text('x')
        path = FolderStructure.folder_tracker(fname='data.fif', overwrite=False)
        assert path == os.path.join(str(tmp_path), 'data+.fif')

    @pytest.mark.unit
    def test_overwrite_false_increments_until_unique(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / 'data.fif').write_text('x')
        (tmp_path / 'data+.fif').write_text('x')
        path = FolderStructure.folder_tracker(fname='data.fif', overwrite=False)
        assert path == os.path.join(str(tmp_path), 'data++.fif')


# ============================================================================
# read_raw_beh
# ============================================================================

class TestReadRawBeh:
    @pytest.mark.unit
    def test_explicit_empty_list_returns_empty_list(self):
        assert FolderStructure().read_raw_beh(files=[]) == []

    @pytest.mark.unit
    def test_files_false_without_sj_session_raises(self):
        with pytest.raises(ValueError, match="Must provide"):
            FolderStructure().read_raw_beh(files=False)

    @pytest.mark.unit
    def test_explicit_file_list_concatenates(self, tmp_path):
        f1 = tmp_path / 'a.csv'
        f2 = tmp_path / 'b.csv'
        pd.DataFrame({'x': [1, 2]}).to_csv(f1, index=False)
        pd.DataFrame({'x': [3, 4]}).to_csv(f2, index=False)

        df = FolderStructure().read_raw_beh(files=[str(f1), str(f2)])

        assert list(df['x']) == [1, 2, 3, 4]
        assert list(df.index) == [0, 1, 2, 3]

    @pytest.mark.unit
    def test_globs_matching_subject_and_session(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        write_raw_beh(tmp_path, 1, 1, 'main', pd.DataFrame({'x': ['target']}))
        write_raw_beh(tmp_path, 10, 1, 'main', pd.DataFrame({'x': ['wrong_sj']}))
        write_raw_beh(tmp_path, 1, 11, 'main', pd.DataFrame({'x': ['wrong_ses']}))

        df = FolderStructure().read_raw_beh(sj=1, session=1)

        assert list(df['x']) == ['target']

    @pytest.mark.unit
    def test_no_matching_files_returns_empty_list(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        assert FolderStructure().read_raw_beh(sj=99, session=1) == []

    @pytest.mark.unit
    def test_string_sj_session_with_padding(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        write_raw_beh(tmp_path, '01', '01', 'main', pd.DataFrame({'x': [1]}))

        df = FolderStructure().read_raw_beh(sj='1', session='1')

        assert list(df['x']) == [1]


# ============================================================================
# read_erps
# ============================================================================

class TestReadErps:
    @pytest.mark.unit
    def test_defaults_to_all_data_condition(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        write_evoked(tmp_path, '01', 'all_data', 'locked', make_evoked(value=1.0))

        erps, times = FolderStructure().read_erps(erp_name='locked', sjs=[1])

        assert list(erps.keys()) == ['all_data']
        assert len(erps['all_data']) == 1

    @pytest.mark.unit
    def test_specific_conditions_and_subjects(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        for sj in ['01', '02']:
            write_evoked(tmp_path, sj, 'left', 'locked', make_evoked(value=1.0))
            write_evoked(tmp_path, sj, 'right', 'locked', make_evoked(value=2.0))

        erps, times = FolderStructure().read_erps(
            erp_name='locked', cnds=['left', 'right'], sjs=[1, 2]
        )

        assert set(erps.keys()) == {'left', 'right'}
        assert len(erps['left']) == 2
        assert len(erps['right']) == 2

    @pytest.mark.unit
    def test_sjs_all_globs_every_subject_sorted(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        for sj in ['01', '02', '10']:
            write_evoked(tmp_path, sj, 'all_data', 'locked', make_evoked(value=1.0))

        erps, times = FolderStructure().read_erps(erp_name='locked', sjs='all')

        assert len(erps['all_data']) == 3

    @pytest.mark.unit
    def test_returns_times_from_evoked(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        ev = make_evoked(value=1.0, sfreq=100, n_samples=10)
        write_evoked(tmp_path, '01', 'all_data', 'locked', ev)

        erps, times = FolderStructure().read_erps(erp_name='locked', sjs=[1])

        # fif round-trip introduces float32-level rounding in the timing axis
        np.testing.assert_allclose(times, ev.times, atol=1e-6)

    @pytest.mark.unit
    def test_match_true_truncates_mismatched_sample_counts(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.chdir(tmp_path)
        write_evoked(tmp_path, '01', 'all_data', 'locked',
                     make_evoked(n_samples=10, value=1.0))
        write_evoked(tmp_path, '02', 'all_data', 'locked',
                     make_evoked(n_samples=8, value=1.0))

        erps, times = FolderStructure().read_erps(
            erp_name='locked', sjs=[1, 2], match=True
        )

        assert erps['all_data'][0].times.size == 8
        assert erps['all_data'][1].times.size == 8


# ============================================================================
# read_tfr
# ============================================================================

class TestReadTfr:
    @pytest.mark.unit
    def test_single_condition_single_subject(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        write_tfr(tmp_path, ['wavelet'], '01', 'locked', 'all_data', make_tfr())

        tfr = FolderStructure().read_tfr(
            tfr_folder_path=['wavelet'], tfr_name='locked', sjs=[1]
        )

        assert list(tfr.keys()) == ['all_data']
        assert len(tfr['all_data']) == 1

    @pytest.mark.unit
    def test_multiple_conditions_and_subjects(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        for sj in ['01', '02']:
            write_tfr(tmp_path, ['wavelet'], sj, 'locked', 'left', make_tfr())
            write_tfr(tmp_path, ['wavelet'], sj, 'locked', 'right', make_tfr())

        tfr = FolderStructure().read_tfr(
            tfr_folder_path=['wavelet'], tfr_name='locked',
            cnds=['left', 'right'], sjs=[1, 2]
        )

        assert set(tfr.keys()) == {'left', 'right'}
        assert len(tfr['left']) == 2

    @pytest.mark.unit
    def test_sjs_all_globs_every_subject(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        for sj in ['01', '02', '10']:
            write_tfr(tmp_path, ['wavelet'], sj, 'locked', 'all_data', make_tfr())

        tfr = FolderStructure().read_tfr(
            tfr_folder_path=['wavelet'], tfr_name='locked', sjs='all'
        )

        assert len(tfr['all_data']) == 3


# ============================================================================
# read_bdm
# ============================================================================

class TestReadBdm:
    @pytest.mark.unit
    def test_single_analysis_no_prefix(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        write_bdm_pickle(tmp_path, ['loc'], '01', 'standard', {
            'left': {'dec_scores': 1}, 'info': {'times': [1, 2]}, 'bdm_info': {},
        })

        bdm = FolderStructure().read_bdm(bdm_folder_path=['loc'], bdm_name='standard')

        assert bdm[0]['left'] == {'dec_scores': 1}

    @pytest.mark.unit
    def test_multiple_analyses_prefixed_with_labels(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        write_bdm_pickle(tmp_path, ['loc'], '01', 'standard', {
            'left': {'dec_scores': 1}, 'info': {'times': [1, 2]}, 'bdm_info': {'k': 1},
        })
        write_bdm_pickle(tmp_path, ['loc'], '01', 'cross_temporal', {
            'left': {'dec_scores': 2}, 'info': {'times': [1, 2]}, 'bdm_info': {'k': 2},
        })

        bdm = FolderStructure().read_bdm(
            bdm_folder_path=['loc'], bdm_name=['standard', 'cross_temporal'],
            analysis_labels=['within', 'cross'],
        )

        assert bdm[0]['within_left'] == {'dec_scores': 1}
        assert bdm[0]['cross_left'] == {'dec_scores': 2}
        assert bdm[0]['bdm_info'] == {'within_k': 1, 'cross_k': 2}

    @pytest.mark.unit
    def test_multiple_analyses_prefixed_with_names_when_no_labels(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.chdir(tmp_path)
        write_bdm_pickle(tmp_path, ['loc'], '01', 'standard', {
            'left': {'dec_scores': 1}, 'info': {'times': [1, 2]},
            'bdm_info': {'k': 1},
        })
        write_bdm_pickle(tmp_path, ['loc'], '01', 'cross_temporal', {
            'left': {'dec_scores': 2}, 'info': {'times': [1, 2]},
            'bdm_info': {'k': 2},
        })

        # explicit sjs list (not the default 'all' glob path)
        bdm = FolderStructure().read_bdm(
            bdm_folder_path=['loc'], bdm_name=['standard', 'cross_temporal'],
            sjs=[1],
        )

        assert bdm[0]['standard_left'] == {'dec_scores': 1}
        assert bdm[0]['cross_temporal_left'] == {'dec_scores': 2}
        # bdm_info also falls back to the analysis name when no labels given
        assert bdm[0]['bdm_info'] == {'standard_k': 1, 'cross_temporal_k': 2}

    @pytest.mark.unit
    def test_underscore_containing_condition_names_preserved(
        self, tmp_path, monkeypatch
    ):
        # regression: single-condition analyses used to have the last
        # underscore-segment silently stripped from the merged key
        monkeypatch.chdir(tmp_path)
        write_bdm_pickle(tmp_path, ['loc'], '01', 'standard', {
            'target_absent': {'dec_scores': 1}, 'info': {'times': [1, 2]},
            'bdm_info': {},
        })
        write_bdm_pickle(tmp_path, ['loc'], '01', 'cross_temporal', {
            'target_present': {'dec_scores': 2}, 'info': {'times': [1, 2]},
            'bdm_info': {},
        })

        bdm = FolderStructure().read_bdm(
            bdm_folder_path=['loc'], bdm_name=['standard', 'cross_temporal'],
            analysis_labels=['within_time', 'across_time'],
        )

        assert bdm[0]['within_time_target_absent'] == {'dec_scores': 1}
        assert bdm[0]['across_time_target_present'] == {'dec_scores': 2}

    @pytest.mark.unit
    def test_subject_mismatch_across_analyses_raises(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        write_bdm_pickle(tmp_path, ['loc'], '01', 'standard', {
            'left': {}, 'info': {'times': [1, 2]}, 'bdm_info': {},
        })
        write_bdm_pickle(tmp_path, ['loc'], '02', 'cross_temporal', {
            'left': {}, 'info': {'times': [1, 2]}, 'bdm_info': {},
        })

        with pytest.raises(ValueError, match='Subject mismatch'):
            FolderStructure().read_bdm(
                bdm_folder_path=['loc'], bdm_name=['standard', 'cross_temporal'],
            )

    @pytest.mark.unit
    def test_time_mismatch_across_analyses_raises(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        write_bdm_pickle(tmp_path, ['loc'], '01', 'standard', {
            'left': {}, 'info': {'times': [1, 2]}, 'bdm_info': {},
        })
        write_bdm_pickle(tmp_path, ['loc'], '01', 'cross_temporal', {
            'left': {}, 'info': {'times': [9, 9]}, 'bdm_info': {},
        })

        with pytest.raises(ValueError, match='Time mismatch'):
            FolderStructure().read_bdm(
                bdm_folder_path=['loc'], bdm_name=['standard', 'cross_temporal'],
            )

    @pytest.mark.unit
    def test_no_files_found_raises(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        os.makedirs(tmp_path / 'bdm' / 'loc')

        with pytest.raises(ValueError, match='No files found'):
            FolderStructure().read_bdm(bdm_folder_path=['loc'], bdm_name='standard')

    @pytest.mark.unit
    def test_no_unclosed_file_handles(self, tmp_path, monkeypatch, recwarn):
        monkeypatch.chdir(tmp_path)
        write_bdm_pickle(tmp_path, ['loc'], '01', 'standard', {
            'left': {}, 'info': {'times': [1, 2]}, 'bdm_info': {},
        })

        FolderStructure().read_bdm(bdm_folder_path=['loc'], bdm_name='standard')

        assert not any(issubclass(w.category, ResourceWarning) for w in recwarn.list)


# ============================================================================
# read_ctfs
# ============================================================================

class TestReadCtfs:
    @pytest.mark.unit
    def test_output_type_ctf(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        write_ctf_pickle(tmp_path, ['orient'], '01', 'main', 'ctf', {'slopes': [1, 2]})

        ctfs = FolderStructure().read_ctfs(
            ctf_folder_path=['orient'], output_type='ctf', ctf_name='main', sjs=[1]
        )

        assert ctfs == [{'slopes': [1, 2]}]

    @pytest.mark.unit
    def test_output_type_info_and_param(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        write_ctf_pickle(tmp_path, ['orient'], '01', 'main', 'info', {'a': 1})
        write_ctf_pickle(tmp_path, ['orient'], '01', 'main', 'param', {'b': 2})

        info = FolderStructure().read_ctfs(
            ctf_folder_path=['orient'], output_type='info', ctf_name='main', sjs=[1]
        )
        param = FolderStructure().read_ctfs(
            ctf_folder_path=['orient'], output_type='param', ctf_name='main', sjs=[1]
        )

        assert info == [{'a': 1}]
        assert param == [{'b': 2}]

    @pytest.mark.unit
    def test_sjs_all_globs_sorted_by_subject(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        for sj, val in [('02', 2), ('01', 1), ('10', 10)]:
            write_ctf_pickle(tmp_path, ['orient'], sj, 'main', 'ctf', {'v': val})

        ctfs = FolderStructure().read_ctfs(
            ctf_folder_path=['orient'], output_type='ctf', ctf_name='main'
        )

        assert [c['v'] for c in ctfs] == [1, 2, 10]

    @pytest.mark.unit
    def test_no_unclosed_file_handles(self, tmp_path, monkeypatch, recwarn):
        monkeypatch.chdir(tmp_path)
        write_ctf_pickle(tmp_path, ['orient'], '01', 'main', 'ctf', {'v': 1})

        FolderStructure().read_ctfs(
            ctf_folder_path=['orient'], output_type='ctf', ctf_name='main', sjs=[1]
        )

        assert not any(issubclass(w.category, ResourceWarning) for w in recwarn.list)


# ============================================================================
# load_processed_epochs
# ============================================================================

class TestLoadProcessedEpochs:
    @pytest.mark.unit
    def test_uses_epochs_metadata_when_present(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        write_epochs(tmp_path, '01', 'main',
                     metadata=pd.DataFrame({'cond': ['a', 'b', 'a', 'b', 'a']}))

        df, epochs = FolderStructure().load_processed_epochs(
            sj=1, fname='main', preproc_name='main'
        )

        assert list(df['cond']) == ['a', 'b', 'a', 'b', 'a']
        assert len(epochs) == 5

    @pytest.mark.unit
    def test_falls_back_to_beh_csv_when_no_metadata(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        write_epochs(tmp_path, '01', 'main')
        write_processed_beh(tmp_path, '01', 'main',
                             pd.DataFrame({'cond': ['x'] * 5}))

        df, epochs = FolderStructure().load_processed_epochs(
            sj=1, fname='main', preproc_name='main'
        )

        assert list(df['cond']) == ['x'] * 5

    @pytest.mark.unit
    def test_beh_file_false_uses_epoch_events(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        write_epochs(tmp_path, '01', 'main')

        df, epochs = FolderStructure().load_processed_epochs(
            sj=1, fname='main', preproc_name='main', beh_file=False
        )

        assert 'condition' in df.columns
        assert len(df) == len(epochs)

    @pytest.mark.unit
    def test_invalid_modality_raises(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        write_epochs(tmp_path, '01', 'main')

        with pytest.raises(ValueError, match="modality must be"):
            FolderStructure().load_processed_epochs(
                sj=1, fname='main', preproc_name='main', modality='ecog'
            )

    @pytest.mark.unit
    def test_meg_modality_loads_from_meg_folder(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        write_epochs(tmp_path, '01', 'main', modality='meg')

        df, epochs = FolderStructure().load_processed_epochs(
            sj=1, fname='main', preproc_name='main', modality='MEG', beh_file=False
        )

        assert len(epochs) == 5

    @pytest.mark.unit
    def test_excl_factor_removes_matching_trials(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        write_epochs(tmp_path, '01', 'main',
                     metadata=pd.DataFrame({'cue': ['left', 'right', 'left',
                                                     'right', 'left']}))

        df, epochs = FolderStructure().load_processed_epochs(
            sj=1, fname='main', preproc_name='main',
            excl_factor={'cue': ['right']},
        )

        assert set(df['cue']) == {'left'}
        assert len(epochs) == 3

    @pytest.mark.unit
    def test_eye_dict_missing_preproc_file_warns_and_falls_back_to_eog(
        self, tmp_path, monkeypatch, capsys
    ):
        monkeypatch.chdir(tmp_path)
        write_epochs(tmp_path, '01', 'ses_1_main',
                     metadata=pd.DataFrame({'cond': ['a'] * 5}))

        eye_dict = {'use_tracker': True, 'eye_ch': 'HEOG'}
        captured_calls = []

        def fake_exclude_eye(sj, session, df, epochs, eye_dict_arg, eye, preproc_file):
            captured_calls.append(dict(eye_dict_arg))
            return df, epochs

        with patch.object(fs_module, 'exclude_eye', side_effect=fake_exclude_eye):
            FolderStructure().load_processed_epochs(
                sj=1, fname='ses_1_main', preproc_name='main', eye_dict=eye_dict,
            )

        out = capsys.readouterr().out
        assert 'not found' in out
        # exclude_eye was called with use_tracker forced False (no preproc file)
        assert captured_calls[0]['use_tracker'] is False
        # caller's dict is restored to its original value afterwards
        assert eye_dict['use_tracker'] is True

    @pytest.mark.unit
    def test_eye_dict_restored_even_if_exclude_eye_raises(
        self, tmp_path, monkeypatch
    ):
        # regression: eye_dict['use_tracker'] used to be permanently
        # left False if exclude_eye raised before the manual restore line
        monkeypatch.chdir(tmp_path)
        write_epochs(tmp_path, '01', 'ses_1_main',
                     metadata=pd.DataFrame({'cond': ['a'] * 5}))

        eye_dict = {'use_tracker': True, 'eye_ch': 'HEOG'}

        def raising_exclude_eye(*args, **kwargs):
            raise RuntimeError('boom')

        with patch.object(fs_module, 'exclude_eye', side_effect=raising_exclude_eye):
            with pytest.raises(RuntimeError, match='boom'):
                FolderStructure().load_processed_epochs(
                    sj=1, fname='ses_1_main', preproc_name='main', eye_dict=eye_dict,
                )

        assert eye_dict['use_tracker'] is True

    @pytest.mark.unit
    def test_eye_dict_uses_tracker_data_when_preproc_file_present(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.chdir(tmp_path)
        write_epochs(tmp_path, '01', 'ses_1_main',
                     metadata=pd.DataFrame({'cond': ['a'] * 5}))
        os.makedirs(tmp_path / 'preprocessing' / 'group_info', exist_ok=True)
        (tmp_path / 'preprocessing' / 'group_info' /
         'preproc_param_main.json').write_text('{}')
        os.makedirs(tmp_path / 'eye' / 'processed', exist_ok=True)
        np.savez(tmp_path / 'eye' / 'processed' / 'sub_01_ses_1_main.npz', x=1)

        eye_dict = {'use_tracker': True, 'eye_ch': 'HEOG'}
        captured = {}

        def fake_exclude_eye(sj, session, df, epochs, eye_dict_arg, eye, preproc_file):
            captured['eye_is_not_none'] = eye is not None
            captured['use_tracker'] = eye_dict_arg['use_tracker']
            return df, epochs

        with patch.object(fs_module, 'exclude_eye', side_effect=fake_exclude_eye):
            FolderStructure().load_processed_epochs(
                sj=1, fname='ses_1_main', preproc_name='main', eye_dict=eye_dict,
            )

        assert captured['eye_is_not_none'] is True
        assert captured['use_tracker'] is True  # untouched, tracker file was found

    @pytest.mark.unit
    def test_eye_dict_finds_combined_session_npz_file(self, tmp_path, monkeypatch):
        # regression: fname='all_main' (a session-combined epochs file, per
        # Epochs.save_preprocessed(..., combine_sessions=True)) has no
        # 'ses_XX' substring, so the session-extraction regex used to fall
        # back to session '1' and look for the wrong eye .npz file
        # (sub_01_ses_1_main.npz) instead of the real combined file
        # (sub_01_all_main.npz).
        monkeypatch.chdir(tmp_path)
        write_epochs(tmp_path, '01', 'all_main',
                     metadata=pd.DataFrame({'cond': ['a'] * 5}))
        os.makedirs(tmp_path / 'preprocessing' / 'group_info', exist_ok=True)
        (tmp_path / 'preprocessing' / 'group_info' /
         'preproc_param_main.json').write_text('{}')
        os.makedirs(tmp_path / 'eye' / 'processed', exist_ok=True)
        np.savez(tmp_path / 'eye' / 'processed' / 'sub_01_all_main.npz', x=1)

        eye_dict = {'use_tracker': True, 'eye_ch': 'HEOG'}
        captured = {}

        def fake_exclude_eye(sj, session, df, epochs, eye_dict_arg, eye, preproc_file):
            captured['session'] = session
            captured['eye_is_not_none'] = eye is not None
            return df, epochs

        with patch.object(fs_module, 'exclude_eye', side_effect=fake_exclude_eye):
            FolderStructure().load_processed_epochs(
                sj=1, fname='all_main', preproc_name='main', eye_dict=eye_dict,
            )

        assert captured['eye_is_not_none'] is True
        assert captured['session'] == 'all'


# ============================================================================
# blockPrinting
# ============================================================================

class TestBlockPrinting:
    @pytest.mark.unit
    def test_suppresses_stdout_during_call(self, capsys):
        @blockPrinting
        def noisy():
            print('should not appear')
            return 42

        result = noisy()

        assert result == 42
        assert capsys.readouterr().out == ''

    @pytest.mark.unit
    def test_restores_stdout_after_normal_return(self, capsys):
        import sys
        original = sys.stdout

        @blockPrinting
        def noisy():
            print('suppressed')

        noisy()

        assert sys.stdout is original

    @pytest.mark.unit
    def test_restores_stdout_even_if_wrapped_function_raises(self):
        import sys
        original = sys.stdout

        @blockPrinting
        def raises():
            raise ValueError('boom')

        with pytest.raises(ValueError):
            raises()

        assert sys.stdout is original


# ============================================================================
# Cross-cutting regressions
# ============================================================================

class TestRegressions:
    @pytest.mark.unit
    def test_folder_tracker_dotless_filename_no_crash(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / 'data').write_text('x')

        path = FolderStructure.folder_tracker(fname='data', overwrite=False)

        assert path == os.path.join(str(tmp_path), 'data+')

    @pytest.mark.unit
    def test_folder_tracker_multidot_filename_uses_last_dot(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.chdir(tmp_path)
        (tmp_path / 'sub_1.2_data.fif').write_text('x')

        path = FolderStructure.folder_tracker(
            fname='sub_1.2_data.fif', overwrite=False
        )

        assert path == os.path.join(str(tmp_path), 'sub_1.2_data+.fif')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
