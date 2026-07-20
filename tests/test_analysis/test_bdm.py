"""
Test suite for open_dvm.analysis.BDM.

Organization
------------
- TestDataTypeProperty: data_type immutability
- TestSelectClassifier / TestGetClassifierWeights: classifier setup
- TestScoreAuc / TestComputeClassPerf: scoring metrics
- TestSelectMaxTrials / TestSetBdmParam: cross-validation parameter setup
- TestGetConditionLabels: condition/label selection
- TestTrainTestSplit / TestTrainTestCross / TestTrainTestSelect: CV splitting
- TestSetBdmWeights: Haufe-transform activation patterns
- TestSlidingWindow: temporal feature windowing
- TestCrossTimeDecoding: core per-timepoint decoding loop (PCA modes, GAT)
- TestClassifyIntegration: classify() end-to-end with known-separable data
- TestLocalizerClassifyIntegration: independent train/test set decoding
- TestIterClassify: split-factor iterative decoding
"""

import warnings
from unittest.mock import patch

import mne
import numpy as np
import pandas as pd
import pytest
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB

from open_dvm.analysis.BDM import BDM
from tests.fixtures.bdm_sample_data import (
    make_cross_condition_priming_epochs,
    make_localizer_epoch_pair,
    make_separable_epochs,
)

mne.set_log_level("ERROR")


def make_bdm(epochs, df, **kwargs):
    defaults = dict(
        sj=1,
        to_decode="label",
        baseline=None,
        nr_folds=5,
        elec_oi="all",
        data_type="raw",
        downsample=100,
        avg_trials=1,
    )
    defaults.update(kwargs)
    return BDM(epochs=epochs, df=df, **defaults)


# ============================================================================
# data_type property
# ============================================================================


class TestDataTypeProperty:
    @pytest.mark.unit
    def test_data_type_is_immutable(self):
        epochs, df = make_separable_epochs(n_trials=10)
        bdm = make_bdm(epochs, df)

        assert bdm.data_type == "raw"
        with pytest.raises(AttributeError):
            bdm.data_type = "tfr"


# ============================================================================
# select_classifier / get_classifier_weights
# ============================================================================


class TestSelectClassifier:
    @pytest.mark.unit
    @pytest.mark.parametrize(
        "name,expected_type_name",
        [
            ("LDA", "LinearDiscriminantAnalysis"),
            ("GNB", "GaussianNB"),
            ("svm", "CalibratedClassifierCV"),
        ],
    )
    def test_returns_expected_classifier_type(self, name, expected_type_name):
        bdm = BDM.__new__(BDM)
        bdm.classifier = name
        bdm.scale = True
        bdm.seed = 42213

        clf = bdm.select_classifier()

        assert type(clf).__name__ == expected_type_name

    @pytest.mark.unit
    def test_unknown_classifier_raises(self):
        bdm = BDM.__new__(BDM)
        bdm.classifier = "bogus"
        bdm.scale = True
        bdm.seed = 42213

        with pytest.raises(ValueError):
            bdm.select_classifier()

    @pytest.mark.unit
    def test_svm_without_scale_warns(self):
        bdm = BDM.__new__(BDM)
        bdm.classifier = "svm"
        bdm.scale = False
        bdm.seed = 42213

        with pytest.warns(UserWarning):
            bdm.select_classifier()


class TestGetClassifierWeights:
    @pytest.mark.unit
    def test_lda_weights_shape(self):
        bdm = BDM.__new__(BDM)
        bdm.classifier = "LDA"
        bdm.scale = True
        bdm.seed = 42213
        rng = np.random.default_rng(0)
        X = np.vstack(
            [
                rng.normal(0, 1, (20, 3)) + [3, 0, 0],
                rng.normal(0, 1, (20, 3)),
            ]
        )
        y = np.array([1] * 20 + [0] * 20)
        clf = bdm.select_classifier()
        clf.fit(X, y)

        w = bdm.get_classifier_weights(clf, X)

        assert w.shape == (3,)

    @pytest.mark.unit
    def test_gnb_returns_zeros_with_warning(self):
        bdm = BDM.__new__(BDM)
        rng = np.random.default_rng(0)
        X = np.vstack([rng.normal(0, 1, (20, 3)), rng.normal(2, 1, (20, 3))])
        y = np.array([0] * 20 + [1] * 20)
        clf = GaussianNB()
        clf.fit(X, y)

        with pytest.warns(UserWarning):
            w = bdm.get_classifier_weights(clf, X)

        assert np.all(w == 0)


# ============================================================================
# score_auc
# ============================================================================


class TestScoreAuc:
    @pytest.mark.unit
    def test_perfect_separation_with_boolean_labels(self):
        bdm = BDM.__new__(BDM)
        labels = np.array([False, False, True, True])
        scores = np.array([0.1, 0.2, 0.8, 0.9])

        assert bdm.score_auc(labels, scores) == pytest.approx(1.0)

    @pytest.mark.unit
    def test_perfect_separation_with_integer_labels_regression(self):
        """Regression test: rank[labels] previously did integer fancy
        indexing instead of boolean masking when labels was an int 0/1
        array (exactly the docstring's own example), silently returning
        0.75 instead of the documented 1.0."""
        bdm = BDM.__new__(BDM)
        labels = np.array([0, 0, 1, 1])
        scores = np.array([0.1, 0.2, 0.8, 0.9])

        assert bdm.score_auc(labels, scores) == pytest.approx(1.0)

    @pytest.mark.unit
    def test_perfect_inversion(self):
        bdm = BDM.__new__(BDM)
        labels = np.array([True, True, False, False])
        scores = np.array([0.1, 0.2, 0.8, 0.9])

        assert bdm.score_auc(labels, scores) == pytest.approx(0.0)

    @pytest.mark.unit
    def test_chance_performance(self):
        bdm = BDM.__new__(BDM)
        labels = np.array([0, 1, 0, 1])
        scores = np.array([0.4, 0.3, 0.6, 0.7])

        assert bdm.score_auc(labels, scores) == pytest.approx(0.5)

    @pytest.mark.unit
    def test_agrees_with_sklearn_on_random_data(self):
        from sklearn.metrics import roc_auc_score

        bdm = BDM.__new__(BDM)
        rng = np.random.default_rng(1)
        labels = rng.integers(0, 2, size=200).astype(bool)
        scores = rng.normal(0, 1, size=200)

        ours = bdm.score_auc(labels, scores)
        theirs = roc_auc_score(labels, scores)

        assert ours == pytest.approx(theirs)

    @pytest.mark.unit
    def test_all_same_class_raises(self):
        bdm = BDM.__new__(BDM)
        with pytest.raises(AssertionError):
            bdm.score_auc(np.array([True, True]), np.array([0.1, 0.2]))


# ============================================================================
# compute_class_perf
# ============================================================================


class TestComputeClassPerf:
    @pytest.mark.unit
    def test_binary_auc_perfect_separation(self):
        bdm = BDM.__new__(BDM)
        bdm.metric = "auc"
        scores = np.array([[0.9, 0.1], [0.8, 0.2], [0.1, 0.9], [0.2, 0.8]])
        true_labels = np.array(["A", "A", "B", "B"])
        predict = np.array(["A", "A", "B", "B"])

        perf = bdm.compute_class_perf(scores, true_labels, ["A", "B"], predict)

        assert perf == pytest.approx(1.0)

    @pytest.mark.unit
    def test_binary_acc_balanced_accuracy(self):
        bdm = BDM.__new__(BDM)
        bdm.metric = "acc"
        scores = np.array([[0.9, 0.1], [0.4, 0.6], [0.1, 0.9], [0.9, 0.1]])
        true_labels = np.array(["A", "A", "B", "B"])
        predict = np.array(["A", "B", "B", "A"])  # 1/2 correct per class

        perf = bdm.compute_class_perf(scores, true_labels, ["A", "B"], predict)

        assert perf == pytest.approx(0.5)


# ============================================================================
# select_max_trials
# ============================================================================


class TestSelectMaxTrials:
    @pytest.mark.unit
    def test_floors_to_multiple_of_nr_folds_per_condition(self):
        bdm = BDM.__new__(BDM)
        bdm.to_decode = "label"
        bdm.nr_folds = 2
        # cnd a: 4 per label (left/right) -> floor(4/2)*2=4
        # cnd b: 2 per label -> floor(2/2)*2=2
        df = pd.DataFrame(
            {
                "cnd": ["a"] * 8 + ["b"] * 4,
                "label": (["left"] * 4 + ["right"] * 4) + (["left"] * 2 + ["right"] * 2),
            }
        )

        max_trials = bdm.select_max_trials(df, ["a", "b"], ["left", "right"], "cnd")

        assert max_trials == 2  # min across conditions

    @pytest.mark.unit
    def test_all_data_uses_full_dataframe(self):
        bdm = BDM.__new__(BDM)
        bdm.to_decode = "label"
        bdm.nr_folds = 3
        df = pd.DataFrame({"label": ["x"] * 9 + ["y"] * 6})

        max_trials = bdm.select_max_trials(df, ["all_data"])

        # min class count = 6, floor(6/3)*3 = 6
        assert max_trials == 6

    @pytest.mark.unit
    def test_prints_warning_when_zero(self, capsys):
        bdm = BDM.__new__(BDM)
        bdm.to_decode = "label"
        bdm.nr_folds = 5
        df = pd.DataFrame({"cnd": ["a"] * 3, "label": ["x", "x", "y"]})

        max_trials = bdm.select_max_trials(df, ["a"], ["x", "y"], "cnd")

        assert max_trials == 0
        assert "not contain sufficient" in capsys.readouterr().out


# ============================================================================
# set_bdm_param
# ============================================================================


class TestSetBdmParam:
    @pytest.mark.unit
    def test_within_condition_setup(self):
        bdm = BDM.__new__(BDM)
        bdm.nr_folds = 2
        bdm.to_decode = "label"
        df = pd.DataFrame({"cnd": ["a"] * 8 + ["b"] * 8, "label": (["x"] * 4 + ["y"] * 4) * 2})
        y = df["label"].values

        nr_labels, tr_max, tr_cnds, te_cnds = bdm.set_bdm_param(
            y, df, ["a", "b"], "cnd", "all", downscale=False
        )

        assert nr_labels == 2
        assert tr_max == [4]
        assert tr_cnds == ["a", "b"]
        assert te_cnds is None
        assert bdm.cross is False

    @pytest.mark.unit
    def test_downscale_produces_descending_step_list(self):
        bdm = BDM.__new__(BDM)
        bdm.nr_folds = 2
        bdm.to_decode = "label"
        df = pd.DataFrame({"cnd": ["a"] * 8 + ["b"] * 8, "label": (["x"] * 4 + ["y"] * 4) * 2})
        y = df["label"].values

        _, tr_max, _, _ = bdm.set_bdm_param(y, df, ["a", "b"], "cnd", "all", downscale=True)

        assert tr_max == [4, 2]

    @pytest.mark.unit
    def test_cross_condition_setup(self):
        bdm = BDM.__new__(BDM)
        bdm.nr_folds = 5
        bdm.to_decode = "label"
        df = pd.DataFrame({"cnd": ["a"] * 8 + ["b"] * 8, "label": (["x"] * 4 + ["y"] * 4) * 2})
        y = df["label"].values

        _, tr_max, tr_cnds, te_cnds = bdm.set_bdm_param(
            y, df, [["a"], ["b"]], "cnd", "all", downscale=False
        )

        assert bdm.cross is True
        assert bdm.nr_folds == 1
        assert tr_cnds == ["a"]
        assert te_cnds == ["b"]

    @pytest.mark.unit
    def test_nr_folds_one_resets_with_warning_for_within_condition(self):
        bdm = BDM.__new__(BDM)
        bdm.nr_folds = 1
        bdm.to_decode = "label"
        df = pd.DataFrame({"cnd": ["a"] * 20, "label": ["x"] * 10 + ["y"] * 10})
        y = df["label"].values

        with pytest.warns(UserWarning, match="Nr folds"):
            bdm.set_bdm_param(y, df, ["a"], "cnd", "all", downscale=False)

        assert bdm.nr_folds == 10


# ============================================================================
# get_condition_labels
# ============================================================================


class TestGetConditionLabels:
    @pytest.mark.unit
    def test_selects_condition_trials_and_labels(self):
        bdm = BDM.__new__(BDM)
        bdm.to_decode = "label"
        bdm.cross = False
        df = pd.DataFrame({"cnd": ["a", "a", "a", "b", "b"], "label": [0, 1, 0, 1, 0]})

        out_df, cnd_idx, cnd_labels, labels, max_tr = bdm.get_condition_labels(
            df, "cnd", "a", [10], "all"
        )

        np.testing.assert_array_equal(cnd_idx, [0, 1, 2])
        np.testing.assert_array_equal(cnd_labels, [0, 1, 0])
        np.testing.assert_array_equal(labels, [0, 1])

    @pytest.mark.unit
    def test_labels_oi_subset_filters_trials(self):
        bdm = BDM.__new__(BDM)
        bdm.to_decode = "label"
        bdm.cross = False
        df = pd.DataFrame({"cnd": ["a", "a", "a", "b", "b"], "label": [0, 1, 0, 1, 0]})

        out_df, cnd_idx, cnd_labels, labels, max_tr = bdm.get_condition_labels(
            df, "cnd", "a", [10], [1]
        )

        np.testing.assert_array_equal(cnd_idx, [1])
        np.testing.assert_array_equal(cnd_labels, [1])

    @pytest.mark.unit
    def test_all_data_uses_every_trial(self):
        bdm = BDM.__new__(BDM)
        bdm.to_decode = "label"
        bdm.cross = False
        df = pd.DataFrame({"cnd": ["a", "a", "b", "b"], "label": [0, 1, 0, 1]})

        _, cnd_idx, cnd_labels, labels, _ = bdm.get_condition_labels(
            df, "cnd", "all_data", [10], "all"
        )

        np.testing.assert_array_equal(cnd_idx, [0, 1, 2, 3])


# ============================================================================
# train_test_split / train_test_cross / train_test_select
# ============================================================================


class TestTrainTestSplit:
    @pytest.mark.unit
    def test_fold_shapes_and_no_train_test_overlap(self):
        bdm = BDM.__new__(BDM)
        bdm.nr_folds = 5
        bdm.seed = 42213
        bdm.avg_trials = 1
        bdm.run_info = 1
        idx = np.arange(20)
        labels = np.array([0] * 10 + [1] * 10)
        bdm_info = {"run_1": {}}

        train_tr, test_tr, bdm_info = bdm.train_test_split(idx, labels, 10, bdm_info)

        assert train_tr.shape == (5, 2, 8)
        assert test_tr.shape == (5, 2, 2)
        for f in range(5):
            train_set = set(np.hstack(train_tr[f]))
            test_set = set(np.hstack(test_tr[f]))
            assert train_set.isdisjoint(test_set)

    @pytest.mark.unit
    def test_every_trial_used_as_test_exactly_once(self):
        """A proper k-fold partition: across all folds, each original
        trial index appears in the test set exactly once."""
        bdm = BDM.__new__(BDM)
        bdm.nr_folds = 5
        bdm.seed = 42213
        bdm.avg_trials = 1
        bdm.run_info = 1
        idx = np.arange(20)
        labels = np.array([0] * 10 + [1] * 10)
        bdm_info = {"run_1": {}}

        _, test_tr, _ = bdm.train_test_split(idx, labels, 10, bdm_info)

        te_all = np.hstack([np.hstack(test_tr[f]) for f in range(test_tr.shape[0])])
        assert sorted(te_all.tolist()) == sorted(idx.tolist())


class TestTrainTestCross:
    @pytest.mark.unit
    def test_train_only_when_test_idx_false(self):
        bdm = BDM.__new__(BDM)
        bdm.seed = 42213
        X = np.random.default_rng(0).normal(0, 1, size=(20, 3, 5))
        y = pd.Series(np.array([0] * 10 + [1] * 10))
        train_idx = np.arange(20)

        Xtr, Xte, Ytr, Yte = bdm.train_test_cross(X, y, train_idx, False)

        assert Xtr.shape[0] == 1  # single "fold"
        assert Xte is None
        assert set(np.unique(Ytr)) == {0, 1}

    @pytest.mark.unit
    def test_train_and_test_from_disjoint_pools(self):
        bdm = BDM.__new__(BDM)
        bdm.seed = 42213
        X = np.random.default_rng(0).normal(0, 1, size=(40, 3, 5))
        y = pd.Series(np.array([0] * 10 + [1] * 10 + [0] * 10 + [1] * 10))
        train_idx = np.arange(20)
        test_idx = np.arange(20, 40)

        Xtr, Xte, Ytr, Yte = bdm.train_test_cross(X, y, train_idx, test_idx)

        assert Xtr.shape[0] == 1
        assert Xte.shape[0] == 1
        assert set(np.unique(Ytr)) == {0, 1}
        assert set(np.unique(Yte)) == {0, 1}


class TestTrainTestSelect:
    @pytest.mark.unit
    def test_gathers_expected_data_and_labels(self):
        """Y is indexed by trial ID (Y[trial_id] = that trial's true
        label) -- distinguishing per-trial values below make it
        unambiguous exactly which trial's data/label ends up where."""
        bdm = BDM.__new__(BDM)
        bdm.nr_folds = 1
        X = np.arange(10 * 2 * 3).reshape(10, 2, 3).astype(float)
        Y = np.arange(10, 20)  # Y[i] = 10+i, so lookups are unambiguous
        train_tr = np.array([[[0, 1], [5, 6]]])  # fold 0: class0->[0,1], class1->[5,6]
        test_tr = np.array([[[2], [7]]])  # class0->[2], class1->[7]

        Xtr, Xte, Ytr, Yte = bdm.train_test_select(X, Y, train_tr, test_tr)

        np.testing.assert_array_equal(Ytr[0], [10, 11, 15, 16])
        np.testing.assert_array_equal(Yte[0], [12, 17])
        np.testing.assert_allclose(Xtr[0, 0], X[0])
        np.testing.assert_allclose(Xtr[0, 2], X[5])
        np.testing.assert_allclose(Xte[0, 0], X[2])
        np.testing.assert_allclose(Xte[0, 1], X[7])


# ============================================================================
# set_bdm_weights
# ============================================================================


class TestSetBdmWeights:
    @pytest.mark.unit
    def test_haufe_transform_matches_manual_computation(self):
        bdm = BDM.__new__(BDM)
        rng = np.random.default_rng(0)
        nr_elec, nr_time = 3, 2
        Xtr = rng.normal(0, 1, size=(15, nr_elec, nr_time))
        W = rng.normal(0, 1, size=(1, nr_time, 1, nr_elec))  # (freq,train,test,elec)

        A = bdm.set_bdm_weights(W.copy(), Xtr, nr_elec, nr_time)

        for train_t in range(nr_time):
            cov = np.cov(Xtr[..., train_t].T)
            expected = cov @ W[0, train_t, 0, :]
            expected = (expected - expected.mean()) / expected.std()
            np.testing.assert_allclose(A[0, train_t, 0, :], expected, atol=1e-8)

    @pytest.mark.unit
    def test_output_shape_matches_input_weight_shape(self):
        bdm = BDM.__new__(BDM)
        rng = np.random.default_rng(0)
        nr_elec, nr_time = 4, 3
        Xtr = rng.normal(0, 1, size=(10, nr_elec, nr_time))
        W = rng.normal(0, 1, size=(2, nr_time, 5, nr_elec))

        A = bdm.set_bdm_weights(W, Xtr, nr_elec, nr_time)

        assert A.shape == W.shape


# ============================================================================
# sliding_window
# ============================================================================


class TestSlidingWindow:
    @pytest.mark.unit
    def test_not_suitable_preserves_3d_shape_regression(self):
        """Regression test: previously returned a 4D array (with a
        spurious leading axis) for 3D input whenever the early-return
        'data not suitable' path fired."""
        bdm = BDM.__new__(BDM)
        X = np.random.default_rng(0).normal(0, 1, size=(5, 4, 3))

        out = bdm.sliding_window(X, window_size=10)  # window > n_time

        assert out.ndim == 3
        assert out.shape == (5, 4, 3)

    @pytest.mark.unit
    def test_not_suitable_preserves_4d_shape(self):
        bdm = BDM.__new__(BDM)
        X = np.random.default_rng(0).normal(0, 1, size=(1, 5, 4, 3))

        out = bdm.sliding_window(X, window_size=10)

        assert out.ndim == 4
        assert out.shape == (1, 5, 4, 3)

    @pytest.mark.unit
    def test_avg_window_matches_manual_mean(self):
        bdm = BDM.__new__(BDM)
        bdm.window_size = (4, False, False)  # (size, demean, downsample)
        X = np.random.default_rng(0).normal(0, 1, size=(3, 2, 10))

        out = bdm.sliding_window(X, window_size=4, demean=False, avg_window=True)

        t = 5
        expected = X[:, :, t - 3 : t + 1].mean(axis=-1)
        np.testing.assert_allclose(out[:, :, t], expected)


# ============================================================================
# cross_time_decoding
# ============================================================================


class TestCrossTimeDecoding:
    @staticmethod
    def _make_bdm(pca_components=(0, "across"), scale=True):
        bdm = BDM.__new__(BDM)
        bdm.scale = scale
        bdm.pca_components = pca_components
        bdm.metric = "auc"
        bdm.classifier = "LDA"
        bdm.tfr = None
        bdm.run_info = 1
        bdm.nr_folds = 1
        return bdm

    @pytest.mark.unit
    def test_gat_false_shape(self):
        bdm = self._make_bdm()
        rng = np.random.default_rng(0)
        Xtr = rng.normal(0, 1, size=(1, 1, 10, 3, 4))
        Xte = rng.normal(0, 1, size=(1, 1, 4, 3, 4))
        Ytr = np.tile(np.array([0] * 5 + [1] * 5), (1, 1))
        Yte = np.tile(np.array([0, 0, 1, 1]), (1, 1))
        labels = np.array([0, 1])

        class_acc, weights, conf_matrix = bdm.cross_time_decoding(
            Xtr, Xte, Ytr, Yte, labels, GAT=False
        )

        assert class_acc.shape == (1, 4, 1)
        assert weights.shape == (1, 4, 1, 3)
        assert conf_matrix.shape == (1, 4, 1, 2, 2)

    @pytest.mark.unit
    def test_gat_true_shape(self):
        bdm = self._make_bdm()
        rng = np.random.default_rng(0)
        Xtr = rng.normal(0, 1, size=(1, 1, 10, 3, 4))
        Xte = rng.normal(0, 1, size=(1, 1, 4, 3, 4))
        Ytr = np.tile(np.array([0] * 5 + [1] * 5), (1, 1))
        Yte = np.tile(np.array([0, 0, 1, 1]), (1, 1))
        labels = np.array([0, 1])

        class_acc, weights, conf_matrix = bdm.cross_time_decoding(
            Xtr, Xte, Ytr, Yte, labels, GAT=True
        )

        assert class_acc.shape == (1, 4, 4)

    @pytest.mark.unit
    def test_asymmetric_train_test_labels_no_crash_regression(self):
        """Regression test: confusion-matrix construction previously
        crashed with a shape mismatch whenever train and test label
        sets had different sizes. Train/test labels are disjoint here
        (no overlap) so this exercises the get_fake_confusion_matrix
        path specifically -- the exact code touched by the fix."""
        bdm = self._make_bdm()
        rng = np.random.default_rng(0)
        Xtr = rng.normal(0, 1, size=(1, 1, 12, 3, 2))
        Xte = rng.normal(0, 1, size=(1, 1, 6, 3, 2))
        Ytr = np.tile(np.array([0] * 4 + [1] * 4 + [2] * 4), (1, 1))
        Yte = np.tile(np.array([10, 10, 10, 11, 11, 11]), (1, 1))  # disjoint from Ytr
        labels = (np.array([0, 1, 2]), np.array([10, 11]))

        class_acc, weights, conf_matrix = bdm.cross_time_decoding(
            Xtr, Xte, Ytr, Yte, labels, GAT=False
        )

        assert conf_matrix.shape == (1, 2, 1, 2, 3)  # (folds,time,time,labels_te,labels_tr)

    @pytest.mark.unit
    def test_pca_across_fits_on_train_only(self):
        """Spy on PCA._fit (the shared internal method both .fit() and
        .fit_transform() delegate to) to confirm 'across' mode fits
        once per iteration on training data only (no leakage)."""
        bdm = self._make_bdm(pca_components=(2, "across"))
        rng = np.random.default_rng(0)
        Xtr = rng.normal(0, 1, size=(1, 1, 10, 4, 1))
        Xte = rng.normal(0, 1, size=(1, 1, 4, 4, 1))
        Ytr = np.tile(np.array([0] * 5 + [1] * 5), (1, 1))
        Yte = np.tile(np.array([0, 0, 1, 1]), (1, 1))
        labels = np.array([0, 1])

        fit_calls = []
        orig_fit = PCA._fit

        def spy_fit(self, X, *a, **k):
            fit_calls.append(X.shape[0])
            return orig_fit(self, X, *a, **k)

        with patch.object(PCA, "_fit", spy_fit):
            bdm.cross_time_decoding(Xtr, Xte, Ytr, Yte, labels, GAT=False)

        assert len(fit_calls) == 1
        assert fit_calls[0] == 10  # only the 10 training trials
        assert fit_calls[0] == 10  # only the 10 training trials

    @pytest.mark.unit
    def test_pca_all_fits_once_on_combined_pool(self):
        """Regression test: 'all' mode previously fit PCA on the
        combined pool then immediately discarded that fit via a second
        fit_transform() on Xtr_ alone, silently degrading into
        'across' behavior. Now verifies exactly one fit call, sized to
        the full combined pool (X), not just Xtr."""
        bdm = self._make_bdm(pca_components=(2, "all"))
        rng = np.random.default_rng(0)
        Xtr = rng.normal(0, 1, size=(1, 1, 10, 4, 1))
        Xte = rng.normal(0, 1, size=(1, 1, 4, 4, 1))
        Ytr = np.tile(np.array([0] * 5 + [1] * 5), (1, 1))
        Yte = np.tile(np.array([0, 0, 1, 1]), (1, 1))
        labels = np.array([0, 1])
        X_full = rng.normal(0, 1, size=(14, 4, 1))  # combined train+test pool

        fit_calls = []
        orig_fit = PCA.fit

        def spy_fit(self, X, *a, **k):
            fit_calls.append(X.shape[0])
            return orig_fit(self, X, *a, **k)

        with patch.object(PCA, "fit", spy_fit):
            bdm.cross_time_decoding(Xtr, Xte, Ytr, Yte, labels, GAT=False, X=X_full)

        assert fit_calls == [14]

    @pytest.mark.unit
    def test_pca_enabled_warns_weights_are_zero(self):
        bdm = self._make_bdm(pca_components=(2, "across"))
        rng = np.random.default_rng(0)
        Xtr = rng.normal(0, 1, size=(1, 1, 10, 4, 1))
        Xte = rng.normal(0, 1, size=(1, 1, 4, 4, 1))
        Ytr = np.tile(np.array([0] * 5 + [1] * 5), (1, 1))
        Yte = np.tile(np.array([0, 0, 1, 1]), (1, 1))
        labels = np.array([0, 1])

        with pytest.warns(UserWarning, match="not computed when PCA"):
            class_acc, weights, conf_matrix = bdm.cross_time_decoding(
                Xtr, Xte, Ytr, Yte, labels, GAT=False
            )

        assert np.all(weights == 0)


# ============================================================================
# classify(): end-to-end integration
# ============================================================================


class TestClassifyIntegration:
    @pytest.mark.unit
    def test_separable_signal_detected_only_after_onset(self):
        epochs, df = make_separable_epochs(
            n_trials=80,
            n_ch=4,
            n_samples=50,
            sfreq=100,
            seed=0,
            separable_from_sample=25,
        )
        bdm = make_bdm(epochs, df)

        output, _ = bdm.classify(
            cnds=dict(block_type=["main"]),
            window_oi=(-0.1, 0.4),
            labels_oi="all",
            GAT=False,
        )
        scores = output["main"]["dec_scores"]

        assert scores[:20].mean() == pytest.approx(0.5, abs=0.1)
        assert scores[30:].mean() > 0.95

    @pytest.mark.unit
    def test_excl_factor_not_mutated_across_calls_regression(self):
        """Regression test: excl_factor dict was previously mutated in
        place, leaking exclusion criteria into subsequent classify()
        calls sharing the same dict object."""
        epochs, df = make_separable_epochs(n_trials=80, seed=3)
        df["extra_cnd"] = (["x"] * 40) + (["y"] * 40)
        bdm = make_bdm(epochs, df)

        excl = {"extra_cnd": ["y"]}
        bdm.classify(
            cnds=dict(block_type=["main"]),
            window_oi=(-0.1, 0.4),
            labels_oi="all",
            GAT=False,
            excl_factor=excl,
        )

        assert excl == {"extra_cnd": ["y"]}

    @pytest.mark.unit
    def test_permutation_scores_near_chance_regression(self):
        """Regression test: label shuffling for nr_perm>0 never actually
        reached the classifier (Ytr/Yte were always looked up from the
        original, unshuffled label array), so perm_scores silently
        reflected genuine (non-null) decoding instead of chance."""
        epochs, df = make_separable_epochs(
            n_trials=80,
            n_ch=4,
            n_samples=50,
            sfreq=100,
            seed=0,
            separable_from_sample=25,
        )
        bdm = make_bdm(epochs, df)

        output, _ = bdm.classify(
            cnds=dict(block_type=["main"]),
            window_oi=(-0.1, 0.4),
            labels_oi="all",
            GAT=False,
            nr_perm=5,
        )

        perm_mean = output["main"]["perm_scores"].mean()
        real_late_mean = output["main"]["dec_scores"][30:].mean()

        assert perm_mean == pytest.approx(0.5, abs=0.1)
        assert real_late_mean > 0.95  # real run unaffected by permutation

    @pytest.mark.unit
    def test_avg_runs_greater_than_one_does_not_crash_regression(self):
        """Regression test: set_bdm_weights previously received only the
        last of several averaged runs' training data, and (separately)
        this whole path previously wasn't exercised/verified at all."""
        epochs, df = make_separable_epochs(
            n_trials=60,
            n_ch=4,
            n_samples=20,
            sfreq=100,
            seed=0,
            separable_from_sample=0,
        )
        bdm = make_bdm(epochs, df, nr_folds=5, avg_runs=3, output_params=True)

        output, params = bdm.classify(
            cnds=dict(block_type=["main"]),
            window_oi=(-0.1, 0.1),
            labels_oi="all",
            GAT=False,
        )

        assert output["main"]["dec_scores"].mean() > 0.9
        assert params["main"]["W"].shape[-1] == 4  # nr_elec


# ============================================================================
# classify(): special_col test-label override
# ============================================================================


class TestSpecialCol:
    @pytest.mark.unit
    def test_special_col_overrides_test_labels_only(self):
        """Ground truth: test-condition trials carry no genuine signal
        for `label` (decoding should sit at chance), but `prev_label`
        drives the same injected signal that trained the classifier on
        the training condition's `label`. special_col='prev_label'
        should recover near-perfect decoding after signal onset."""
        epochs, df = make_cross_condition_priming_epochs(seed=0)
        bdm = make_bdm(epochs, df, nr_folds=5)
        cnds = dict(block_type=[["localizer"], ["main"]])

        out_plain, _ = bdm.classify(
            cnds=cnds,
            window_oi=(-0.1, 0.4),
            labels_oi="all",
            GAT=False,
        )
        out_override, _ = bdm.classify(
            cnds=cnds,
            window_oi=(-0.1, 0.4),
            labels_oi="all",
            GAT=False,
            special_col="prev_label",
        )

        plain_scores = out_plain["localizer_main"]["dec_scores"]
        override_scores = out_override["localizer_main"]["dec_scores"]

        assert plain_scores[30:].mean() == pytest.approx(0.5, abs=0.15)
        assert override_scores[:20].mean() == pytest.approx(0.5, abs=0.1)
        assert override_scores[30:].mean() > 0.95

    @pytest.mark.unit
    def test_special_col_does_not_mutate_original_dataframe(self):
        epochs, df = make_cross_condition_priming_epochs(seed=1)
        bdm = make_bdm(epochs, df, nr_folds=5)
        df_before = bdm.df.copy()

        bdm.classify(
            cnds=dict(block_type=[["localizer"], ["main"]]),
            window_oi=(-0.1, 0.4),
            labels_oi="all",
            GAT=False,
            special_col="prev_label",
        )

        pd.testing.assert_frame_equal(bdm.df, df_before)

    @pytest.mark.unit
    def test_special_col_without_cross_condition_raises(self):
        epochs, df = make_cross_condition_priming_epochs(seed=2)
        bdm = make_bdm(epochs, df, nr_folds=5)

        with pytest.raises(ValueError, match="cross-condition"):
            bdm.classify(
                cnds=dict(block_type=["localizer", "main"]),
                window_oi=(-0.1, 0.4),
                labels_oi="all",
                GAT=False,
                special_col="prev_label",
            )

    @pytest.mark.unit
    def test_special_col_bad_column_name_raises(self):
        epochs, df = make_cross_condition_priming_epochs(seed=3)
        bdm = make_bdm(epochs, df, nr_folds=5)

        with pytest.raises(ValueError, match="not found"):
            bdm.classify(
                cnds=dict(block_type=[["localizer"], ["main"]]),
                window_oi=(-0.1, 0.4),
                labels_oi="all",
                GAT=False,
                special_col="nonexistent",
            )


# ============================================================================
# localizer_classify(): independent train/test set decoding
# ============================================================================


class TestLocalizerClassifyIntegration:
    """Regression tests: this entire method previously crashed via four
    stacked bugs (self.beh undefined, an unguarded headers[0] crash in
    select_bdm_data, an undefined `beh` variable in get_train_X, and a
    missing nr_freq axis in localizer_classify_'s array pre-allocation).
    """

    @pytest.mark.unit
    def test_gat_false_recovers_near_perfect_auc(self):
        epochs_tr, df_tr, epochs_te, df_te = make_localizer_epoch_pair()
        bdm = BDM(
            sj=1,
            epochs=[epochs_tr, epochs_te],
            df=[df_tr, df_te],
            to_decode="label",
            baseline=None,
            nr_folds=1,
            elec_oi="all",
            data_type="raw",
            downsample=100,
            avg_trials=1,
        )

        scores = bdm.localizer_classify(
            tr_window_oi=(-0.1, 0.4),
            te_window_oi=(-0.1, 0.4),
            tr_labels_oi="all",
            te_labels_oi="all",
            GAT=False,
        )

        assert scores["all_data"]["dec_scores"].mean() > 0.95

    @pytest.mark.unit
    def test_gat_true_diagonal_matches_within_time_result(self):
        epochs_tr, df_tr, epochs_te, df_te = make_localizer_epoch_pair()
        bdm = BDM(
            sj=1,
            epochs=[epochs_tr, epochs_te],
            df=[df_tr, df_te],
            to_decode="label",
            baseline=None,
            nr_folds=1,
            elec_oi="all",
            data_type="raw",
            downsample=100,
            avg_trials=1,
        )

        scores = bdm.localizer_classify(
            tr_window_oi=(-0.1, 0.4),
            te_window_oi=(-0.1, 0.4),
            tr_labels_oi="all",
            te_labels_oi="all",
            GAT=True,
        )

        gat = scores["all_data"]["dec_scores"]
        assert gat.ndim == 2
        assert gat.shape[0] == gat.shape[1]
        assert np.diag(gat).mean() > 0.95


# ============================================================================
# iter_classify_ (via classify with split_fact)
# ============================================================================


class TestIterClassify:
    @pytest.mark.unit
    def test_split_factor_averages_across_subsets(self):
        epochs, df = make_separable_epochs(
            n_trials=80,
            n_ch=4,
            n_samples=20,
            sfreq=100,
            seed=0,
            separable_from_sample=0,
        )
        df["session"] = (["s1"] * 40) + (["s2"] * 40)
        bdm = make_bdm(epochs, df, nr_folds=5)

        output, _ = bdm.classify(
            cnds=dict(block_type=["main"]),
            window_oi=(-0.1, 0.1),
            labels_oi="all",
            GAT=False,
            split_fact={"session": ["s1", "s2"]},
        )

        assert output["main"]["dec_scores"].mean() > 0.9

    @pytest.mark.unit
    def test_multi_condition_split_preserves_all_bdm_params_regression(self):
        """Regression test: bdm_params = {key: {}} previously replaced
        the ENTIRE bdm_params dict instead of just that key's entry,
        discarding params for all previously-processed conditions."""
        epochs, df = make_separable_epochs(
            n_trials=160,
            n_ch=4,
            n_samples=20,
            sfreq=100,
            seed=0,
            separable_from_sample=0,
        )
        # crossed (not correlated) so every block_type x session
        # combination has trials of both labels
        df["block_type"] = (["a"] * 80) + (["b"] * 80)
        df["session"] = (["s1"] * 40 + ["s2"] * 40) * 2
        bdm = make_bdm(epochs, df, nr_folds=5, output_params=False)

        output, params = bdm.classify(
            cnds=dict(block_type=["a", "b"]),
            window_oi=(-0.1, 0.1),
            labels_oi="all",
            GAT=False,
            split_fact={"session": ["s1", "s2"]},
        )

        assert {"a", "b"} <= set(output.keys())
        assert set(params.keys()) == {"a", "b"}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
