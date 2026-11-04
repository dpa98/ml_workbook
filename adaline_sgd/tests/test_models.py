import sys
import os
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from adaline_sgd.models import AdalineSGD
from adaline_sgd.data import standardize


def separable_data(n=100, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.normal(size=(n, 2))
    y = np.where(X[:, 0] + X[:, 1] > 0, 1, 0)
    return X, y


class TestAdalineSGD(unittest.TestCase):
    def test_fit_converges_on_separable_data(self):
        X, y = separable_data()
        ada = AdalineSGD(eta=0.01, n_iter=20, random_state=1)
        ada.fit(X, y)
        self.assertEqual(len(ada.losses_), 20)
        self.assertLess(ada.losses_[-1], ada.losses_[0])

    def test_predict_on_separable_data(self):
        X, y = separable_data()
        ada = AdalineSGD(eta=0.01, n_iter=50, random_state=1)
        ada.fit(X, y)
        accuracy = np.mean(ada.predict(X) == y)
        self.assertGreaterEqual(accuracy, 0.95)

    def test_reproducibility_with_random_state(self):
        X, y = separable_data()
        ada1 = AdalineSGD(eta=0.01, n_iter=5, random_state=42)
        ada2 = AdalineSGD(eta=0.01, n_iter=5, random_state=42)
        ada1.fit(X, y)
        ada2.fit(X, y)
        np.testing.assert_allclose(ada1.w_, ada2.w_)
        np.testing.assert_allclose(ada1.losses_, ada2.losses_)

    def test_fit_creates_weight_history(self):
        X, y = separable_data()
        ada = AdalineSGD(eta=0.01, n_iter=3, random_state=1)
        ada.fit(X, y)
        self.assertEqual(len(ada.w_history_), 3 * len(X))
        self.assertTrue(ada.w_initialized)

    def test_partial_fit_single_sample(self):
        X, y = separable_data()
        ada = AdalineSGD(eta=0.01, n_iter=1, random_state=1)
        ada.partial_fit(X[0], y[0])
        self.assertEqual(ada.w_.shape, (2,))
        w_before = ada.w_.copy()
        ada.partial_fit(X[0], y[0])
        self.assertFalse(np.array_equal(ada.w_, w_before))

    def test_partial_fit_batch(self):
        X, y = separable_data()
        ada = AdalineSGD(eta=0.01, n_iter=1, random_state=1)
        ada.partial_fit(X, y)
        self.assertTrue(ada.w_initialized)

    def test_shuffle_false_keeps_order(self):
        X, y = separable_data()
        ada = AdalineSGD(eta=0.01, n_iter=2, shuffle=False, random_state=1)
        ada.fit(X, y)
        self.assertEqual(len(ada.losses_), 2)


class TestStandardize(unittest.TestCase):
    def test_zero_mean_unit_variance(self):
        rng = np.random.RandomState(0)
        X = rng.uniform(low=10.0, high=20.0, size=(50, 2))
        X_std = standardize(X)
        np.testing.assert_allclose(X_std.mean(axis=0), 0.0, atol=1e-10)
        np.testing.assert_allclose(X_std.std(axis=0), 1.0, atol=1e-10)

    def test_does_not_mutate_input(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        X_copy = X.copy()
        standardize(X)
        np.testing.assert_array_equal(X, X_copy)


if __name__ == "__main__":
    unittest.main()
