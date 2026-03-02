"""Tests for Online Random Forest."""

import numpy as np
import pytest
from crypto_analysis.online.models.online_rf import OnlineRandomForest


class TestOnlineRandomForest:
    """Tests for OnlineRandomForest."""

    @pytest.fixture
    def rf(self):
        """Create RF instance."""
        return OnlineRandomForest(n_trees=5, max_samples_per_tree=100)

    def test_initialization(self, rf):
        """Test initialization."""
        assert rf.n_trees == 5
        assert rf.max_samples == 100
        assert len(rf.trees) == 0
        assert len(rf.sample_buffers) == 5

    def test_partial_fit(self, rf):
        """Test partial_fit updates trees."""
        X = np.random.randn(50, 10)
        y = np.random.randn(50)

        rf.partial_fit(X, y)

        assert len(rf.trees) == 5
        assert all(tree is not None for tree in rf.trees)

    def test_predict(self, rf):
        """Test prediction."""
        X_train = np.random.randn(100, 10)
        y_train = np.random.randn(100)
        rf.partial_fit(X_train, y_train)

        X_test = np.random.randn(10, 10)
        predictions = rf.predict(X_test)

        assert len(predictions) == 10

    def test_predict_before_fit(self):
        """Test predict returns zeros before fitting."""
        rf = OnlineRandomForest(n_trees=3)
        X = np.random.randn(5, 10)

        predictions = rf.predict(X)

        np.testing.assert_array_equal(predictions, np.zeros(5))

    def test_incremental_learning(self, rf):
        """Test incremental updates."""
        X1 = np.random.randn(30, 10)
        y1 = np.random.randn(30)
        rf.partial_fit(X1, y1)

        X2 = np.random.randn(30, 10)
        y2 = np.random.randn(30)
        rf.partial_fit(X2, y2)

        assert all(tree is not None for tree in rf.trees)
