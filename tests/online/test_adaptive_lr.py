"""Tests for Adaptive Learning Rate."""

import numpy as np
from crypto_analysis.online.detection.adaptive_lr import AdaptiveLearningRate


class TestAdaptiveLearningRate:
    """Tests for AdaptiveLearningRate."""

    def test_initialization(self):
        """Test initialization with default values."""
        scheduler = AdaptiveLearningRate()

        assert scheduler.base_lr == 0.001
        assert scheduler.min_lr == 1e-6
        assert scheduler.max_lr == 0.1
        assert scheduler.current_lr == 0.001

    def test_custom_initialization(self):
        """Test initialization with custom values."""
        scheduler = AdaptiveLearningRate(base_lr=0.01, min_lr=1e-5, max_lr=0.5)

        assert scheduler.base_lr == 0.01
        assert scheduler.min_lr == 1e-5
        assert scheduler.max_lr == 0.5

    def test_update_without_sufficient_history(self):
        """Test update without enough loss history."""
        scheduler = AdaptiveLearningRate()

        lr = scheduler.update(recent_loss=0.5, market_volatility=0.02)

        assert lr == 0.001

    def test_update_decreasing_loss(self):
        """Test update with decreasing loss."""
        scheduler = AdaptiveLearningRate()

        for i in range(15):
            loss = 1.0 - i * 0.05
            scheduler.update(recent_loss=loss, market_volatility=0.02)

        assert scheduler.current_lr > 0.001

    def test_update_increasing_loss(self):
        """Test update with increasing loss."""
        scheduler = AdaptiveLearningRate()

        for i in range(15):
            loss = 0.1 + i * 0.1
            scheduler.update(recent_loss=loss, market_volatility=0.02)

        assert scheduler.current_lr < 0.001

    def test_high_volatility_reduces_lr(self):
        """Test high volatility reduces learning rate."""
        scheduler = AdaptiveLearningRate(base_lr=0.01)

        for _i in range(15):
            scheduler.update(recent_loss=0.5, market_volatility=0.5)

        assert scheduler.current_lr < 0.01

    def test_low_volatility_allows_higher_lr(self):
        """Test low volatility allows higher learning rate."""
        scheduler = AdaptiveLearningRate(base_lr=0.001)

        for _i in range(15):
            scheduler.update(recent_loss=0.5, market_volatility=0.001)

        assert scheduler.current_lr > 0.001

    def test_lr_bounds(self):
        """Test learning rate stays within bounds."""
        scheduler = AdaptiveLearningRate(min_lr=0.0001, max_lr=0.1)

        for _i in range(100):
            scheduler.update(
                recent_loss=np.random.random(),
                market_volatility=np.random.random(),
            )

        assert scheduler.min_lr <= scheduler.current_lr <= scheduler.max_lr

    def test_lr_history_tracking(self):
        """Test learning rate history is tracked."""
        scheduler = AdaptiveLearningRate()

        for _i in range(50):
            scheduler.update(recent_loss=0.5, market_volatility=0.02)

        assert len(scheduler.lr_history) == 50
