"""Adaptive learning rate scheduling for online learning."""

from collections import deque

import numpy as np


class AdaptiveLearningRate:
    """Adjusts learning rate based on market conditions and model performance.

    Implements adaptive learning rate that responds to:
    - Loss trends (increasing = reduce LR, decreasing = can increase)
    - Market volatility (high vol = lower LR for stability)

    Attributes:
        base_lr: Initial learning rate
        min_lr: Minimum allowed learning rate
        max_lr: Maximum allowed learning rate
        current_lr: Current active learning rate
        volatility_estimate: Running volatility estimate
    """

    def __init__(
        self,
        base_lr: float = 0.001,
        min_lr: float = 1e-6,
        max_lr: float = 0.1,
    ) -> None:
        """Initialize adaptive learning rate.

        Args:
            base_lr: Initial learning rate
            min_lr: Minimum allowed learning rate
            max_lr: Maximum allowed learning rate
        """
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.current_lr = base_lr

        self.loss_history: deque[float] = deque(maxlen=100)
        self.lr_history: deque[float] = deque(maxlen=100)

        self.volatility_estimate = 0.02

    def update(
        self,
        recent_loss: float,
        market_volatility: float,
        gradient_norm: float | None = None,
    ) -> float:
        """Update learning rate based on conditions.

        Args:
            recent_loss: Most recent training loss
            market_volatility: Current market volatility estimate
            gradient_norm: Current gradient norm (optional)

        Returns:
            Updated learning rate
        """
        self.loss_history.append(recent_loss)
        self.volatility_estimate = 0.9 * self.volatility_estimate + 0.1 * market_volatility

        self.lr_history.append(self.current_lr)

        if len(self.loss_history) < 10:
            return self.current_lr

        recent_mean = np.mean(list(self.loss_history)[-5:])
        older_mean = (
            np.mean(list(self.loss_history)[:5]) if len(self.loss_history) >= 10 else recent_mean
        )

        loss_trend = recent_mean - older_mean

        if loss_trend > 0:
            self.current_lr *= 0.98  # More subtle adjustment
        elif loss_trend < -0.01:
            self.current_lr *= 1.02

        # Use 0.03 as baseline (0.02 + 0.01) so vol_factor is 1.0 when vol is 0.02
        vol_factor = 0.03 / (self.volatility_estimate + 0.01)
        # Apply volatility adjustment more subtly
        self.current_lr *= 0.9 + 0.1 * np.clip(vol_factor, 0.5, 2.0)

        self.current_lr = np.clip(self.current_lr, self.min_lr, self.max_lr)

        return self.current_lr
