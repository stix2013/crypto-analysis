"""Signal aggregation and combination."""

from collections import Counter

import numpy as np

from crypto_analysis.signals.base import Signal, SignalGenerator, SignalType


class SignalAggregator:
    """Combines signals from multiple generators using ensemble methods.

    Supports weighted confidence, majority vote, and best confidence aggregation.

    Attributes:
        method: Aggregation method to use
        generators: List of signal generators
        weights: Dictionary of generator weights
        signal_history: History of aggregated signals

    """

    def __init__(self, method: str = "weighted_confidence") -> None:
        """Initialize signal aggregator.

        Args:
            method: Aggregation method ('weighted_confidence', 'majority_vote', 'best_confidence')

        Raises:
            ValueError: If method is not recognized

        """
        valid_methods = ["weighted_confidence", "majority_vote", "best_confidence"]
        if method not in valid_methods:
            raise ValueError(f"Method must be one of {valid_methods}, got {method}")

        self.method = method
        self.generators: list = []
        self.weights: dict[str, float] = {}
        self.signal_history: list[Signal] = []

    def add_generator(self, generator: SignalGenerator, weight: float = 1.0) -> None:
        """Add a signal generator with weight.

        Args:
            generator: Signal generator instance
            weight: Weight for this generator's signals

        Raises:
            ValueError: If weight is not positive

        """
        if weight <= 0:
            raise ValueError(f"Weight must be positive, got {weight}")

        self.generators.append(generator)
        self.weights[generator.name] = weight

    def aggregate(
        self,
        signals: list[Signal],
        current_position: float | None = None,
    ) -> Signal | None:
        """Combine multiple signals into final decision.

        Args:
            signals: List of signals to aggregate
            current_position: Current position size

        Returns:
            Aggregated signal or None if no signals

        """
        if not signals:
            return None

        if self.method == "weighted_confidence":
            return self._weighted_confidence(signals, current_position)
        elif self.method == "majority_vote":
            return self._majority_vote(signals, current_position)
        else:
            return self._best_confidence(signals, current_position)

    def _weighted_confidence(
        self,
        signals: list[Signal],
        current_position: float | None,
    ) -> Signal | None:
        """Weight signals by generator performance and confidence.

        Args:
            signals: List of signals to aggregate
            current_position: Current position size

        Returns:
            Aggregated signal or None

        """
        # Group by signal type
        by_type: dict[SignalType, list[float]] = {}
        for sig in signals:
            if sig.signal_type not in by_type:
                by_type[sig.signal_type] = []
            weight = self.weights.get(sig.source.split("_")[0], 1.0)
            by_type[sig.signal_type].append(sig.confidence * weight)

        # Calculate weighted score for each type
        scores = {k: sum(v) / len(v) for k, v in by_type.items()}

        if not scores:
            return None

        # Select highest scoring signal type
        best_type = max(scores, key=lambda k: scores[k])
        best_score = scores[best_type]

        # Threshold
        if best_score < 0.5:
            return None

        # Create aggregated signal
        best_signals = [s for s in signals if s.signal_type == best_type]

        return Signal(
            symbol=best_signals[0].symbol,
            signal_type=best_type,
            confidence=best_score,
            timestamp=best_signals[0].timestamp,
            source="Aggregated",
            metadata={
                "contributing_signals": len(best_signals),
                "individual_confidences": [s.confidence for s in best_signals],
                "all_scores": scores,
            },
        )

    def _majority_vote(
        self,
        signals: list[Signal],
        current_position: float | None,
    ) -> Signal | None:
        """Simple majority vote aggregation.

        Args:
            signals: List of signals to aggregate
            current_position: Current position size

        Returns:
            Aggregated signal or None

        """
        types = [s.signal_type for s in signals]
        votes = Counter(types)

        if not votes:
            return None

        best_type, count = votes.most_common(1)[0]

        # Need majority
        if count <= len(signals) / 2:
            return None

        best_signals = [s for s in signals if s.signal_type == best_type]
        avg_confidence = float(np.mean([s.confidence for s in best_signals]))

        return Signal(
            symbol=best_signals[0].symbol,
            signal_type=best_type,
            confidence=avg_confidence,
            timestamp=best_signals[0].timestamp,
            source="MajorityVote",
            metadata={"votes": count, "total_signals": len(signals)},
        )

    def _best_confidence(
        self,
        signals: list[Signal],
        current_position: float | None,
    ) -> Signal | None:
        """Pick highest confidence signal.

        Args:
            signals: List of signals to aggregate
            current_position: Current position size

        Returns:
            Signal with highest confidence

        """
        return max(signals, key=lambda x: x.confidence)
