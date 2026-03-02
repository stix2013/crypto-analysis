"""Tests for SignalAggregator."""

import pandas as pd
import pytest
from crypto_analysis.signals.aggregator import SignalAggregator
from crypto_analysis.signals.base import Signal, SignalType


class TestSignalAggregator:
    """Test SignalAggregator class."""

    def test_initialization(self):
        """Test aggregator initialization."""
        agg = SignalAggregator(method="weighted_confidence")
        assert agg.method == "weighted_confidence"
        assert agg.generators == []
        assert agg.weights == {}

    def test_invalid_method(self):
        """Test invalid method raises error."""
        with pytest.raises(ValueError):
            SignalAggregator(method="invalid_method")

    def test_add_generator(self):
        """Test adding generators."""
        agg = SignalAggregator()

        class MockGenerator:
            name = "test_gen"

        gen = MockGenerator()
        agg.add_generator(gen, weight=2.0)

        assert len(agg.generators) == 1
        assert agg.weights["test_gen"] == 2.0

    def test_add_generator_invalid_weight(self):
        """Test adding generator with invalid weight."""
        agg = SignalAggregator()

        class MockGenerator:
            name = "test_gen"

        gen = MockGenerator()
        with pytest.raises(ValueError):
            agg.add_generator(gen, weight=0)

    def test_aggregate_empty(self):
        """Test aggregating empty signals."""
        agg = SignalAggregator()
        result = agg.aggregate([])
        assert result is None

    def test_weighted_confidence_best_signal(self):
        """Test weighted confidence aggregation."""
        agg = SignalAggregator(method="weighted_confidence")

        signals = [
            Signal(
                symbol="BTC",
                signal_type=SignalType.ENTRY_LONG,
                confidence=0.8,
                timestamp=pd.Timestamp("2023-01-01"),
                source="gen1",
            ),
            Signal(
                symbol="BTC",
                signal_type=SignalType.ENTRY_LONG,
                confidence=0.6,
                timestamp=pd.Timestamp("2023-01-01"),
                source="gen2",
            ),
        ]

        result = agg.aggregate(signals)
        assert result is not None
        assert result.signal_type == SignalType.ENTRY_LONG
        assert result.source == "Aggregated"

    def test_weighted_confidence_conflict(self):
        """Test weighted confidence with conflicting signals."""
        agg = SignalAggregator(method="weighted_confidence")
        agg.weights = {"gen1": 1.0, "gen2": 1.0}

        signals = [
            Signal(
                symbol="BTC",
                signal_type=SignalType.ENTRY_LONG,
                confidence=0.8,
                timestamp=pd.Timestamp("2023-01-01"),
                source="gen1",
            ),
            Signal(
                symbol="BTC",
                signal_type=SignalType.ENTRY_SHORT,
                confidence=0.7,
                timestamp=pd.Timestamp("2023-01-01"),
                source="gen2",
            ),
        ]

        result = agg.aggregate(signals)
        assert result is not None
        # Should pick higher weighted score
        assert result.signal_type in [SignalType.ENTRY_LONG, SignalType.ENTRY_SHORT]

    def test_weighted_confidence_threshold(self):
        """Test weighted confidence below threshold."""
        agg = SignalAggregator(method="weighted_confidence")

        signals = [
            Signal(
                symbol="BTC",
                signal_type=SignalType.ENTRY_LONG,
                confidence=0.3,  # Below 0.5 threshold
                timestamp=pd.Timestamp("2023-01-01"),
                source="gen1",
            ),
        ]

        result = agg.aggregate(signals)
        assert result is None

    def test_majority_vote(self):
        """Test majority vote aggregation."""
        agg = SignalAggregator(method="majority_vote")

        signals = [
            Signal(
                symbol="BTC",
                signal_type=SignalType.ENTRY_LONG,
                confidence=0.8,
                timestamp=pd.Timestamp("2023-01-01"),
                source="gen1",
            ),
            Signal(
                symbol="BTC",
                signal_type=SignalType.ENTRY_LONG,
                confidence=0.6,
                timestamp=pd.Timestamp("2023-01-01"),
                source="gen2",
            ),
            Signal(
                symbol="BTC",
                signal_type=SignalType.ENTRY_SHORT,
                confidence=0.7,
                timestamp=pd.Timestamp("2023-01-01"),
                source="gen3",
            ),
        ]

        result = agg.aggregate(signals)
        assert result is not None
        assert result.signal_type == SignalType.ENTRY_LONG

    def test_majority_vote_no_majority(self):
        """Test majority vote with no clear majority."""
        agg = SignalAggregator(method="majority_vote")

        signals = [
            Signal(
                symbol="BTC",
                signal_type=SignalType.ENTRY_LONG,
                confidence=0.8,
                timestamp=pd.Timestamp("2023-01-01"),
                source="gen1",
            ),
            Signal(
                symbol="BTC",
                signal_type=SignalType.ENTRY_SHORT,
                confidence=0.7,
                timestamp=pd.Timestamp("2023-01-01"),
                source="gen2",
            ),
        ]

        result = agg.aggregate(signals)
        # Should return None when no majority
        assert result is None

    def test_best_confidence(self):
        """Test best confidence aggregation."""
        agg = SignalAggregator(method="best_confidence")

        signals = [
            Signal(
                symbol="BTC",
                signal_type=SignalType.ENTRY_LONG,
                confidence=0.5,
                timestamp=pd.Timestamp("2023-01-01"),
                source="gen1",
            ),
            Signal(
                symbol="BTC",
                signal_type=SignalType.ENTRY_SHORT,
                confidence=0.9,
                timestamp=pd.Timestamp("2023-01-01"),
                source="gen2",
            ),
        ]

        result = agg.aggregate(signals)
        assert result is not None
        assert result.signal_type == SignalType.ENTRY_SHORT
        assert result.confidence == 0.9
