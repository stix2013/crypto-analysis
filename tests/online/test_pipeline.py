"""Tests for Continuous Learning Pipeline."""

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from crypto_analysis.online.pipeline import ContinuousLearningPipeline


def create_test_data(n_points: int = 100) -> pd.DataFrame:
    """Create test OHLCV data."""
    np.random.seed(42)

    prices = 100 + np.cumsum(np.random.randn(n_points) * 0.5)

    return pd.DataFrame(
        {
            "open": prices + np.random.randn(n_points) * 0.2,
            "high": prices + np.abs(np.random.randn(n_points)),
            "low": prices - np.abs(np.random.randn(n_points)),
            "close": prices,
            "volume": np.random.uniform(1000, 5000, n_points),
        },
        index=pd.date_range("2023-01-01", periods=n_points, freq="1h"),
    )


class TestContinuousLearningPipeline:
    """Tests for ContinuousLearningPipeline."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        temp_path = tempfile.mkdtemp()
        yield temp_path
        shutil.rmtree(temp_path)

    def test_initialization(self, temp_dir):
        """Test pipeline initialization."""
        pipeline = ContinuousLearningPipeline(checkpoint_dir=temp_dir)

        assert pipeline.checkpoint_dir == Path(temp_dir)
        assert pipeline.active_model is None
        assert pipeline.candidate_model is None
        assert pipeline.ab_test_active is False
        assert pipeline.ab_split_ratio == 0.1

    def test_stream_data(self, temp_dir):
        """Test data streaming."""
        pipeline = ContinuousLearningPipeline(checkpoint_dir=temp_dir)
        data = create_test_data(100)

        pipeline.stream_data(data)

        assert len(pipeline.data_buffer) == 1

    def test_stream_data_buffer_limit(self, temp_dir, mocker):
        """Test data buffer respects max size."""
        pipeline = ContinuousLearningPipeline(checkpoint_dir=temp_dir)
        mocker.patch.object(pipeline, "_trigger_retraining")

        for _ in range(15000):
            pipeline.stream_data(create_test_data(10))

        assert len(pipeline.data_buffer) == 10000

    def test_get_prediction_no_model(self, temp_dir):
        """Test get_prediction without trained model."""
        pipeline = ContinuousLearningPipeline(checkpoint_dir=temp_dir)
        data = create_test_data(100)

        signals, source = pipeline.get_prediction(data)

        assert source == "none"

    def test_update_performance(self, temp_dir):
        """Test performance tracking."""
        pipeline = ContinuousLearningPipeline(checkpoint_dir=temp_dir)

        pipeline.update_performance("active", 0.5, 0.03)
        pipeline.update_performance("active", -0.2, -0.01)

        assert len(pipeline.model_performance["active"]["predictions"]) == 2
        assert len(pipeline.model_performance["active"]["returns"]) == 2

    def test_save_and_load_model(self, temp_dir):
        """Test model persistence."""
        from crypto_analysis.online.generator import OnlineSignalGenerator

        pipeline = ContinuousLearningPipeline(checkpoint_dir=temp_dir)
        pipeline.active_model = OnlineSignalGenerator(name="Test")

        pipeline._save_model(pipeline.active_model, "test_model.pkl")

        assert (Path(temp_dir) / "test_model.pkl").exists()

        new_pipeline = ContinuousLearningPipeline(checkpoint_dir=temp_dir)
        new_pipeline.load_model("test_model.pkl")

        assert new_pipeline.active_model is not None
        assert new_pipeline.active_model.name == "Test"

    def test_promote_candidate(self, temp_dir):
        """Test candidate promotion."""
        from crypto_analysis.online.generator import OnlineSignalGenerator

        pipeline = ContinuousLearningPipeline(checkpoint_dir=temp_dir)
        pipeline.active_model = OnlineSignalGenerator(name="Active")
        pipeline.candidate_model = OnlineSignalGenerator(name="Candidate")
        pipeline.ab_test_active = True

        pipeline.model_performance["active"] = {
            "predictions": [],
            "returns": [0.01] * 100,
        }
        pipeline.model_performance["candidate"] = {
            "predictions": [],
            "returns": [0.02] * 100,
        }

        pipeline._evaluate_ab_test()

        assert pipeline.active_model.name == "Candidate"
        assert pipeline.candidate_model is None
        assert pipeline.ab_test_active is False
