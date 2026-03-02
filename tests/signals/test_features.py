"""Tests for FeatureEngineer."""


from crypto_analysis.signals.features import FeatureEngineer


class TestFeatureEngineer:
    """Test FeatureEngineer class."""

    def test_initialization(self):
        """Test FeatureEngineer initialization."""
        fe = FeatureEngineer()
        assert fe.scaler is not None
        assert fe.feature_names == []

    def test_create_features_basic(self, sample_ohlcv_data):
        """Test basic feature creation."""
        fe = FeatureEngineer()
        features = fe.create_features(sample_ohlcv_data)

        # Should have more columns than original
        assert len(features.columns) > len(sample_ohlcv_data.columns)

        # Should have log_returns
        assert "log_returns" in features.columns

        # Should drop NaN rows
        assert not features.isnull().any().any()

    def test_price_features(self, minimal_ohlcv_data):
        """Test price-based features."""
        fe = FeatureEngineer()
        data = fe._add_price_features(minimal_ohlcv_data)

        assert "log_returns" in data.columns
        assert "close_position" in data.columns
        assert "body_size" in data.columns

    def test_volume_features(self, minimal_ohlcv_data):
        """Test volume features."""
        fe = FeatureEngineer()
        data = fe._add_price_features(minimal_ohlcv_data)
        data = fe._add_volume_features(data)

        assert "volume_ma_10" in data.columns
        assert "volume_ma_30" in data.columns
        assert "volume_ratio" in data.columns
        assert "obv" in data.columns

    def test_volatility_features(self, minimal_ohlcv_data):
        """Test volatility features."""
        fe = FeatureEngineer()
        data = fe._add_price_features(minimal_ohlcv_data)
        data = fe._add_volatility_features(data)

        assert "volatility_5" in data.columns
        assert "volatility_20" in data.columns
        assert "atr_14" in data.columns
        assert "bb_position" in data.columns

    def test_trend_features(self, minimal_ohlcv_data):
        """Test trend features."""
        fe = FeatureEngineer()
        data = fe._add_price_features(minimal_ohlcv_data)
        data = fe._add_volatility_features(data)
        data = fe._add_trend_features(data)

        assert "ma_7" in data.columns
        assert "ma_21" in data.columns
        assert "ma_50" in data.columns
        assert "adx" in data.columns

    def test_momentum_features(self, minimal_ohlcv_data):
        """Test momentum features."""
        fe = FeatureEngineer()
        data = fe._add_momentum_features(minimal_ohlcv_data)

        assert "rsi_14" in data.columns
        assert "macd" in data.columns
        assert "macd_signal" in data.columns
        assert "stoch_k" in data.columns

    def test_time_features(self, minimal_ohlcv_data):
        """Test time features."""
        fe = FeatureEngineer()
        data = fe._add_time_features(minimal_ohlcv_data)

        assert "hour_sin" in data.columns
        assert "hour_cos" in data.columns
        assert "dow_sin" in data.columns
        assert "dow_cos" in data.columns
        assert "is_weekend" in data.columns

    def test_target_generation(self, minimal_ohlcv_data):
        """Test target variable generation."""
        fe = FeatureEngineer()
        data = minimal_ohlcv_data.copy()
        data = fe._add_price_features(data)
        data = fe._add_targets(data)

        assert "target_return_1" in data.columns
        assert "target_return_3" in data.columns
        assert "target_direction_1" in data.columns

    def test_get_feature_columns(self, sample_ohlcv_data):
        """Test getting feature column names."""
        fe = FeatureEngineer()
        features = fe.create_features(sample_ohlcv_data)
        cols = fe.get_feature_columns(features)

        # Should not include OHLCV or targets
        assert "open" not in cols
        assert "close" not in cols
        assert "target_return_1" not in cols
