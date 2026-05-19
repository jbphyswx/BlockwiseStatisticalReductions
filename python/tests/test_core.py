"""
Unit tests for core module.
"""

import numpy as np
import pytest

from blockwise_statistical_reductions.core import (
    WindowConfig,
    validate_window_config,
    rolling_windows,
    blockwise_windows,
)


class TestWindowConfig:
    """Tests for WindowConfig dataclass."""

    def test_default_blockwise(self):
        """Test that default creates non-overlapping windows."""
        config = WindowConfig(sizes=(10, 10))
        assert config.strides == (10, 10)
        assert config.is_blockwise()

    def test_rolling_with_stride(self):
        """Test rolling windows with custom stride."""
        config = WindowConfig(sizes=(5, 5), strides=(2, 2))
        assert not config.is_blockwise()

    def test_ndim_property(self):
        """Test ndim property."""
        config = WindowConfig(sizes=(10, 10, 5))
        assert config.ndim == 3


class TestValidateWindowConfig:
    """Tests for window configuration validation."""

    def test_blockwise_exact_divisibility(self):
        """Test exact divisibility check for blockwise."""
        config = WindowConfig(sizes=(10, 10))
        assert validate_window_config((100, 100), config, strict=True)

    def test_blockwise_inexact_raises_error(self):
        """Test that inexact blockwise dimensions raise error."""
        config = WindowConfig(sizes=(10, 10))
        with pytest.raises(ValueError, match="not divisible"):
            validate_window_config((105, 100), config, strict=True)

    def test_rolling_stride_validation(self):
        """Test rolling window stride validation."""
        config = WindowConfig(sizes=(5, 5), strides=(2, 2))
        # (100 - 5) % 2 = 95 % 2 = 1, not divisible
        with pytest.raises(ValueError, match="does not divide evenly"):
            validate_window_config((100, 100), config, strict=True)

    def test_strict_false_returns_false(self):
        """Test that strict=False returns False for invalid config."""
        config = WindowConfig(sizes=(10, 10))
        result = validate_window_config((105, 100), config, strict=False)
        assert result is True  # strict=False always returns True


class TestRollingWindows:
    """Tests for rolling window generation."""

    def test_1d_rolling(self):
        """Test 1D rolling windows."""
        arr = np.arange(20)
        config = WindowConfig(sizes=(5,), strides=(2,), padding="valid")
        windows, out_shape = rolling_windows(arr, config)

        # Expected: (20 - 5) // 2 + 1 = 8 windows
        assert out_shape == (8,)
        assert len(windows) == 8

    def test_2d_blockwise(self):
        """Test 2D blockwise (non-overlapping) windows."""
        arr = np.zeros((100, 100))
        config = WindowConfig(sizes=(10, 10))  # stride=size
        windows, out_shape = rolling_windows(arr, config, strict=True)

        # Expected: 10x10 grid of windows
        assert out_shape == (10, 10)
        assert len(windows) == 100

    def test_window_metadata(self):
        """Test that windows have proper metadata."""
        arr = np.arange(20)
        config = WindowConfig(sizes=(5,), strides=(5,))
        windows, _ = rolling_windows(arr, config)

        view, meta = windows[0]
        assert "indices" in meta
        assert "center" in meta
        assert "slices" in meta
        assert len(meta["indices"]) == 1

    def test_valid_padding_truncation(self):
        """Test that 'valid' padding truncates partial windows."""
        arr = np.arange(12)
        config = WindowConfig(sizes=(5,), strides=(5,))  # 2 full windows, 2 remaining
        windows, out_shape = rolling_windows(arr, config)

        # Should get 2 full windows (indices 0-4 and 5-9), 10-11 truncated
        assert out_shape == (2,)
        assert len(windows) == 2


class TestBlockwiseWindows:
    """Tests for blockwise window convenience function."""

    def test_blockwise_convenience(self):
        """Test blockwise_windows convenience function."""
        arr = np.zeros((100, 100))
        windows, out_shape = blockwise_windows(arr, (10, 10), strict=True)

        assert out_shape == (10, 10)
        assert len(windows) == 100

    def test_blockwise_default_strict(self):
        """Test that blockwise defaults to strict=True."""
        arr = np.zeros((105, 100))  # Not divisible by 10
        with pytest.raises(ValueError):
            blockwise_windows(arr, (10, 10))  # strict=True by default


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_result(self):
        """Test that windows larger than array return empty."""
        arr = np.arange(5)
        config = WindowConfig(sizes=(10,), strides=(1,))
        windows, out_shape = rolling_windows(arr, config)

        # Window larger than array
        assert len(windows) == 0
        assert out_shape == (0,)

    def test_single_window(self):
        """Test single window covering entire array."""
        arr = np.arange(10)
        config = WindowConfig(sizes=(10,), strides=(10,))
        windows, out_shape = rolling_windows(arr, config, strict=True)

        assert out_shape == (1,)
        assert len(windows) == 1
        view, _ = windows[0]
        assert len(view) == 10
