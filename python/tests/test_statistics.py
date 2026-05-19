"""
Unit tests for statistics module.
"""

import numpy as np
import pytest

from blockwise_statistical_reductions.core import WindowConfig
from blockwise_statistical_reductions.statistics import (
    window_stat,
    rolling_stats,
    blockwise_stats,
    merge_stats,
    tiled_stats,
    tree_reduce,
)


class TestWindowStat:
    """Tests for single window statistics."""

    def test_mean(self):
        """Test mean computation."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = window_stat(data, "mean")
        assert result == pytest.approx(3.0)

    def test_var(self):
        """Test variance computation."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = window_stat(data, "var")
        # Population variance
        expected = np.var(data)
        assert result == pytest.approx(expected)

    def test_nan_handling(self):
        """Test NaN handling in statistics."""
        data = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        result = window_stat(data, "mean")
        expected = np.nanmean(data)
        assert result == pytest.approx(expected)


class TestRollingStats:
    """Tests for rolling statistics."""

    def test_rolling_mean_1d(self):
        """Test 1D rolling mean."""
        arr = np.arange(20, dtype=float)
        config = WindowConfig(sizes=(5,), strides=(5,))
        result = rolling_stats(arr, config, "mean", strict=True)

        # 4 windows: [0-4], [5-9], [10-14], [15-19]
        assert result.values.shape == (4,)
        assert result.values[0] == pytest.approx(2.0)  # mean of 0,1,2,3,4
        assert result.values[1] == pytest.approx(7.0)  # mean of 5,6,7,8,9

    def test_rolling_mean_2d(self):
        """Test 2D rolling mean."""
        arr = np.ones((20, 20))
        config = WindowConfig(sizes=(10, 10))
        result = rolling_stats(arr, config, "mean", strict=True)

        # 2x2 grid of blocks
        assert result.values.shape == (2, 2)
        assert np.allclose(result.values, 1.0)

    def test_rolling_sum(self):
        """Test rolling sum."""
        arr = np.ones((10, 10))
        config = WindowConfig(sizes=(5, 5))
        result = rolling_stats(arr, config, "sum", strict=True)

        # 2x2 grid, each block sums to 25
        assert result.values.shape == (2, 2)
        assert np.allclose(result.values, 25.0)

    def test_rolling_min_max(self):
        """Test rolling min/max."""
        arr = np.arange(100).reshape(10, 10)
        config = WindowConfig(sizes=(5, 5))
        result_min = rolling_stats(arr, config, "min", strict=True)
        result_max = rolling_stats(arr, config, "max", strict=True)

        assert result_min.values[0, 0] == 0.0
        assert result_max.values[0, 0] == 55.0  # max of top-left 5x5 block


class TestBlockwiseStats:
    """Tests for blockwise statistics."""

    def test_blockwise_mean(self):
        """Test blockwise mean."""
        arr = np.arange(100).reshape(10, 10).astype(float)
        result = blockwise_stats(arr, (5, 5), "mean", strict=True)

        # 2x2 grid
        assert result.values.shape == (2, 2)
        # Top-left block mean: 0-4 rows, 0-4 cols
        expected_tl = arr[:5, :5].mean()
        assert result.values[0, 0] == pytest.approx(expected_tl)

    def test_blockwise_with_inexact_raises(self):
        """Test that inexact block sizes raise error."""
        arr = np.zeros((11, 10))
        with pytest.raises(ValueError):
            blockwise_stats(arr, (5, 5), "mean", strict=True)


class TestMergeStats:
    """Tests for statistic merging."""

    def test_merge_mean(self):
        """Test merging mean statistics."""
        result1 = rolling_stats(np.ones((10, 10)), WindowConfig((5, 5)), "mean", strict=True)
        result2 = rolling_stats(np.ones((10, 10)), WindowConfig((5, 5)), "mean", strict=True)

        merged = merge_stats([result1, result2], method="mean")

        assert merged.values.shape == (2, 2)
        assert np.allclose(merged.values, 1.0)

    def test_merge_sum(self):
        """Test merging with sum method."""
        result1 = rolling_stats(np.ones((10, 10)), WindowConfig((5, 5)), "mean", strict=True)
        result2 = rolling_stats(np.ones((10, 10)), WindowConfig((5, 5)), "mean", strict=True)

        merged = merge_stats([result1, result2], method="sum")

        assert np.allclose(merged.values, 2.0)

    def test_merge_single_element(self):
        """Test merging single element returns it."""
        result = rolling_stats(np.ones((10, 10)), WindowConfig((5, 5)), "mean", strict=True)
        merged = merge_stats([result])

        assert np.allclose(merged.values, result.values)

    def test_merge_empty_raises(self):
        """Test merging empty list raises error."""
        with pytest.raises(ValueError, match="empty"):
            merge_stats([])


class TestTiledStats:
    """Tests for tiled statistics (mergeable)."""

    def test_tiled_returns_count(self):
        """Test that tiled stats include count for weighting."""
        arr = np.ones((10, 10))
        config = WindowConfig((5, 5))
        results = tiled_stats(arr, config, "mean")

        # Should return (value, count) tuples
        assert len(results) == 4  # 2x2 grid
        value, count = results[0]
        assert count == 25  # 5x5 block

    def test_tiled_requires_blockwise(self):
        """Test that tiled enforces non-overlapping."""
        # This should still work with strict=True (non-overlapping)
        arr = np.ones((10, 10))
        config = WindowConfig((5, 5))  # stride=size, so non-overlapping
        results = tiled_stats(arr, config, "mean")
        assert len(results) == 4


class TestTreeReduce:
    """Tests for tree reduction."""

    def test_tree_reduce_list(self):
        """Test tree reduction of a list."""
        data = [1, 2, 3, 4, 5, 6, 7, 8]
        result = tree_reduce(data, lambda x: sum(x), arity=2)

        assert result == sum(data)

    def test_tree_reduce_empty(self):
        """Test tree reduce with empty list."""
        result = tree_reduce([], lambda x: sum(x))
        assert result is None

    def test_tree_reduce_single(self):
        """Test tree reduce with single element."""
        result = tree_reduce([42], lambda x: sum(x))
        assert result == 42

    def test_tree_reduce_reduction_results(self):
        """Test tree reduce with ReductionResult objects."""
        arr = np.ones((10, 10))
        results = [
            blockwise_stats(arr, (5, 5), "mean", strict=True)
            for _ in range(4)
        ]

        def merge_fn(chunk):
            return merge_stats(chunk, method="mean")

        result = tree_reduce(results, merge_fn, arity=2)

        assert np.allclose(result.values, 1.0)


class TestNumbaOptimization:
    """Tests for Numba-optimized paths."""

    def test_numba_kernel_used(self):
        """Test that numba kernel is used when appropriate."""
        arr = np.random.randn(100, 100)
        config = WindowConfig((10, 10))

        # Both paths should give same result
        result_numba = rolling_stats(arr, config, "mean", use_numba=True, strict=True)
        result_python = rolling_stats(arr, config, "mean", use_numba=False, strict=True)

        assert np.allclose(result_numba.values, result_python.values, rtol=1e-10)

    def test_numba_faster_for_large(self):
        """Smoke test that numba path works for large arrays."""
        arr = np.random.randn(1000, 1000)
        config = WindowConfig((50, 50))

        # Should complete without error
        result = rolling_stats(arr, config, "mean", use_numba=True, strict=True)
        assert result.values.shape == (20, 20)
