"""
Integration tests for end-to-end workflows.

These tests verify that different components work together correctly.
"""

import numpy as np
import pytest

from blockwise_statistical_reductions.core import WindowConfig, ReductionPlan
from blockwise_statistical_reductions.statistics import (
    rolling_stats,
    blockwise_stats,
    tree_reduce,
    merge_stats,
)
from blockwise_statistical_reductions.backends import CPUBackend, get_backend


class TestEndToEndBlockwise:
    """End-to-end tests for blockwise workflows."""

    def test_simple_blockwise_mean(self):
        """Test simple blockwise mean workflow."""
        # Create test data
        data = np.random.randn(100, 100)

        # Compute blockwise mean
        result = blockwise_stats(data, (10, 10), "mean", strict=True)

        # Verify shape
        assert result.values.shape == (10, 10)

        # Verify values are reasonable
        assert np.all(np.isfinite(result.values))

    def test_blockwise_multiple_stats(self):
        """Test computing multiple statistics on same blocks."""
        data = np.arange(10000).reshape(100, 100).astype(float)

        mean_result = blockwise_stats(data, (10, 10), "mean", strict=True)
        std_result = blockwise_stats(data, (10, 10), "std", strict=True)
        max_result = blockwise_stats(data, (10, 10), "max", strict=True)

        # All should have same shape
        assert mean_result.values.shape == std_result.values.shape == max_result.values.shape

        # Mean should be less than max
        assert np.all(mean_result.values <= max_result.values)


class TestEndToEndRolling:
    """End-to-end tests for rolling window workflows."""

    def test_rolling_vs_manual(self):
        """Test that rolling matches manual window computation."""
        data = np.arange(100).reshape(10, 10).astype(float)
        config = WindowConfig(sizes=(5, 5), strides=(5, 5))

        # Compute with rolling_stats
        result = rolling_stats(data, config, "mean", strict=True)

        # Manually compute first window
        manual_mean = data[:5, :5].mean()

        assert result.values[0, 0] == pytest.approx(manual_mean)


class TestEndToEndTreeReduce:
    """End-to-end tests for tree reduction workflows."""

    def test_parallel_block_reduction(self):
        """Test reducing blocks in parallel tree pattern."""
        data = np.ones((100, 100))

        # Create 4 blockwise results
        results = [
            blockwise_stats(data, (10, 10), "mean", strict=True)
            for _ in range(4)
        ]

        # Tree reduce
        final = tree_reduce(results, lambda x: merge_stats(x, method="mean"))

        assert np.allclose(final.values, 1.0)


class TestEndToEndBackend:
    """End-to-end tests with different backends."""

    def test_cpu_backend(self):
        """Test execution with CPU backend."""
        backend = get_backend("cpu")
        data = np.ones((50, 50))

        # Create simple plan
        plan = ReductionPlan()
        plan.add_node("input", "input", data=data)
        plan.add_node("mean", "stats", stat="mean")
        plan.add_edge("input", "mean")
        plan.outputs = ["mean"]

        # Execute
        result = backend.execute(plan, data)
        assert result is not None

    def test_backend_factory(self):
        """Test backend factory function."""
        cpu_backend = get_backend("cpu")
        assert cpu_backend is not None


class TestEndToEndValidation:
    """End-to-end tests with strict validation."""

    def test_strict_validation_catches_error(self):
        """Test that strict validation catches dimension mismatches."""
        data = np.zeros((105, 100))  # Not divisible by 10

        with pytest.raises(ValueError, match="not divisible"):
            blockwise_stats(data, (10, 10), "mean", strict=True)

    def test_non_strict_allows_partial(self):
        """Test that non-strict allows partial windows."""
        data = np.zeros((105, 100))

        # Should not raise with strict=False
        result = blockwise_stats(data, (10, 10), "mean", strict=False)
        assert result is not None


class TestEndToEndLargeArrays:
    """Tests with larger arrays for performance validation."""

    def test_large_array_blockwise(self):
        """Test blockwise on large array."""
        data = np.random.randn(1000, 1000)

        result = blockwise_stats(data, (50, 50), "mean", strict=True)

        assert result.values.shape == (20, 20)
        assert np.all(np.isfinite(result.values))


class TestEndToEndEdgeCases:
    """Edge case integration tests."""

    def test_single_block(self):
        """Test with single block covering entire array."""
        data = np.random.randn(10, 10)
        result = blockwise_stats(data, (10, 10), "mean", strict=True)

        assert result.values.shape == (1, 1)
        assert result.values[0, 0] == pytest.approx(data.mean())

    def test_nan_handling(self):
        """Test NaN handling in end-to-end workflow."""
        data = np.random.randn(100, 100)
        data[0:10, 0:10] = np.nan  # Add NaN block

        result = blockwise_stats(data, (10, 10), "mean", strict=True)

        # First block should be NaN, others finite
        assert np.isnan(result.values[0, 0])
        assert np.all(np.isfinite(result.values[1:, 1:]))

    def test_very_small_windows(self):
        """Test with 1x1 windows."""
        data = np.arange(100).reshape(10, 10).astype(float)
        result = blockwise_stats(data, (1, 1), "mean", strict=True)

        # Should return original array
        assert result.values.shape == (10, 10)
        assert np.allclose(result.values, data)


@pytest.mark.skipif(
    pytest.importorskip("dask", reason="Dask not installed") is None,
    reason="Dask not installed",
)
class TestEndToEndDask:
    """End-to-end tests with Dask integration."""

    def test_dask_blockwise_matches_numpy(self):
        """Test that Dask gives same results as NumPy."""
        import dask.array as da
        from blockwise_statistical_reductions.dask_integration import dask_blockwise_stats

        np_arr = np.random.randn(100, 100)
        darr = da.from_array(np_arr, chunks=(50, 50))

        dask_result = dask_blockwise_stats(darr, (10, 10), "mean").compute()
        numpy_result = blockwise_stats(np_arr, (10, 10), "mean", strict=True).values

        assert np.allclose(dask_result, numpy_result)


@pytest.mark.skipif(
    pytest.importorskip("xarray", reason="Xarray not installed") is None,
    reason="Xarray not installed",
)
class TestEndToEndXarray:
    """End-to-end tests with Xarray integration."""

    def test_xarray_workflow(self):
        """Test complete xarray workflow."""
        import xarray as xr
        from blockwise_statistical_reductions.xarray_integration import xr_blockwise_stats

        da = xr.DataArray(
            np.random.randn(100, 100),
            dims=["lon", "lat"],
            coords={"lon": np.linspace(0, 360, 100), "lat": np.linspace(-90, 90, 100)},
            attrs={"units": "m/s"},
            name="velocity",
        )

        result = xr_blockwise_stats(da, {"lon": 10, "lat": 10}, "mean", strict=True)

        assert result.sizes == {"lon": 10, "lat": 10}
        assert "units" in result.attrs
        assert result.name == "velocity_mean_blockwise"
