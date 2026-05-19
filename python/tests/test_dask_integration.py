"""
Unit and integration tests for Dask integration.
"""

import numpy as np
import pytest

# Skip all Dask tests if not installed
dask = pytest.importorskip("dask")
distributed = pytest.importorskip("distributed")

import dask.array as da
from dask.distributed import Client, LocalCluster

from blockwise_statistical_reductions.core import WindowConfig
from blockwise_statistical_reductions.dask_integration import (
    dask_rolling_stats,
    dask_blockwise_stats,
    create_dask_plan,
    execute_dask_plan,
)
from blockwise_statistical_reductions.statistics import rolling_stats


@pytest.fixture(scope="module")
def dask_client():
    """Create a local Dask cluster for testing."""
    cluster = LocalCluster(n_workers=2, threads_per_worker=2, processes=False)
    client = Client(cluster)
    yield client
    client.close()
    cluster.close()


class TestDaskRollingStats:
    """Tests for Dask rolling statistics."""

    def test_dask_rolling_mean(self, dask_client):
        """Test rolling mean on Dask array."""
        # Create dask array
        np_arr = np.arange(100).reshape(10, 10).astype(float)
        darr = da.from_array(np_arr, chunks=(5, 5))

        config = WindowConfig(sizes=(5, 5))
        result = dask_rolling_stats(darr, config, "mean")

        # Compute and verify
        result_np = result.compute()
        expected = rolling_stats(np_arr, config, "mean", strict=True).values

        assert result_np.shape == (2, 2)
        assert np.allclose(result_np, expected)

    def test_dask_lazy_evaluation(self, dask_client):
        """Test that Dask arrays remain lazy."""
        np_arr = np.ones((100, 100))
        darr = da.from_array(np_arr, chunks=(50, 50))

        config = WindowConfig(sizes=(10, 10))
        result = dask_rolling_stats(darr, config, "mean")

        # Should be a Dask array, not computed yet
        assert isinstance(result, da.Array)
        assert result.chunks is not None


class TestDaskBlockwiseStats:
    """Tests for Dask blockwise statistics."""

    def test_dask_blockwise_mean(self, dask_client):
        """Test blockwise mean on Dask array."""
        np_arr = np.arange(100).reshape(10, 10).astype(float)
        darr = da.from_array(np_arr, chunks=(5, 5))

        result = dask_blockwise_stats(darr, (5, 5), "mean")
        result_np = result.compute()

        # Should be 2x2 grid
        assert result_np.shape == (2, 2)

    def test_dask_blockwise_matches_numpy(self, dask_client):
        """Test that Dask and NumPy give same results."""
        np_arr = np.random.randn(20, 20)
        darr = da.from_array(np_arr, chunks=(10, 10))

        from blockwise_statistical_reductions.statistics import blockwise_stats

        dask_result = dask_blockwise_stats(darr, (5, 5), "mean").compute()
        numpy_result = blockwise_stats(np_arr, (5, 5), "mean", strict=False).values

        assert np.allclose(dask_result, numpy_result)


class TestDaskPlanExecution:
    """Tests for Dask plan execution."""

    def test_create_dask_plan(self):
        """Test creation of Dask task graph."""
        plan = create_dask_plan(
            input_shape=(100, 100),
            operations=[
                {"op": "window", "config": WindowConfig((10, 10))},
                {"op": "stats", "stat": "mean"},
            ],
        )

        assert "input" in plan
        assert "op-0" in plan
        assert "op-1" in plan


class TestDaskDistributed:
    """Integration tests for Dask distributed."""

    def test_scatter_to_workers(self, dask_client):
        """Test scattering data to workers."""
        from blockwise_statistical_reductions.dask_integration import scatter_to_workers

        data = np.arange(100).reshape(10, 10)
        futures = scatter_to_workers(dask_client, data, n_partitions=2)

        assert len(futures) == 2

    def test_distributed_execution(self, dask_client):
        """Test execution on distributed cluster."""
        np_arr = np.random.randn(100, 100)
        darr = da.from_array(np_arr, chunks=(50, 50))

        config = WindowConfig(sizes=(10, 10))
        result = dask_rolling_stats(darr, config, "mean")

        # Use distributed scheduler
        result_computed = result.compute(scheduler=dask_client)

        assert result_computed.shape == (10, 10)


class TestDaskFlox:
    """Tests for Flox integration."""

    def test_flox_reduce_available(self):
        """Test that flox is available."""
        flox = pytest.importorskip("flox")
        assert flox is not None


class TestDaskEdgeCases:
    """Tests for edge cases with Dask."""

    def test_uneven_chunks(self, dask_client):
        """Test handling of uneven chunks."""
        np_arr = np.arange(105).reshape(21, 5).astype(float)
        # Uneven chunks: 10 + 11
        darr = da.from_array(np_arr, chunks=((10, 11), (5,)))

        config = WindowConfig(sizes=(5, 5))
        result = dask_rolling_stats(darr, config, "mean")

        # Should handle boundary correctly
        result_np = result.compute()
        assert result_np.shape[0] == 4  # (21 - 5) // 5 + 1 = 4

    def test_empty_chunks(self, dask_client):
        """Test handling of edge chunks."""
        np_arr = np.ones((10, 10))
        darr = da.from_array(np_arr, chunks=(3, 3))

        config = WindowConfig(sizes=(5, 5))
        result = dask_rolling_stats(darr, config, "mean")

        # Should not error
        result_np = result.compute()
        assert result_np.shape == (2, 2)
