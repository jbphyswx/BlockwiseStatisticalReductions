"""
Pytest configuration and fixtures.
"""

import numpy as np
import pytest


@pytest.fixture
def sample_2d_array():
    """Return a sample 2D array for testing."""
    np.random.seed(42)
    return np.random.randn(100, 100)


@pytest.fixture
def sample_3d_array():
    """Return a sample 3D array for testing."""
    np.random.seed(42)
    return np.random.randn(50, 50, 20)


@pytest.fixture
def blockwise_config_2d():
    """Return a standard 2D blockwise window config."""
    from blockwise_statistical_reductions.core import WindowConfig
    return WindowConfig(sizes=(10, 10))


@pytest.fixture
def rolling_config_2d():
    """Return a 2D rolling window config with overlap."""
    from blockwise_statistical_reductions.core import WindowConfig
    return WindowConfig(sizes=(10, 10), strides=(5, 5))


@pytest.fixture(scope="session")
def dask_client():
    """Create a Dask client for testing (session-scoped)."""
    try:
        from dask.distributed import Client, LocalCluster

        cluster = LocalCluster(
            n_workers=2,
            threads_per_worker=2,
            processes=False,  # Use threads for faster test setup
            silence_logs=False,
        )
        client = Client(cluster)

        yield client

        client.close()
        cluster.close()
    except ImportError:
        pytest.skip("Dask not installed")


@pytest.fixture
def sample_xarray_da():
    """Return a sample xarray DataArray."""
    try:
        import xarray as xr

        return xr.DataArray(
            np.random.randn(100, 100),
            dims=["x", "y"],
            coords={"x": np.arange(100), "y": np.arange(100)},
            attrs={"units": "m/s", "description": "velocity"},
            name="u",
        )
    except ImportError:
        pytest.skip("Xarray not installed")
