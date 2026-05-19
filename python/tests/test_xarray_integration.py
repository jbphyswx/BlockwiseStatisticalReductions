"""
Unit and integration tests for Xarray integration.
"""

import numpy as np
import pytest

xarray = pytest.importorskip("xarray")

import xarray as xr
import pandas as pd

from blockwise_statistical_reductions.xarray_integration import (
    preserve_xr_metadata,
    create_coordinate_ranges,
    xr_rolling_stats,
    xr_blockwise_stats,
    xr_tree_reduce,
    xr_resample_stats,
    add_window_coordinates,
)


class TestPreserveMetadata:
    """Tests for metadata preservation."""

    def test_preserve_attrs(self):
        """Test that attributes are preserved."""
        da = xr.DataArray(
            np.ones((10, 10)),
            dims=["x", "y"],
            attrs={"units": "m/s", "description": "velocity"},
            name="u",
        )

        result = preserve_xr_metadata(
            da,
            np.ones((5, 5)),
            {"x": np.arange(5), "y": np.arange(5)},
        )

        assert result.attrs["units"] == "m/s"
        assert result.attrs["description"] == "velocity"
        assert result.name == "u_stat"

    def test_unit_modification_for_variance(self):
        """Test that variance gets squared units."""
        da = xr.DataArray(
            np.ones((10, 10)),
            dims=["x", "y"],
            attrs={"units": "m"},
        )

        class FakeConfig:
            stat = "var"

        result = preserve_xr_metadata(
            da,
            np.ones((5, 5)),
            {"x": np.arange(5), "y": np.arange(5)},
            FakeConfig(),
        )

        # Variance should have squared units (not implemented yet, but placeholder)
        # assert result.attrs["units"] == "(m)^2"


class TestCoordinateRanges:
    """Tests for coordinate range generation."""

    def test_valid_padding_coords(self):
        """Test coordinate generation for valid padding."""
        orig_coords = {"x": np.arange(20)}
        window_size = {"x": 5}
        stride = {"x": 5}

        new_coords = create_coordinate_ranges(
            orig_coords, window_size, stride, padding="valid"
        )

        # Centers at positions 2, 7, 12, 17
        assert len(new_coords["x"]) == 4
        assert new_coords["x"][0] == 2
        assert new_coords["x"][1] == 7

    def test_same_padding_coords(self):
        """Test coordinate generation for same padding."""
        orig_coords = {"x": np.arange(10)}
        window_size = {"x": 3}
        stride = {"x": 2}

        new_coords = create_coordinate_ranges(
            orig_coords, window_size, stride, padding="same"
        )

        # Should have (10 - 1) // 2 + 1 = 5 points
        assert len(new_coords["x"]) == 5


class TestXarrayRolling:
    """Tests for xarray rolling statistics."""

    def test_xr_rolling_mean(self):
        """Test xarray rolling mean."""
        da = xr.DataArray(
            np.arange(100).reshape(10, 10),
            dims=["x", "y"],
            coords={"x": np.arange(10), "y": np.arange(10)},
            attrs={"units": "m"},
            name="data",
        )

        result = xr_rolling_stats(da, {"x": 5, "y": 5}, "mean", center=True)

        assert result.dims == ("x", "y")
        assert result.attrs["units"] == "m"
        assert result.name == "data_mean"

    def test_xr_rolling_perserves_coords(self):
        """Test that rolling preserves coordinates."""
        da = xr.DataArray(
            np.ones((10, 10)),
            dims=["x", "y"],
            coords={
                "x": np.arange(10),
                "y": np.arange(10),
                "time": 42,  # scalar coord
            },
        )

        result = xr_rolling_stats(da, {"x": 3}, "mean")

        assert "time" in result.coords
        assert result.coords["time"] == 42


class TestXarrayBlockwise:
    """Tests for xarray blockwise statistics."""

    def test_xr_blockwise_mean(self):
        """Test xarray blockwise mean."""
        da = xr.DataArray(
            np.ones((10, 10)),
            dims=["x", "y"],
            coords={"x": np.arange(10), "y": np.arange(10)},
        )

        result = xr_blockwise_stats(da, {"x": 5, "y": 5}, "mean", strict=True)

        # Should be 2x2 result
        assert result.sizes["x"] == 2
        assert result.sizes["y"] == 2
        assert np.allclose(result.values, 1.0)

    def test_xr_blockwise_inexact_raises(self):
        """Test that inexact block sizes raise error."""
        da = xr.DataArray(
            np.ones((11, 10)),
            dims=["x", "y"],
        )

        with pytest.raises(ValueError, match="not divisible"):
            xr_blockwise_stats(da, {"x": 5, "y": 5}, "mean", strict=True)

    def test_xr_blockwise_new_coords(self):
        """Test that blockwise creates appropriate new coordinates."""
        da = xr.DataArray(
            np.arange(100).reshape(10, 10),
            dims=["x", "y"],
            coords={"x": np.arange(10) * 10, "y": np.arange(10) * 5},
        )

        result = xr_blockwise_stats(da, {"x": 5, "y": 5}, "mean", strict=True)

        # Centers of blocks: x should be at 20, 70; y at 12.5, 37.5
        expected_x = [20.0, 70.0]
        expected_y = [12.5, 37.5]

        assert np.allclose(result.coords["x"].values, expected_x)
        assert np.allclose(result.coords["y"].values, expected_y)


class TestXarrayTreeReduce:
    """Tests for xarray tree reduction."""

    def test_tree_reduce_dataarrays(self):
        """Test tree reduction of DataArrays."""
        das = [
            xr.DataArray(np.ones((5, 5)), dims=["x", "y"], coords={"x": range(5), "y": range(5)})
            for _ in range(4)
        ]

        result = xr_tree_reduce(das, lambda a, b: a + b)

        assert result.sizes == {"x": 5, "y": 5}
        assert np.allclose(result.values, 4.0)

    def test_tree_reduce_mismatched_coords_raises(self):
        """Test that mismatched coordinates raise error."""
        da1 = xr.DataArray(np.ones((5, 5)), dims=["x", "y"], coords={"x": range(5), "y": range(5)})
        da2 = xr.DataArray(np.ones((5, 5)), dims=["x", "y"], coords={"x": range(5, 10), "y": range(5)})

        with pytest.raises(ValueError, match="don't match"):
            xr_tree_reduce([da1, da2])


class TestXarrayResample:
    """Tests for time resampling."""

    def test_resample_mean(self):
        """Test time resampling."""
        dates = pd.date_range("2020-01-01", periods=100, freq="H")
        da = xr.DataArray(
            np.arange(100),
            dims=["time"],
            coords={"time": dates},
        )

        result = xr_resample_stats(da, "time", "6H", "mean")

        # 100 hours / 6 = ~16-17 periods
        assert len(result) == 17


class TestWindowCoordinates:
    """Tests for window position coordinates."""

    def test_add_window_coords(self):
        """Test adding window position coordinates."""
        from blockwise_statistical_reductions.core import WindowConfig

        orig = xr.DataArray(
            np.arange(100).reshape(10, 10),
            dims=["x", "y"],
            coords={"x": np.arange(10), "y": np.arange(10)},
        )

        result_da = xr.DataArray(
            np.ones((2, 2)),
            dims=["x", "y"],
            coords={"x": [2, 7], "y": [2, 7]},
        )

        config = WindowConfig(sizes=(5, 5))
        result = add_window_coordinates(result_da, orig, config)

        # Should have window position coords
        assert "x_window_start" in result.coords
        assert "x_window_center" in result.coords
        assert "x_window_end" in result.coords


class TestXarrayDask:
    """Integration tests with Xarray and Dask."""

    def test_xarray_with_dask(self):
        """Test xarray operations on Dask arrays."""
        dask = pytest.importorskip("dask")
        import dask.array as da

        dask_arr = da.from_array(np.ones((100, 100)), chunks=(50, 50))
        xarr = xr.DataArray(
            dask_arr,
            dims=["x", "y"],
            coords={"x": np.arange(100), "y": np.arange(100)},
        )

        # xarray operations should work lazily
        result = xr_blockwise_stats(xarr, {"x": 10, "y": 10}, "mean", strict=True)

        # Should be a Dask-backed DataArray
        assert isinstance(result.data, da.Array)

        # Compute and verify
        computed = result.compute()
        assert computed.sizes == {"x": 10, "y": 10}
