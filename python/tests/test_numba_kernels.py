"""
Tests for Numba-optimized kernels.
"""

import numpy as np
import pytest
from blockwise_statistical_reductions._numba_kernels import (
    blockwise_mean,
    product_mean,
    product_moments,
)


def test_blockwise_mean_2d():
    """Test 2D blockwise mean."""
    # Create 4x4 array with known values
    data = np.arange(16.0).reshape(4, 4)
    
    # 2x2 blocks
    result = blockwise_mean(data, (2, 2))
    
    assert result.shape == (2, 2)
    
    # Block (0,0): mean([0,1,4,5]) = 2.5
    assert result[0, 0] == pytest.approx(2.5)
    
    # Block (1,1): mean([10,11,14,15]) = 12.5
    assert result[1, 1] == pytest.approx(12.5)


def test_blockwise_mean_3d():
    """Test 3D blockwise mean."""
    # Create 4x4x4 array
    data = np.arange(64.0).reshape(4, 4, 4)
    
    # 2x2x2 blocks
    result = blockwise_mean(data, (2, 2, 2))
    
    assert result.shape == (2, 2, 2)
    
    # Each block should have correct mean
    # Block (0,0,0) contains values 0,1,2,3,4,5,6,7
    expected_mean = np.mean([0, 1, 2, 3, 4, 5, 6, 7])
    assert result[0, 0, 0] == pytest.approx(expected_mean)


def test_product_mean():
    """Test product mean <x*y> without intermediate allocation."""
    # Create arrays where y = 2x
    x = np.arange(16.0).reshape(4, 4)
    y = 2 * x
    
    # 2x2 blocks
    result = product_mean(x, y, (2, 2))
    
    assert result.shape == (2, 2)
    
    # Block (0,0): x=[0,1,4,5], y=[0,2,8,10]
    # Products: [0, 2, 32, 50], mean = 21.0
    x_block = np.array([0, 1, 4, 5])
    y_block = 2 * x_block
    expected = np.mean(x_block * y_block)
    assert result[0, 0] == pytest.approx(expected)


def test_product_moments():
    """Test joint moments computation in one pass."""
    # Create correlated arrays
    x = np.random.randn(8, 8, 8)
    y = 2 * x + np.random.randn(8, 8, 8) * 0.1
    
    # 2x2x2 blocks
    results = product_moments(x, y, (2, 2, 2))
    
    assert results['mean_x'].shape == (4, 4, 4)
    assert results['mean_y'].shape == (4, 4, 4)
    assert results['var_x'].shape == (4, 4, 4)
    assert results['var_y'].shape == (4, 4, 4)
    assert results['cov_xy'].shape == (4, 4, 4)
    assert results['mean_xy'].shape == (4, 4, 4)
    
    # Verify covariance identity: Cov(x,y) = <xy> - <x><y>
    # (approximately, due to numerical precision)
    for i in range(4):
        for j in range(4):
            for k in range(4):
                cov = results['cov_xy'][i, j, k]
                mean_xy = results['mean_xy'][i, j, k]
                mean_x = results['mean_x'][i, j, k]
                mean_y = results['mean_y'][i, j, k]
                
                # Should be close to identity
                computed_cov = mean_xy - mean_x * mean_y
                assert cov == pytest.approx(computed_cov, abs=1e-10)


def test_product_moments_vs_explicit():
    """Compare product_moments to explicit computation."""
    # Create test data
    np.random.seed(42)
    x = np.random.randn(4, 4, 4)
    y = np.random.randn(4, 4, 4)
    
    # Compute with product_moments
    results = product_moments(x, y, (2, 2, 2))
    
    # Check first block explicitly
    x_block = x[0:2, 0:2, 0:2].flatten()
    y_block = y[0:2, 0:2, 0:2].flatten()
    
    expected_mean_x = np.mean(x_block)
    expected_mean_y = np.mean(y_block)
    expected_var_x = np.var(x_block, ddof=1)  # Sample variance
    expected_var_y = np.var(y_block, ddof=1)
    expected_cov = np.cov(x_block, y_block, ddof=1)[0, 1]
    
    assert results['mean_x'][0, 0, 0] == pytest.approx(expected_mean_x, abs=1e-10)
    assert results['mean_y'][0, 0, 0] == pytest.approx(expected_mean_y, abs=1e-10)
    assert results['var_x'][0, 0, 0] == pytest.approx(expected_var_x, abs=1e-10)
    assert results['var_y'][0, 0, 0] == pytest.approx(expected_var_y, abs=1e-10)
    assert results['cov_xy'][0, 0, 0] == pytest.approx(expected_cov, abs=1e-10)


def test_blockwise_mean_matches_numpy():
    """Verify blockwise_mean matches manual numpy computation."""
    np.random.seed(42)
    data = np.random.randn(8, 8, 8)
    
    result = blockwise_mean(data, (2, 2, 2))
    
    # Manual computation
    expected = np.zeros((4, 4, 4))
    for i in range(4):
        for j in range(4):
            for k in range(4):
                block = data[i*2:(i+1)*2, j*2:(j+1)*2, k*2:(k+1)*2]
                expected[i, j, k] = np.mean(block)
    
    np.testing.assert_array_almost_equal(result, expected, decimal=10)
