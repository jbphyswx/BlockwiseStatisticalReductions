"""
Core Numba-optimized kernels for blockwise reductions.

These are low-level performance-critical functions.
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange
from typing import Tuple


@njit(cache=True, parallel=True)
def _blockwise_mean_3d(
    out: np.ndarray,
    data: np.ndarray,
    fx: int,
    fy: int,
    fz: int
) -> None:
    """
    Compute blockwise mean for 3D array.
    
    Parameters
    ----------
    out : (nx, ny, nz) array, preallocated output
    data : (nx*fx, ny*fy, nz*fz) input array
    fx, fy, fz : block sizes in each dimension
    """
    nx, ny, nz = out.shape
    
    for i in prange(nx):
        for j in range(ny):
            for k in range(nz):
                # Block start indices
                i0 = i * fx
                j0 = j * fy
                k0 = k * fz
                
                # Compute mean over block
                s = 0.0
                for ii in range(fx):
                    for jj in range(fy):
                        for kk in range(fz):
                            s += data[i0 + ii, j0 + jj, k0 + kk]
                
                out[i, j, k] = s / (fx * fy * fz)


@njit(cache=True, parallel=True)
def _blockwise_mean_2d(
    out: np.ndarray,
    data: np.ndarray,
    fx: int,
    fy: int
) -> None:
    """Compute blockwise mean for 2D array."""
    nx, ny = out.shape
    
    for i in prange(nx):
        for j in range(ny):
            i0 = i * fx
            j0 = j * fy
            
            s = 0.0
            for ii in range(fx):
                for jj in range(fy):
                    s += data[i0 + ii, j0 + jj]
            
            out[i, j] = s / (fx * fy)


@njit(cache=True, parallel=True)
def _blockwise_variance_3d(
    out: np.ndarray,
    data: np.ndarray,
    fx: int,
    fy: int,
    fz: int,
    corrected: bool = True
) -> None:
    """
    Compute blockwise variance for 3D array using Welford's algorithm.
    """
    nx, ny, nz = out.shape
    block_size = fx * fy * fz
    
    for i in prange(nx):
        for j in range(ny):
            for k in range(nz):
                i0 = i * fx
                j0 = j * fy
                k0 = k * fz
                
                # Welford's algorithm
                n = 0
                mean = 0.0
                m2 = 0.0
                
                for ii in range(fx):
                    for jj in range(fy):
                        for kk in range(fz):
                            x = data[i0 + ii, j0 + jj, k0 + kk]
                            n += 1
                            delta = x - mean
                            mean += delta / n
                            delta2 = x - mean
                            m2 += delta * delta2
                
                # Variance
                if corrected:
                    variance = m2 / (block_size - 1) if block_size > 1 else 0.0
                else:
                    variance = m2 / block_size
                
                out[i, j, k] = variance


@njit(cache=True, parallel=True)
def _product_mean_3d(
    out: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    fx: int,
    fy: int,
    fz: int
) -> None:
    """
    Compute mean of products <x*y> without materializing intermediate array.
    
    This is the product coarsening operation - fuses the multiplication
    into the accumulation loop to avoid allocating x*y.
    """
    nx, ny, nz = out.shape
    
    for i in prange(nx):
        for j in range(ny):
            for k in range(nz):
                i0 = i * fx
                j0 = j * fy
                k0 = k * fz
                
                s = 0.0
                for ii in range(fx):
                    for jj in range(fy):
                        for kk in range(fz):
                            s += x[i0 + ii, j0 + jj, k0 + kk] * y[i0 + ii, j0 + jj, k0 + kk]
                
                out[i, j, k] = s / (fx * fy * fz)


@njit(cache=True, parallel=True)
def _product_moments_3d(
    mean_x: np.ndarray,
    mean_y: np.ndarray,
    mean_xy: np.ndarray,
    var_x: np.ndarray,
    var_y: np.ndarray,
    cov_xy: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    fx: int,
    fy: int,
    fz: int,
    corrected: bool = True
) -> None:
    """
    Compute joint moments (means, variances, covariance) in one fused pass.
    
    All six output arrays are computed simultaneously without intermediate
    allocations. This is the most efficient way to get covariance.
    """
    nx, ny, nz = mean_x.shape
    
    for i in prange(nx):
        for j in range(ny):
            for k in range(nz):
                i0 = i * fx
                j0 = j * fy
                k0 = k * fz
                
                # Accumulators for this block
                n = 0
                mx, my = 0.0, 0.0
                m2_x, m2_y = 0.0, 0.0
                cross_dev = 0.0
                
                for ii in range(fx):
                    for jj in range(fy):
                        for kk in range(fz):
                            xv = x[i0 + ii, j0 + jj, k0 + kk]
                            yv = y[i0 + ii, j0 + jj, k0 + kk]
                            
                            n += 1
                            
                            # Online mean updates
                            dx = xv - mx
                            dy = yv - my
                            mx += dx / n
                            my += dy / n
                            
                            # Welford M2 updates
                            m2_x += dx * (xv - mx)
                            m2_y += dy * (yv - my)
                            
                            # Cross-deviation (using updated means)
                            cross_dev += dx * (yv - my)
                
                # Store results
                mean_x[i, j, k] = mx
                mean_y[i, j, k] = my
                
                block_size = fx * fy * fz
                denom = block_size - 1 if corrected else block_size
                
                var_x[i, j, k] = m2_x / denom if denom > 0 else 0.0
                var_y[i, j, k] = m2_y / denom if denom > 0 else 0.0
                cov_xy[i, j, k] = cross_dev / denom if denom > 0 else 0.0
                
                # Also compute mean of products
                mean_xy[i, j, k] = (mx * my + cov_xy[i, j, k])


@njit(cache=True)
def _covariance_from_moments(
    mean_x: float,
    mean_y: float,
    mean_xy: float
) -> float:
    """
    Covariance identity: Cov(x,y) = <xy> - <x><y>
    """
    return mean_xy - mean_x * mean_y


@njit(cache=True)
def _variance_from_moments(
    mean: float,
    mean_sq: float
) -> float:
    """
    Variance identity: Var(x) = <x²> - <x>²
    """
    return mean_sq - mean * mean


@njit(cache=True)
def _chan_merge_variance(
    n1: int, mean1: float, m2_1: float,
    n2: int, mean2: float, m2_2: float
) -> tuple[int, float, float]:
    """
    Chan's algorithm for merging two variance accumulators.
    Returns (count, mean, m2) for merged result.
    """
    if n1 == 0:
        return n2, mean2, m2_2
    if n2 == 0:
        return n1, mean1, m2_1
    
    n = n1 + n2
    pooled_mean = (n1 * mean1 + n2 * mean2) / n
    
    delta1 = mean1 - pooled_mean
    delta2 = mean2 - pooled_mean
    pooled_m2 = m2_1 + m2_2 + n1 * delta1 * delta1 + n2 * delta2 * delta2
    
    return n, pooled_mean, pooled_m2


@njit(cache=True)
def _pebay_merge_covariance(
    n1: int, mean_x1: float, mean_y1: float, c_1: float,
    n2: int, mean_x2: float, mean_y2: float, c_2: float
) -> tuple[int, float, float, float]:
    """
    Pebay's algorithm for merging two covariance accumulators.
    Returns (count, mean_x, mean_y, cross_dev) for merged result.
    """
    if n1 == 0:
        return n2, mean_x2, mean_y2, c_2
    if n2 == 0:
        return n1, mean_x1, mean_y1, c_1
    
    n = n1 + n2
    pooled_mean_x = (n1 * mean_x1 + n2 * mean_x2) / n
    pooled_mean_y = (n1 * mean_y1 + n2 * mean_y2) / n
    
    delta_x1 = mean_x1 - pooled_mean_x
    delta_y1 = mean_y1 - pooled_mean_y
    delta_x2 = mean_x2 - pooled_mean_x
    delta_y2 = mean_y2 - pooled_mean_y
    
    pooled_c = c_1 + c_2 + n1 * delta_x1 * delta_y1 + n2 * delta_x2 * delta_y2
    
    return n, pooled_mean_x, pooled_mean_y, pooled_c


# Convenience wrappers that dispatch to correct dimension

def blockwise_mean(
    data: np.ndarray,
    window_sizes: tuple[int, ...]
) -> np.ndarray:
    """Dispatch to appropriate Numba kernel based on dimensionality."""
    ndim = data.ndim
    out_shape = tuple(s // w for s, w in zip(data.shape, window_sizes))
    out = np.empty(out_shape, dtype=data.dtype)
    
    if ndim == 2:
        _blockwise_mean_2d(out, data, window_sizes[0], window_sizes[1])
    elif ndim == 3:
        _blockwise_mean_3d(out, data, window_sizes[0], window_sizes[1], window_sizes[2])
    else:
        raise NotImplementedError(f"Only 2D and 3D supported, got {ndim}D")
    
    return out


def blockwise_variance(
    data: np.ndarray,
    window_sizes: tuple[int, ...],
    corrected: bool = True
) -> np.ndarray:
    """Dispatch to appropriate Numba kernel based on dimensionality."""
    ndim = data.ndim
    out_shape = tuple(s // w for s, w in zip(data.shape, window_sizes))
    out = np.empty(out_shape, dtype=data.dtype)
    
    if ndim == 3:
        _blockwise_variance_3d(out, data, window_sizes[0], window_sizes[1], window_sizes[2], corrected)
    else:
        raise NotImplementedError(f"Only 3D supported, got {ndim}D")
    
    return out


def product_mean(
    x: np.ndarray,
    y: np.ndarray,
    window_sizes: tuple[int, ...]
) -> np.ndarray:
    """Compute <x*y> without intermediate allocation."""
    ndim = x.ndim
    out_shape = tuple(s // w for s, w in zip(x.shape, window_sizes))
    out = np.empty(out_shape, dtype=x.dtype)
    
    if ndim == 3:
        _product_mean_3d(out, x, y, window_sizes[0], window_sizes[1], window_sizes[2])
    else:
        raise NotImplementedError(f"Only 3D supported, got {ndim}D")
    
    return out


def product_moments(
    x: np.ndarray,
    y: np.ndarray,
    window_sizes: tuple[int, ...],
    corrected: bool = True
) -> dict[str, np.ndarray]:
    """Compute joint moments (means, variances, covariance) in one pass."""
    ndim = x.ndim
    out_shape = tuple(s // w for s, w in zip(x.shape, window_sizes))
    
    mean_x = np.empty(out_shape, dtype=x.dtype)
    mean_y = np.empty(out_shape, dtype=x.dtype)
    mean_xy = np.empty(out_shape, dtype=x.dtype)
    var_x = np.empty(out_shape, dtype=x.dtype)
    var_y = np.empty(out_shape, dtype=x.dtype)
    cov_xy = np.empty(out_shape, dtype=x.dtype)
    
    if ndim == 3:
        _product_moments_3d(
            mean_x, mean_y, mean_xy, var_x, var_y, cov_xy,
            x, y, window_sizes[0], window_sizes[1], window_sizes[2], corrected
        )
    else:
        raise NotImplementedError(f"Only 3D supported, got {ndim}D")
    
    return {
        'mean_x': mean_x,
        'mean_y': mean_y,
        'mean_xy': mean_xy,
        'var_x': var_x,
        'var_y': var_y,
        'cov_xy': cov_xy
    }
