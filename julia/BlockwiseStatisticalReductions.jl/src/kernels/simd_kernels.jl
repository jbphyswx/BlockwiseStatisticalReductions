"""
SIMD-vectorized kernels for core reductions.

Uses LoopVectorization.jl for vectorized loops targeting modern CPUs.
Note: LoopVectorization is imported at module level.
"""

"""
    simd_blockwise_mean!(out, data, window_sizes)

SIMD-optimized blockwise mean computation.
"""
function simd_blockwise_mean!(
    out::AbstractArray{T,N},
    data::AbstractArray{T,N},
    window_sizes::NTuple{N,Int}
) where {T,N}
    
    # Calculate output dimensions
    out_dims = size(out)
    
    # Main loop over output blocks
    Base.@inbounds for I in CartesianIndices(out)
        # Compute block start indices
        start_idx = ntuple(i -> (I[i] - 1) * window_sizes[i] + 1, N)
        
        # Accumulate with SIMD
        s = zero(T)
        count = 0
        
        # Inner loops over block elements
        inner_ranges = ntuple(i -> start_idx[i]:start_idx[i]+window_sizes[i]-1, N)
        
        # Manual loop for SIMD
        if N == 3
            i0, j0, k0 = start_idx
            LoopVectorization.@fastmath LoopVectorization.@turbo for kk in 0:window_sizes[3]-1
                for jj in 0:window_sizes[2]-1
                    for ii in 0:window_sizes[1]-1
                        s += data[i0+ii, j0+jj, k0+kk]
                    end
                end
            end
            count = window_sizes[1] * window_sizes[2] * window_sizes[3]
        elseif N == 2
            i0, j0 = start_idx
            LoopVectorization.@fastmath LoopVectorization.@turbo for jj in 0:window_sizes[2]-1
                for ii in 0:window_sizes[1]-1
                    s += data[i0+ii, j0+jj]
                end
            end
            count = window_sizes[1] * window_sizes[2]
        else
            # Generic fallback
            for J in CartesianIndices(inner_ranges)
                s += data[J]
                count += 1
            end
        end
        
        out[I] = s / count
    end
    
    return out
end

"""
    simd_blockwise_variance!(out, data, window_sizes; corrected=true)

SIMD-optimized blockwise variance using Welford's algorithm.
"""
function simd_blockwise_variance!(
    out::AbstractArray{T,N},
    data::AbstractArray{T,N},
    window_sizes::NTuple{N,Int};
    corrected::Bool=true
) where {T,N}
    
    out_dims = size(out)
    block_size = prod(window_sizes)
    
    Base.@inbounds for I in CartesianIndices(out)
        start_idx = ntuple(i -> (I[I] - 1) * window_sizes[i] + 1, N)
        
        # Welford accumulators
        n = 0
        mean = zero(T)
        m2 = zero(T)
        
        # SIMD-friendly accumulation
        if N == 3
            i0, j0, k0 = start_idx
            # Process in chunks for better SIMD utilization
            LoopVectorization.@fastmath for kk in 0:window_sizes[3]-1
                for jj in 0:window_sizes[2]-1
                    # Inner loop vectorized
                    LoopVectorization.@simd for ii in 0:window_sizes[1]-1
                        x = data[i0+ii, j0+jj, k0+kk]
                        n += 1
                        delta = x - mean
                        mean += delta / n
                        delta2 = x - mean
                        m2 += delta * delta2
                    end
                end
            end
        else
            # Generic fallback
            inner_ranges = ntuple(i -> start_idx[i]:start_idx[i]+window_sizes[i]-1, N)
            for J in CartesianIndices(inner_ranges)
                x = data[J]
                n += 1
                delta = x - mean
                mean += delta / n
                delta2 = x - mean
                m2 += delta * delta2
            end
        end
        
        # Compute variance
        denom = corrected ? block_size - 1 : block_size
        out[I] = denom > 0 ? m2 / denom : zero(T)
    end
    
    return out
end

"""
    simd_product_mean!(out, x, y, window_sizes)

SIMD-optimized product mean <x*y> without intermediate allocation.
"""
function simd_product_mean!(
    out::AbstractArray{T,N},
    x::AbstractArray{T,N},
    y::AbstractArray{T,N},
    window_sizes::NTuple{N,Int}
) where {T,N}
    
    out_dims = size(out)
    block_size = prod(window_sizes)
    
    Base.@inbounds for I in CartesianIndices(out)
        start_idx = ntuple(i -> (I[i] - 1) * window_sizes[i] + 1, N)
        
        s = zero(T)
        
        # SIMD vectorized accumulation
        if N == 3
            i0, j0, k0 = start_idx
            LoopVectorization.@fastmath LoopVectorization.@turbo for kk in 0:window_sizes[3]-1
                for jj in 0:window_sizes[2]-1
                    for ii in 0:window_sizes[1]-1
                        s += x[i0+ii, j0+jj, k0+kk] * y[i0+ii, j0+jj, k0+kk]
                    end
                end
            end
        else
            inner_ranges = ntuple(i -> start_idx[i]:start_idx[i]+window_sizes[i]-1, N)
            for J in CartesianIndices(inner_ranges)
                s += x[J] * y[J]
            end
        end
        
        out[I] = s / block_size
    end
    
    return out
end

"""
    simd_product_moments!(out_means..., x, y, window_sizes)

SIMD-optimized joint moments (means, variances, covariance) in one pass.
"""
function simd_product_moments!(
    mean_x::AbstractArray{T,N},
    mean_y::AbstractArray{T,N},
    mean_xy::AbstractArray{T,N},
    var_x::AbstractArray{T,N},
    var_y::AbstractArray{T,N},
    cov_xy::AbstractArray{T,N},
    x::AbstractArray{T,N},
    y::AbstractArray{T,N},
    window_sizes::NTuple{N,Int};
    corrected::Bool=true
) where {T,N}
    
    out_dims = size(mean_x)
    block_size = prod(window_sizes)
    
    Base.@inbounds for I in CartesianIndices(out_dims)
        start_idx = ntuple(i -> (I[i] - 1) * window_sizes[i] + 1, N)
        
        # Welford accumulators
        n = 0
        mx, my = zero(T), zero(T)
        m2_x, m2_y = zero(T), zero(T)
        cross_dev = zero(T)
        
        # SIMD vectorized
        if N == 3
            i0, j0, k0 = start_idx
            LoopVectorization.@fastmath for kk in 0:window_sizes[3]-1
                for jj in 0:window_sizes[2]-1
                    LoopVectorization.@simd for ii in 0:window_sizes[1]-1
                        xv = x[i0+ii, j0+jj, k0+kk]
                        yv = y[i0+ii, j0+jj, k0+kk]
                        
                        n += 1
                        
                        # Online mean updates
                        dx = xv - mx
                        dy = yv - my
                        mx += dx / n
                        my += dy / n
                        
                        # Welford updates
                        m2_x += dx * (xv - mx)
                        m2_y += dy * (yv - my)
                        cross_dev += dx * (yv - my)
                    end
                end
            end
        else
            inner_ranges = ntuple(i -> start_idx[i]:start_idx[i]+window_sizes[i]-1, N)
            for J in CartesianIndices(inner_ranges)
                xv, yv = x[J], y[J]
                n += 1
                dx = xv - mx
                dy = yv - my
                mx += dx / n
                my += dy / n
                m2_x += dx * (xv - mx)
                m2_y += dy * (yv - my)
                cross_dev += dx * (yv - my)
            end
        end
        
        # Store results
        mean_x[I] = mx
        mean_y[I] = my
        
        denom = corrected ? block_size - 1 : block_size
        denom = max(denom, 1)  # Avoid division by zero
        
        var_x[I] = m2_x / denom
        var_y[I] = m2_y / denom
        cov_xy[I] = cross_dev / denom
        mean_xy[I] = mx * my + cov_xy[I]
    end
    
    return nothing
end

# Feature detection - check if LoopVectorization is available
const HAS_SIMD = true  # Package is loaded

"""
    use_simd_kernels()

Check if SIMD kernels should be used (always true if LoopVectorization loaded).
"""
use_simd_kernels() = HAS_SIMD

"""
    best_blockwise_mean(data, window_sizes)

Select best implementation based on array size and hardware.
"""
function best_blockwise_mean(
    data::AbstractArray{T,N},
    window_sizes::NTuple{N,Int}
) where {T,N}
    
    out_shape = ntuple(i -> div(size(data, i), window_sizes[i]), N)
    out = similar(data, out_shape)
    
    # Use SIMD for large arrays
    if prod(size(data)) > 10000 && use_simd_kernels()
        return simd_blockwise_mean!(out, data, window_sizes)
    else
        # Fallback to scalar implementation
        return _scalar_blockwise_mean!(out, data, window_sizes)
    end
end

function _scalar_blockwise_mean!(
    out::AbstractArray{T,N},
    data::AbstractArray{T,N},
    window_sizes::NTuple{N,Int}
) where {T,N}
    
    for I in CartesianIndices(out)
        start_idx = ntuple(i -> (I[i] - 1) * window_sizes[i] + 1, N)
        
        s = zero(T)
        count = 0
        inner_ranges = ntuple(i -> start_idx[i]:start_idx[i]+window_sizes[i]-1, N)
        
        for J in CartesianIndices(inner_ranges)
            s += data[J]
            count += 1
        end
        
        out[I] = s / count
    end
    
    return out
end
