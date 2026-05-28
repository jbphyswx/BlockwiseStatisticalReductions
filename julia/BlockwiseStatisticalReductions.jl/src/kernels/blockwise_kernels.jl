"""
    Canonical blockwise reduction kernels.

All blockwise (non-overlapping tiled) reduction operations in the package
route through these kernels. Each kernel:
- Accepts a pre-allocated output array (zero allocation beyond output)
- Uses @inbounds for bounds-checked-at-call-site performance
- Routes to SIMD (@turbo) for 3D Float32/Float64 arrays above a size threshold
- Falls back to tight scalar loops otherwise
"""

#
# ─── Blockwise Mean ───────────────────────────────────────────────────────────
#

"""
    blockwise_mean!(out, data, window_sizes)

Compute blockwise (non-overlapping) mean of `data` with given window sizes,
writing results into pre-allocated `out`.

`out` must have shape `ntuple(i -> div(size(data,i), window_sizes[i]), N)`.

This is the single canonical implementation used by the entire package.
"""
function blockwise_mean!(
    out::AbstractArray{T,N},
    data::AbstractArray{T,N},
    window_sizes::NTuple{N,Int}
) where {T<:AbstractFloat,N}
    # Route to SIMD for large 3D arrays
    if N == 3 && prod(size(data)) > 10000
        return _blockwise_mean_simd_3d!(out, data, window_sizes)
    else
        return _blockwise_mean_scalar!(out, data, window_sizes)
    end
end

# Also accept non-float (e.g. Int) arrays via scalar path
function blockwise_mean!(
    out::AbstractArray{Tout,N},
    data::AbstractArray{T,N},
    window_sizes::NTuple{N,Int}
) where {Tout,T,N}
    return _blockwise_mean_scalar!(out, data, window_sizes)
end

"""
    blockwise_mean(data, window_sizes) -> Array

Allocating convenience wrapper. Allocates output and calls `blockwise_mean!`.
"""
function blockwise_mean_kernel(data::AbstractArray{T,N}, window_sizes::NTuple{N,Int}) where {T,N}
    out_dims = ntuple(i -> div(size(data, i), window_sizes[i]), N)
    out = similar(data, T, out_dims)
    return blockwise_mean!(out, data, window_sizes)
end

function _blockwise_mean_simd_3d!(
    out::AbstractArray{T,3},
    data::AbstractArray{T,3},
    window_sizes::NTuple{3,Int}
) where {T<:AbstractFloat}
    wx, wy, wz = window_sizes
    inv_count = one(T) / T(wx * wy * wz)

    @inbounds for I in CartesianIndices(out)
        i0 = (I[1] - 1) * wx + 1
        j0 = (I[2] - 1) * wy + 1
        k0 = (I[3] - 1) * wz + 1

        s = zero(T)
        LoopVectorization.@turbo for kk in 0:wz-1
            for jj in 0:wy-1
                for ii in 0:wx-1
                    s += data[i0+ii, j0+jj, k0+kk]
                end
            end
        end
        out[I] = s * inv_count
    end
    return out
end

function _blockwise_mean_scalar!(
    out::AbstractArray{Tout,N},
    data::AbstractArray{T,N},
    window_sizes::NTuple{N,Int}
) where {Tout,T,N}
    inv_count = one(Tout) / Tout(prod(window_sizes))

    @inbounds for I in CartesianIndices(out)
        starts = ntuple(i -> (I[i] - 1) * window_sizes[i] + 1, N)
        s = zero(Tout)
        for J in CartesianIndices(ntuple(i -> starts[i]:starts[i]+window_sizes[i]-1, N))
            s += Tout(data[J])
        end
        out[I] = s * inv_count
    end
    return out
end

#
# ─── Blockwise Variance ──────────────────────────────────────────────────────
#

"""
    blockwise_variance!(out, data, window_sizes; corrected=true)

Compute blockwise variance using Welford's online algorithm (numerically stable).
Single pass, zero allocations beyond `out`.
"""
function blockwise_variance!(
    out::AbstractArray{T,N},
    data::AbstractArray{T,N},
    window_sizes::NTuple{N,Int};
    corrected::Bool=true
) where {T<:AbstractFloat,N}
    block_size = prod(window_sizes)
    denom = corrected ? T(block_size - 1) : T(block_size)

    @inbounds for I in CartesianIndices(out)
        starts = ntuple(i -> (I[i] - 1) * window_sizes[i] + 1, N)

        # Welford's algorithm inline — no mutable struct allocation
        n = 0
        mean = zero(T)
        m2 = zero(T)

        for J in CartesianIndices(ntuple(i -> starts[i]:starts[i]+window_sizes[i]-1, N))
            x = T(data[J])
            n += 1
            delta = x - mean
            mean += delta / n
            delta2 = x - mean
            m2 += delta * delta2
        end

        out[I] = denom > zero(T) ? m2 / denom : zero(T)
    end
    return out
end

"""
    blockwise_variance_kernel(data, window_sizes; corrected=true) -> Array

Allocating convenience wrapper.
"""
function blockwise_variance_kernel(data::AbstractArray{T,N}, window_sizes::NTuple{N,Int};
                                   corrected::Bool=true) where {T,N}
    out_dims = ntuple(i -> div(size(data, i), window_sizes[i]), N)
    out = similar(data, T, out_dims)
    return blockwise_variance!(out, data, window_sizes; corrected=corrected)
end

#
# ─── Blockwise Mean + Variance (single pass) ─────────────────────────────────
#

"""
    blockwise_mean_variance!(out_mean, out_var, data, window_sizes; corrected=true)

Compute both mean and variance in a single pass over the data.
Uses Welford's algorithm. Zero allocations beyond outputs.
"""
function blockwise_mean_variance!(
    out_mean::AbstractArray{T,N},
    out_var::AbstractArray{T,N},
    data::AbstractArray{T,N},
    window_sizes::NTuple{N,Int};
    corrected::Bool=true
) where {T<:AbstractFloat,N}
    block_size = prod(window_sizes)
    denom = corrected ? T(block_size - 1) : T(block_size)

    @inbounds for I in CartesianIndices(out_mean)
        starts = ntuple(i -> (I[i] - 1) * window_sizes[i] + 1, N)

        n = 0
        mean = zero(T)
        m2 = zero(T)

        for J in CartesianIndices(ntuple(i -> starts[i]:starts[i]+window_sizes[i]-1, N))
            x = T(data[J])
            n += 1
            delta = x - mean
            mean += delta / n
            delta2 = x - mean
            m2 += delta * delta2
        end

        out_mean[I] = mean
        out_var[I] = denom > zero(T) ? m2 / denom : zero(T)
    end
    return nothing
end

#
# ─── Blockwise Min / Max ─────────────────────────────────────────────────────
#

"""
    blockwise_min!(out, data, window_sizes)

Compute blockwise minimum. Zero allocations beyond `out`.
"""
function blockwise_min!(
    out::AbstractArray{T,N},
    data::AbstractArray{T,N},
    window_sizes::NTuple{N,Int}
) where {T,N}
    @inbounds for I in CartesianIndices(out)
        starts = ntuple(i -> (I[i] - 1) * window_sizes[i] + 1, N)
        m = typemax(T)
        for J in CartesianIndices(ntuple(i -> starts[i]:starts[i]+window_sizes[i]-1, N))
            v = data[J]
            m = ifelse(v < m, v, m)
        end
        out[I] = m
    end
    return out
end

"""
    blockwise_max!(out, data, window_sizes)

Compute blockwise maximum. Zero allocations beyond `out`.
"""
function blockwise_max!(
    out::AbstractArray{T,N},
    data::AbstractArray{T,N},
    window_sizes::NTuple{N,Int}
) where {T,N}
    @inbounds for I in CartesianIndices(out)
        starts = ntuple(i -> (I[i] - 1) * window_sizes[i] + 1, N)
        m = typemin(T)
        for J in CartesianIndices(ntuple(i -> starts[i]:starts[i]+window_sizes[i]-1, N))
            v = data[J]
            m = ifelse(v > m, v, m)
        end
        out[I] = m
    end
    return out
end

#
# ─── Blockwise Product Mean (fused x*y without intermediate) ─────────────────
#

"""
    blockwise_product_mean!(out, x, y, window_sizes)

Compute `<x*y>` per block without materializing the product array.
Zero allocations beyond `out`.
"""
function blockwise_product_mean!(
    out::AbstractArray{T,N},
    x::AbstractArray{T,N},
    y::AbstractArray{T,N},
    window_sizes::NTuple{N,Int}
) where {T<:AbstractFloat,N}
    inv_count = one(T) / T(prod(window_sizes))

    if N == 3 && prod(size(x)) > 10000
        return _blockwise_product_mean_simd_3d!(out, x, y, window_sizes, inv_count)
    end

    @inbounds for I in CartesianIndices(out)
        starts = ntuple(i -> (I[i] - 1) * window_sizes[i] + 1, N)
        s = zero(T)
        for J in CartesianIndices(ntuple(i -> starts[i]:starts[i]+window_sizes[i]-1, N))
            s += x[J] * y[J]
        end
        out[I] = s * inv_count
    end
    return out
end

function _blockwise_product_mean_simd_3d!(
    out::AbstractArray{T,3},
    x::AbstractArray{T,3},
    y::AbstractArray{T,3},
    window_sizes::NTuple{3,Int},
    inv_count::T
) where {T<:AbstractFloat}
    wx, wy, wz = window_sizes

    @inbounds for I in CartesianIndices(out)
        i0 = (I[1] - 1) * wx + 1
        j0 = (I[2] - 1) * wy + 1
        k0 = (I[3] - 1) * wz + 1

        s = zero(T)
        LoopVectorization.@turbo for kk in 0:wz-1
            for jj in 0:wy-1
                for ii in 0:wx-1
                    s += x[i0+ii, j0+jj, k0+kk] * y[i0+ii, j0+jj, k0+kk]
                end
            end
        end
        out[I] = s * inv_count
    end
    return out
end

#
# ─── Blockwise Joint Moments (mean_x, mean_y, var_x, var_y, cov_xy) ─────────
#

"""
    blockwise_joint_moments!(mean_x, mean_y, var_x, var_y, cov_xy, x, y, window_sizes; corrected=true)

Compute all joint first/second moments in a single fused pass.
Uses Welford/Pebay online algorithm. Zero allocations beyond output arrays.
"""
function blockwise_joint_moments!(
    mean_x::AbstractArray{T,N},
    mean_y::AbstractArray{T,N},
    var_x::AbstractArray{T,N},
    var_y::AbstractArray{T,N},
    cov_xy::AbstractArray{T,N},
    x::AbstractArray{T,N},
    y::AbstractArray{T,N},
    window_sizes::NTuple{N,Int};
    corrected::Bool=true
) where {T<:AbstractFloat,N}
    block_size = prod(window_sizes)
    denom = corrected ? T(block_size - 1) : T(block_size)
    denom = max(denom, one(T))

    @inbounds for I in CartesianIndices(mean_x)
        starts = ntuple(i -> (I[i] - 1) * window_sizes[i] + 1, N)

        n = 0
        mx, my = zero(T), zero(T)
        m2_x, m2_y = zero(T), zero(T)
        cross = zero(T)

        for J in CartesianIndices(ntuple(i -> starts[i]:starts[i]+window_sizes[i]-1, N))
            xv = T(x[J])
            yv = T(y[J])
            n += 1

            dx = xv - mx
            dy = yv - my
            mx += dx / n
            my += dy / n

            # Welford updates (after mean update)
            m2_x += dx * (xv - mx)
            m2_y += dy * (yv - my)
            cross += dx * (yv - my)
        end

        mean_x[I] = mx
        mean_y[I] = my
        var_x[I] = m2_x / denom
        var_y[I] = m2_y / denom
        cov_xy[I] = cross / denom
    end
    return nothing
end
