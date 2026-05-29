"""
    Blockwise merge kernels for hierarchical sufficient-statistics composition.

These kernels take arrays of per-block sufficient statistics (mean, M2, C, M3, etc.)
and merge them spatially by a blockwise factor, producing coarser-resolution arrays
of sufficient statistics that are numerically exact (not approximations).

This enables towers of any composable statistic:
- mean: weighted average (trivial)
- M2 (variance numerator): Chan's parallel merge
- C (covariance numerator): Pebay's parallel merge
- M3 (skewness numerator): extended Chan's formula
- count/sum: additive
- raw moments E[x^k]: weighted average

All kernels are in-place, zero-allocation for pre-allocated outputs.
"""

#
# ─── Count/Sum merge (additive) ──────────────────────────────────────────────
#

"""
    blockwise_sum!(out, data, window_sizes)

Blockwise sum: sum all elements in each block. Composable (sum of sums = sum).
"""
function blockwise_sum!(
    out::AbstractArray{T,N},
    data::AbstractArray{T,N},
    window_sizes::NTuple{N,Int}
) where {T,N}
    @inbounds for I in CartesianIndices(out)
        starts = ntuple(i -> (I[i] - 1) * window_sizes[i] + 1, N)
        s = zero(T)
        for J in CartesianIndices(ntuple(i -> starts[i]:starts[i]+window_sizes[i]-1, N))
            s += data[J]
        end
        out[I] = s
    end
    return out
end

#
# ─── Mean + M2 merge (Chan's algorithm, blockwise) ──────────────────────────
#

"""
    blockwise_merge_mean_M2!(out_mean, out_M2, mean_arr, M2_arr, count_per_block, window_sizes)

Chan's parallel variance merge applied blockwise.

Given arrays of per-block (mean, M2) where each block has `count_per_block` samples,
merge them spatially by `window_sizes` to produce coarser (mean, M2).

The merged M2 is the **exact** sum of squared deviations for the combined block —
NOT an approximation. This enables numerically correct variance towers.

# Formula (for merging k children each with count n):
```
N_total = k * n
μ_merged = (1/k) Σ μᵢ                    (equal-count weighted mean)
M2_merged = Σ M2ᵢ + n * Σ (μᵢ - μ_merged)²
```
"""
function blockwise_merge_mean_M2!(
    out_mean::AbstractArray{T,N},
    out_M2::AbstractArray{T,N},
    mean_arr::AbstractArray{T,N},
    M2_arr::AbstractArray{T,N},
    count_per_block::Int,
    window_sizes::NTuple{N,Int}
) where {T<:AbstractFloat,N}
    inv_k = one(T) / T(prod(window_sizes))

    @inbounds for I in CartesianIndices(out_mean)
        starts = ntuple(i -> (I[i] - 1) * window_sizes[i] + 1, N)

        # First pass: compute merged mean
        sum_mean = zero(T)
        for J in CartesianIndices(ntuple(i -> starts[i]:starts[i]+window_sizes[i]-1, N))
            sum_mean += mean_arr[J]
        end
        μ_merged = sum_mean * inv_k

        # Second pass: compute merged M2
        sum_M2 = zero(T)
        sum_dev_sq = zero(T)
        for J in CartesianIndices(ntuple(i -> starts[i]:starts[i]+window_sizes[i]-1, N))
            sum_M2 += M2_arr[J]
            δ = mean_arr[J] - μ_merged
            sum_dev_sq += δ * δ
        end
        # M2_merged = Σ M2_i + n * Σ (μ_i - μ_merged)²
        out_mean[I] = μ_merged
        out_M2[I] = sum_M2 + T(count_per_block) * sum_dev_sq
    end
    return out_mean, out_M2
end

#
# ─── Mean + M2 + M3 merge (extended Chan's for skewness) ────────────────────
#

"""
    blockwise_merge_mean_M2_M3!(out_mean, out_M2, out_M3, mean_arr, M2_arr, M3_arr, count_per_block, window_sizes)

Extended Chan's merge for third central moment (skewness numerator).

Merges (mean, M2, M3) arrays blockwise. The formula for merging two groups
generalizes to k equal-count children:

For each pair (p merging into accumulated):
```
δ = μ_p - μ_acc
M3_new = M3_acc + M3_p + δ³ * n_acc * n_p * (n_acc - n_p) / N²
         + 3δ * (n_acc * M2_p - n_p * M2_acc) / N
```

This kernel uses iterative pairwise merge over the block.
"""
function blockwise_merge_mean_M2_M3!(
    out_mean::AbstractArray{T,N},
    out_M2::AbstractArray{T,N},
    out_M3::AbstractArray{T,N},
    mean_arr::AbstractArray{T,N},
    M2_arr::AbstractArray{T,N},
    M3_arr::AbstractArray{T,N},
    count_per_block::Int,
    window_sizes::NTuple{N,Int}
) where {T<:AbstractFloat,N}
    n = T(count_per_block)

    @inbounds for I in CartesianIndices(out_mean)
        starts = ntuple(i -> (I[i] - 1) * window_sizes[i] + 1, N)
        block_range = CartesianIndices(ntuple(i -> starts[i]:starts[i]+window_sizes[i]-1, N))

        # Iterative merge: start with first element, merge each subsequent
        first_J = first(block_range)
        acc_n = n
        acc_μ = mean_arr[first_J]
        acc_M2 = M2_arr[first_J]
        acc_M3 = M3_arr[first_J]

        for J in Iterators.drop(block_range, 1)
            n_p = n
            μ_p = mean_arr[J]
            M2_p = M2_arr[J]
            M3_p = M3_arr[J]

            N_new = acc_n + n_p
            δ = μ_p - acc_μ
            δ_over_N = δ / N_new
            δ2 = δ * δ

            # Update M3 first (depends on old acc_μ, acc_M2)
            acc_M3 = acc_M3 + M3_p +
                     δ2 * δ_over_N * acc_n * n_p * (acc_n - n_p) / N_new +
                     T(3) * δ_over_N * (acc_n * M2_p - n_p * acc_M2)

            # Update M2
            acc_M2 = acc_M2 + M2_p + δ2 * (acc_n * n_p / N_new)

            # Update mean
            acc_μ = (acc_n * acc_μ + n_p * μ_p) / N_new

            acc_n = N_new
        end

        out_mean[I] = acc_μ
        out_M2[I] = acc_M2
        out_M3[I] = acc_M3
    end
    return out_mean, out_M2, out_M3
end

#
# ─── Mean_x + Mean_y + C merge (Pebay covariance) ───────────────────────────
#

"""
    blockwise_merge_covariance!(out_mx, out_my, out_C, mx_arr, my_arr, C_arr, count_per_block, window_sizes)

Pebay's parallel covariance merge applied blockwise.

Given arrays of per-block (mean_x, mean_y, C) where C = Σ(xᵢ - μx)(yᵢ - μy),
merge them spatially to produce coarser (mean_x, mean_y, C).

# Formula (for merging k equal-count children):
```
μx_merged = (1/k) Σ μx_i
μy_merged = (1/k) Σ μy_i
C_merged = Σ Cᵢ + n * Σ (μx_i - μx_merged)(μy_i - μy_merged)
```
"""
function blockwise_merge_covariance!(
    out_mx::AbstractArray{T,N},
    out_my::AbstractArray{T,N},
    out_C::AbstractArray{T,N},
    mx_arr::AbstractArray{T,N},
    my_arr::AbstractArray{T,N},
    C_arr::AbstractArray{T,N},
    count_per_block::Int,
    window_sizes::NTuple{N,Int}
) where {T<:AbstractFloat,N}
    inv_k = one(T) / T(prod(window_sizes))

    @inbounds for I in CartesianIndices(out_mx)
        starts = ntuple(i -> (I[i] - 1) * window_sizes[i] + 1, N)
        block_range = CartesianIndices(ntuple(i -> starts[i]:starts[i]+window_sizes[i]-1, N))

        # First pass: merged means
        sum_mx = zero(T)
        sum_my = zero(T)
        for J in block_range
            sum_mx += mx_arr[J]
            sum_my += my_arr[J]
        end
        μx = sum_mx * inv_k
        μy = sum_my * inv_k

        # Second pass: merged C
        sum_C = zero(T)
        sum_cross = zero(T)
        for J in block_range
            sum_C += C_arr[J]
            sum_cross += (mx_arr[J] - μx) * (my_arr[J] - μy)
        end

        out_mx[I] = μx
        out_my[I] = μy
        out_C[I] = sum_C + T(count_per_block) * sum_cross
    end
    return out_mx, out_my, out_C
end

#
# ─── Raw moments merge (weighted average, any order) ─────────────────────────
#

"""
    blockwise_merge_raw_moments!(out_moments, moment_arrs, window_sizes)

Merge raw moments E[x^k] blockwise. Since raw moments are expectations,
they merge via simple (equal-weight) averaging across children.

`moment_arrs` is a tuple/vector of arrays, one per moment order.
`out_moments` is a tuple/vector of output arrays.

This handles arbitrary order: mean (k=1), E[x²] (k=2), E[x³] (k=3), etc.
"""
function blockwise_merge_raw_moments!(
    out_moments::NTuple{K, <:AbstractArray{<:AbstractFloat,N}},
    moment_arrs::NTuple{K, <:AbstractArray{<:AbstractFloat,N}},
    window_sizes::NTuple{N,Int}
) where {K, N}
    T = eltype(out_moments[1])
    inv_k = one(T) / T(prod(window_sizes))

    @inbounds for I in CartesianIndices(out_moments[1])
        starts = ntuple(i -> (I[i] - 1) * window_sizes[i] + 1, N)
        block_range = CartesianIndices(ntuple(i -> starts[i]:starts[i]+window_sizes[i]-1, N))

        # Sum each moment order
        sums = ntuple(_ -> zero(T), K)
        for J in block_range
            sums = ntuple(k -> sums[k] + moment_arrs[k][J], K)
        end

        # Average
        for k in 1:K
            out_moments[k][I] = sums[k] * inv_k
        end
    end
    return out_moments
end

#
# ─── Convenience: initial blockwise sufficient statistics from raw data ──────
#

"""
    blockwise_mean_M2!(out_mean, out_M2, data, window_sizes)

Compute per-block mean and M2 (sum of squared deviations) from raw data.
This produces the sufficient statistics needed for variance tower merging.

Returns `(out_mean, out_M2)` where `M2[i] = Σ(x - mean)²` within block i.
"""
function blockwise_mean_M2!(
    out_mean::AbstractArray{T,N},
    out_M2::AbstractArray{T,N},
    data::AbstractArray{T,N},
    window_sizes::NTuple{N,Int}
) where {T<:AbstractFloat,N}
    count = T(prod(window_sizes))
    inv_count = one(T) / count

    @inbounds for I in CartesianIndices(out_mean)
        starts = ntuple(i -> (I[i] - 1) * window_sizes[i] + 1, N)
        block_range = CartesianIndices(ntuple(i -> starts[i]:starts[i]+window_sizes[i]-1, N))

        # Pass 1: mean
        s = zero(T)
        for J in block_range
            s += data[J]
        end
        μ = s * inv_count

        # Pass 2: M2
        m2 = zero(T)
        for J in block_range
            δ = data[J] - μ
            m2 += δ * δ
        end

        out_mean[I] = μ
        out_M2[I] = m2
    end
    return out_mean, out_M2
end

"""
    blockwise_mean_M2_M3!(out_mean, out_M2, out_M3, data, window_sizes)

Compute per-block mean, M2, and M3 (third central moment sum) from raw data.
M3 = Σ(x - mean)³ within each block.
"""
function blockwise_mean_M2_M3!(
    out_mean::AbstractArray{T,N},
    out_M2::AbstractArray{T,N},
    out_M3::AbstractArray{T,N},
    data::AbstractArray{T,N},
    window_sizes::NTuple{N,Int}
) where {T<:AbstractFloat,N}
    count = T(prod(window_sizes))
    inv_count = one(T) / count

    @inbounds for I in CartesianIndices(out_mean)
        starts = ntuple(i -> (I[i] - 1) * window_sizes[i] + 1, N)
        block_range = CartesianIndices(ntuple(i -> starts[i]:starts[i]+window_sizes[i]-1, N))

        # Pass 1: mean
        s = zero(T)
        for J in block_range
            s += data[J]
        end
        μ = s * inv_count

        # Pass 2: M2, M3
        m2 = zero(T)
        m3 = zero(T)
        for J in block_range
            δ = data[J] - μ
            m2 += δ * δ
            m3 += δ * δ * δ
        end

        out_mean[I] = μ
        out_M2[I] = m2
        out_M3[I] = m3
    end
    return out_mean, out_M2, out_M3
end

"""
    blockwise_mean_C!(out_mx, out_my, out_C, x, y, window_sizes)

Compute per-block mean_x, mean_y, and C (sum of cross-deviations) from paired data.
C = Σ(xᵢ - μx)(yᵢ - μy) within each block.
"""
function blockwise_mean_C!(
    out_mx::AbstractArray{T,N},
    out_my::AbstractArray{T,N},
    out_C::AbstractArray{T,N},
    x::AbstractArray{T,N},
    y::AbstractArray{T,N},
    window_sizes::NTuple{N,Int}
) where {T<:AbstractFloat,N}
    count = T(prod(window_sizes))
    inv_count = one(T) / count

    @inbounds for I in CartesianIndices(out_mx)
        starts = ntuple(i -> (I[i] - 1) * window_sizes[i] + 1, N)
        block_range = CartesianIndices(ntuple(i -> starts[i]:starts[i]+window_sizes[i]-1, N))

        # Pass 1: means
        sx = zero(T)
        sy = zero(T)
        for J in block_range
            sx += x[J]
            sy += y[J]
        end
        μx = sx * inv_count
        μy = sy * inv_count

        # Pass 2: C
        c = zero(T)
        for J in block_range
            c += (x[J] - μx) * (y[J] - μy)
        end

        out_mx[I] = μx
        out_my[I] = μy
        out_C[I] = c
    end
    return out_mx, out_my, out_C
end

#
# ─── Derived quantities from sufficient statistics ───────────────────────────
#

"""
    variance_from_M2(M2, count; corrected=true)

Convert M2 (sum of squared deviations) to variance.
"""
@inline function variance_from_M2(M2::T, count::Int; corrected::Bool=true) where T
    denom = corrected ? T(count - 1) : T(count)
    return M2 / denom
end

"""
    std_from_M2(M2, count; corrected=true)

Convert M2 to standard deviation.
"""
@inline function std_from_M2(M2::T, count::Int; corrected::Bool=true) where T
    return sqrt(variance_from_M2(M2, count; corrected=corrected))
end

"""
    skewness_from_M2_M3(M2, M3, count)

Compute skewness from M2 and M3.
skewness = (M3/n) / (M2/n)^(3/2) = M3 * √n / M2^(3/2)
"""
@inline function skewness_from_M2_M3(M2::T, M3::T, count::Int) where T
    n = T(count)
    return (M3 / n) / (M2 / n)^(T(3)/T(2))
end

"""
    covariance_from_C(C, count; corrected=true)

Convert C (sum of cross-deviations) to covariance.
"""
@inline function covariance_from_C(C::T, count::Int; corrected::Bool=true) where T
    denom = corrected ? T(count - 1) : T(count)
    return C / denom
end
