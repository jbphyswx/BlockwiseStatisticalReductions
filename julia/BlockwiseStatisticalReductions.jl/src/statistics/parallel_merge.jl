"""
    Parallel merge algorithms for numerically stable combination of statistics.

These algorithms allow independent computation of statistics on sub-blocks,
then correct merging of results without bias. Essential for hierarchical/tree reductions.

References:
- Chan et al. (1979): Parallel variance computation
- Pebay et al. (2008): Extension to covariance
"""

#
# Variance merge (Chan's algorithm)
#

"""
    merge(acc1::VarianceAccumulator, acc2::VarianceAccumulator)

Merge two variance accumulators using Chan's parallel algorithm.

This computes the pooled variance of two disjoint datasets without bias,
unlike naively averaging the two variances which would be incorrect.

# Algorithm
```
n = n1 + n2
δ = μ2 - μ1
μ = (n1*μ1 + n2*μ2) / n
M2 = M2_1 + M2_2 + δ² * (n1*n2 / n)
```

where M2 is the sum of squared deviations (not divided by n).

# Example
```julia
acc1 = VarianceAccumulator{Float64}()
fit!(acc1, [1.0, 2.0, 3.0])

acc2 = VarianceAccumulator{Float64}()
fit!(acc2, [4.0, 5.0, 6.0])

# Merge gives same result as computing on all 6 values together
merged = merge(acc1, acc2)
```
"""
function Base.merge(acc1::VarianceAccumulator{T}, acc2::VarianceAccumulator{T}) where {T}
    n1, n2 = acc1.count, acc2.count
    n = n1 + n2
    
    # Handle edge cases
    if n == 0
        return VarianceAccumulator{T}()
    elseif n2 == 0
        result = VarianceAccumulator{T}()
        result.count = n1
        result.mean = acc1.mean
        result.sum_sq_dev = acc1.sum_sq_dev
        return result
    elseif n1 == 0
        result = VarianceAccumulator{T}()
        result.count = n2
        result.mean = acc2.mean
        result.sum_sq_dev = acc2.sum_sq_dev
        return result
    end
    
    μ1, μ2 = acc1.mean, acc2.mean
    M2_1, M2_2 = acc1.sum_sq_dev, acc2.sum_sq_dev
    
    # Chan's parallel variance formula
    δ = μ2 - μ1
    μ = (n1 * μ1 + n2 * μ2) / n
    M2 = M2_1 + M2_2 + δ^2 * (n1 * n2 / n)
    
    result = VarianceAccumulator{T}()
    result.count = n
    result.mean = μ
    result.sum_sq_dev = M2
    return result
end

# Allow merging different precisions by promoting
function Base.merge(acc1::VarianceAccumulator{T1}, acc2::VarianceAccumulator{T2}) where {T1,T2}
    T = promote_type(T1, T2)
    
    # Create promoted copies
    acc1_promoted = VarianceAccumulator{T}()
    acc1_promoted.count = acc1.count
    acc1_promoted.mean = convert(T, acc1.mean)
    acc1_promoted.sum_sq_dev = convert(T, acc1.sum_sq_dev)
    
    acc2_promoted = VarianceAccumulator{T}()
    acc2_promoted.count = acc2.count
    acc2_promoted.mean = convert(T, acc2.mean)
    acc2_promoted.sum_sq_dev = convert(T, acc2.sum_sq_dev)
    
    return merge(acc1_promoted, acc2_promoted)
end

"""
    merge_many(counts::AbstractVector{Int}, means::AbstractVector{T}, sum_sq_devs::AbstractVector{T}) where T

Merge multiple variance accumulators at once using Chan's algorithm iteratively.

This is more efficient than pairwise merging for many sub-blocks.
"""
function merge_many(counts::AbstractVector{Int}, means::AbstractVector{T}, sum_sq_devs::AbstractVector{T}) where {T}
    length(counts) == length(means) == length(sum_sq_devs) || 
        throw(DimensionMismatch("counts, means, and sum_sq_devs must have same length"))
    
    n = sum(counts)
    n == 0 && return VarianceAccumulator{T}()
    
    # Compute pooled mean
    pooled_mean = sum(c * m for (c, m) in zip(counts, means)) / n
    
    # Compute pooled M2 (sum of squared deviations)
    # M2_total = Σ M2_i + Σ n_i * (μ_i - μ_pooled)²
    M2_sum = sum(sum_sq_devs)
    mean_dev_sum = sum(c * (m - pooled_mean)^2 for (c, m) in zip(counts, means))
    
    pooled_M2 = M2_sum + mean_dev_sum
    
    result = VarianceAccumulator{T}()
    result.count = n
    result.mean = pooled_mean
    result.sum_sq_dev = pooled_M2
    return result
end

#
# Covariance merge (Pebay's extension of Chan's algorithm)
#

"""
    merge(acc1::CovarianceAccumulator, acc2::CovarianceAccumulator)

Merge two covariance accumulators using Pebay's parallel algorithm.

This is the extension of Chan's variance merge to the covariance case.

# Algorithm
```
n = n1 + n2
δx = μx2 - μx1
δy = μy2 - μy1
μx = (n1*μx1 + n2*μx2) / n
μy = (n1*μy1 + n2*μy2) / n
C = C1 + C2 + (n1*n2 / n) * δx * δy
```

where C is the sum of cross-deviations.

# Example
```julia
acc1 = CovarianceAccumulator{Float64}()
fit!(acc1, [1.0, 2.0, 3.0], [2.0, 4.0, 6.0])

acc2 = CovarianceAccumulator{Float64}()
fit!(acc2, [4.0, 5.0, 6.0], [8.0, 10.0, 12.0])

merged = merge(acc1, acc2)
```
"""
function Base.merge(acc1::CovarianceAccumulator{T}, acc2::CovarianceAccumulator{T}) where {T}
    n1, n2 = acc1.count, acc2.count
    n = n1 + n2
    
    # Handle edge cases
    if n == 0
        return CovarianceAccumulator{T}()
    elseif n2 == 0
        result = CovarianceAccumulator{T}()
        result.count = n1
        result.mean_x = acc1.mean_x
        result.mean_y = acc1.mean_y
        result.sum_cross_dev = acc1.sum_cross_dev
        return result
    elseif n1 == 0
        result = CovarianceAccumulator{T}()
        result.count = n2
        result.mean_x = acc2.mean_x
        result.mean_y = acc2.mean_y
        result.sum_cross_dev = acc2.sum_cross_dev
        return result
    end
    
    μx1, μx2 = acc1.mean_x, acc2.mean_x
    μy1, μy2 = acc1.mean_y, acc2.mean_y
    C1, C2 = acc1.sum_cross_dev, acc2.sum_cross_dev
    
    # Pebay's parallel covariance formula
    δx = μx2 - μx1
    δy = μy2 - μy1
    μx = (n1 * μx1 + n2 * μx2) / n
    μy = (n1 * μy1 + n2 * μy2) / n
    C = C1 + C2 + (n1 * n2 / n) * δx * δy
    
    result = CovarianceAccumulator{T}()
    result.count = n
    result.mean_x = μx
    result.mean_y = μy
    result.sum_cross_dev = C
    return result
end

# Allow merging different precisions by promoting
function Base.merge(acc1::CovarianceAccumulator{T1}, acc2::CovarianceAccumulator{T2}) where {T1,T2}
    T = promote_type(T1, T2)
    
    acc1_promoted = CovarianceAccumulator{T}()
    acc1_promoted.count = acc1.count
    acc1_promoted.mean_x = convert(T, acc1.mean_x)
    acc1_promoted.mean_y = convert(T, acc1.mean_y)
    acc1_promoted.sum_cross_dev = convert(T, acc1.sum_cross_dev)
    
    acc2_promoted = CovarianceAccumulator{T}()
    acc2_promoted.count = acc2.count
    acc2_promoted.mean_x = convert(T, acc2.mean_x)
    acc2_promoted.mean_y = convert(T, acc2.mean_y)
    acc2_promoted.sum_cross_dev = convert(T, acc2.sum_cross_dev)
    
    return merge(acc1_promoted, acc2_promoted)
end

"""
    merge_many(counts::AbstractVector{Int}, means_x::AbstractVector{T}, means_y::AbstractVector{T}, 
               sum_cross_devs::AbstractVector{T}) where T

Merge multiple covariance accumulators at once.

Uses the generalization of Pebay's formula to many groups.
"""
function merge_many(counts::AbstractVector{Int}, means_x::AbstractVector{T}, means_y::AbstractVector{T}, 
                    sum_cross_devs::AbstractVector{T}) where {T}
    length(counts) == length(means_x) == length(means_y) == length(sum_cross_devs) || 
        throw(DimensionMismatch("All input vectors must have same length"))
    
    n = sum(counts)
    n == 0 && return CovarianceAccumulator{T}()
    
    # Compute pooled means
    pooled_mean_x = sum(c * mx for (c, mx) in zip(counts, means_x)) / n
    pooled_mean_y = sum(c * my for (c, my) in zip(counts, means_y)) / n
    
    # Compute pooled cross-deviation sum
    # C_total = Σ C_i + Σ n_i * (μx_i - μx_pooled)(μy_i - μy_pooled)
    C_sum = sum(sum_cross_devs)
    cross_dev_sum = sum(c * (mx - pooled_mean_x) * (my - pooled_mean_y) 
                        for (c, mx, my) in zip(counts, means_x, means_y))
    
    pooled_C = C_sum + cross_dev_sum
    
    result = CovarianceAccumulator{T}()
    result.count = n
    result.mean_x = pooled_mean_x
    result.mean_y = pooled_mean_y
    result.sum_cross_dev = pooled_C
    return result
end

#
# Raw moments merge (simple weighted average)
#

"""
    merge(acc1::RawMomentsAccumulator{T,N}, acc2::RawMomentsAccumulator{T,N}) where {T,N}

Merge two raw moments accumulators.

Raw moments are computed as E[x^k] = Σ x^k / n, so they merge via weighted average:
E_total[x^k] = (n1*E1[x^k] + n2*E2[x^k]) / (n1 + n2)

This is simpler than variance/covariance because raw moments don't depend on the mean.
"""
function Base.merge(acc1::RawMomentsAccumulator{T,N}, acc2::RawMomentsAccumulator{T,N}) where {T,N}
    n1, n2 = acc1.count, acc2.count
    n = n1 + n2
    
    # Handle edge cases
    if n == 0
        return RawMomentsAccumulator{T,N}()
    elseif n2 == 0
        result = RawMomentsAccumulator{T,N}()
        result.count = n1
        result.moments = acc1.moments
        return result
    elseif n1 == 0
        result = RawMomentsAccumulator{T,N}()
        result.count = n2
        result.moments = acc2.moments
        return result
    end
    
    # Weighted average of each moment
    new_moments = ntuple(i -> (n1 * acc1.moments[i] + n2 * acc2.moments[i]) / n, N)
    
    result = RawMomentsAccumulator{T,N}()
    result.count = n
    result.moments = new_moments
    return result
end

# Allow merging different precisions
function Base.merge(acc1::RawMomentsAccumulator{T1,N}, acc2::RawMomentsAccumulator{T2,N}) where {T1,T2,N}
    T = promote_type(T1, T2)
    
    acc1_promoted = RawMomentsAccumulator{T,N}()
    acc1_promoted.count = acc1.count
    acc1_promoted.moments = ntuple(i -> convert(T, acc1.moments[i]), N)
    
    acc2_promoted = RawMomentsAccumulator{T,N}()
    acc2_promoted.count = acc2.count
    acc2_promoted.moments = ntuple(i -> convert(T, acc2.moments[i]), N)
    
    return merge(acc1_promoted, acc2_promoted)
end

#
# Higher-level merge utilities
#

"""
    merge_all(accs::Vector{<:VarianceAccumulator})

Merge a vector of variance accumulators.
"""
function merge_all(accs::AbstractVector{VarianceAccumulator{T}}) where {T}
    isempty(accs) && return VarianceAccumulator{T}()
    
    result = accs[1]
    for i in 2:length(accs)
        result = merge(result, accs[i])
    end
    return result
end

"""
    merge_all(accs::Vector{<:CovarianceAccumulator})

Merge a vector of covariance accumulators.
"""
function merge_all(accs::AbstractVector{CovarianceAccumulator{T}}) where {T}
    isempty(accs) && return CovarianceAccumulator{T}()
    
    result = accs[1]
    for i in 2:length(accs)
        result = merge(result, accs[i])
    end
    return result
end

"""
    merge_all(accs::Vector{<:RawMomentsAccumulator})

Merge a vector of raw moments accumulators.
"""
function merge_all(accs::AbstractVector{RawMomentsAccumulator{T,N}}) where {T,N}
    isempty(accs) && return RawMomentsAccumulator{T,N}()
    
    result = accs[1]
    for i in 2:length(accs)
        result = merge(result, accs[i])
    end
    return result
end
