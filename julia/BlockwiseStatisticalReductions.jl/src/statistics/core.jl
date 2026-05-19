"""
    Core statistics accumulators for mergeable variance, covariance, and moments.

These types support numerically stable parallel merge operations for hierarchical reductions.
"""

"""
    MergeableStatistic{T}

Abstract type for statistics that can be computed incrementally and merged in any order.
Used for tree/hierarchical reductions where sub-blocks are computed independently then combined.
"""
abstract type MergeableStatistic{T} end

"""
    VarianceAccumulator{T}

Accumulator for mean and variance with numerically stable parallel merge.

Uses Welford's online algorithm for incremental updates and Chan's algorithm for parallel merge.
This avoids the bias that occurs when naively averaging variances from sub-blocks.

# Fields
- `count::Int`: Number of samples accumulated
- `mean::T`: Current mean estimate
- `sum_sq_dev::T`: Sum of squared deviations Σ(xᵢ - mean)² (not divided by count)

# Usage
```julia
acc = VarianceAccumulator{Float64}()
for x in data
    fit!(acc, x)
end
mean_result = mean(acc)
var_result = variance(acc)  # Uses Bessel's correction by default
```
"""
mutable struct VarianceAccumulator{T<:AbstractFloat} <: MergeableStatistic{T}
    count::Int
    mean::T
    sum_sq_dev::T
    
    VarianceAccumulator{T}() where {T} = new{T}(0, zero(T), zero(T))
end

VarianceAccumulator() = VarianceAccumulator{Float64}()
VarianceAccumulator(::Type{T}) where {T} = VarianceAccumulator{T}()

"""
    CovarianceAccumulator{T}

Accumulator for covariance between two variables with numerically stable parallel merge.

Uses Pebay's extension of Chan's algorithm for merging covariance accumulators.

# Fields
- `count::Int`: Number of paired samples
- `mean_x::T`: Mean of first variable
- `mean_y::T`: Mean of second variable  
- `sum_cross_dev::T`: Sum of cross-deviations Σ(xᵢ - mean_x)(yᵢ - mean_y)

# Usage
```julia
acc = CovarianceAccumulator{Float64}()
for (x, y) in zip(xs, ys)
    fit!(acc, x, y)
end
cov_result = covariance(acc)
```
"""
mutable struct CovarianceAccumulator{T<:AbstractFloat} <: MergeableStatistic{T}
    count::Int
    mean_x::T
    mean_y::T
    sum_cross_dev::T
    
    CovarianceAccumulator{T}() where {T} = new{T}(0, zero(T), zero(T), zero(T))
end

CovarianceAccumulator() = CovarianceAccumulator{Float64}()
CovarianceAccumulator(::Type{T}) where {T} = CovarianceAccumulator{T}()

"""
    RawMomentsAccumulator{T,N}

Accumulator for raw moments E[x], E[x²], ..., E[x^N] up to order N.

Raw moments merge via simple weighted average (unlike central moments which need Chan/Pebay).
This is used for higher-order statistics (skewness, kurtosis) where exact merge is less critical.

# Fields
- `count::Int`: Number of samples
- `moments::NTuple{N,T}`: Raw moments [E[x], E[x²], ..., E[x^N]]

# Usage
```julia
acc = RawMomentsAccumulator{Float64,4}()  # Up to 4th moment
for x in data
    fit!(acc, x)
end
m1, m2, m3, m4 = acc.moments  # Raw moments
skew = compute_skew(acc)      # Package-defined helper
```
"""
mutable struct RawMomentsAccumulator{T<:AbstractFloat,N} <: MergeableStatistic{T}
    count::Int
    moments::NTuple{N,T}
    
    function RawMomentsAccumulator{T,N}() where {T<:AbstractFloat,N}
        new{T,N}(0, ntuple(_ -> zero(T), N))
    end
end

RawMomentsAccumulator(::Type{T}, N::Int) where {T} = RawMomentsAccumulator{T,N}()
RawMomentsAccumulator(N::Int) = RawMomentsAccumulator{Float64,N}()

#
# Incremental fitting (online updates)
#

"""
    fit!(acc::VarianceAccumulator, x)

Add a single sample to the variance accumulator using Welford's algorithm.
"""
function fit!(acc::VarianceAccumulator{T}, x::Number) where {T}
    x_T = convert(T, x)
    acc.count += 1
    delta = x_T - acc.mean
    acc.mean += delta / acc.count
    delta2 = x_T - acc.mean
    acc.sum_sq_dev += delta * delta2
    return acc
end

"""
    fit!(acc::CovarianceAccumulator, x, y)

Add a paired sample to the covariance accumulator.
"""
function fit!(acc::CovarianceAccumulator{T}, x::Number, y::Number) where {T}
    x_T, y_T = convert(T, x), convert(T, y)
    acc.count += 1
    
    # Update mean_x
    delta_x = x_T - acc.mean_x
    acc.mean_x += delta_x / acc.count
    
    # Update mean_y  
    delta_y = y_T - acc.mean_y
    acc.mean_y += delta_y / acc.count
    
    # Update cross-deviation sum (Pebay formula variant)
    acc.sum_cross_dev += delta_x * (y_T - acc.mean_y)
    
    return acc
end

"""
    fit!(acc::RawMomentsAccumulator{T,N}, x) where {T,N}

Add a sample to the raw moments accumulator.
"""
function fit!(acc::RawMomentsAccumulator{T,N}, x::Number) where {T,N}
    x_T = convert(T, x)
    acc.count += 1
    
    # Update raw moments incrementally
    # New moment_n = (old_moment_n * (n-1) + x^n) / n
    new_moments = ntuple(i -> begin
        power = x_T^i
        (acc.moments[i] * (acc.count - 1) + power) / acc.count
    end, N)
    
    acc.moments = new_moments
    return acc
end

# Batch fitting for efficiency
function fit!(acc::VarianceAccumulator{T}, data::AbstractArray) where {T}
    for x in data
        fit!(acc, x)
    end
    return acc
end

function fit!(acc::CovarianceAccumulator{T}, x::AbstractArray, y::AbstractArray) where {T}
    length(x) == length(y) || throw(DimensionMismatch("x and y must have same length"))
    for i in eachindex(x)
        fit!(acc, x[i], y[i])
    end
    return acc
end

function fit!(acc::RawMomentsAccumulator{T,N}, data::AbstractArray) where {T,N}
    for x in data
        fit!(acc, x)
    end
    return acc
end

#
# Extractor functions
#

"""
    mean(acc::VarianceAccumulator)

Return the accumulated mean.
"""
Statistics.mean(acc::VarianceAccumulator) = acc.mean

"""
    mean(acc::CovarianceAccumulator)

Return the tuple (mean_x, mean_y).
"""
Statistics.mean(acc::CovarianceAccumulator) = (acc.mean_x, acc.mean_y)

"""
    variance(acc::VarianceAccumulator; corrected=true)

Return the variance. Uses Bessel's correction by default (unbiased estimator).
"""
function Statistics.var(acc::VarianceAccumulator; corrected::Bool=true)
    acc.count == 0 && return convert(typeof(acc.mean), NaN)
    acc.count == 1 && return corrected ? convert(typeof(acc.mean), NaN) : zero(typeof(acc.mean))
    
    denom = corrected ? acc.count - 1 : acc.count
    return acc.sum_sq_dev / denom
end

"""
    std(acc::VarianceAccumulator; corrected=true)

Return the standard deviation.
"""
function Statistics.std(acc::VarianceAccumulator; corrected::Bool=true)
    sqrt(var(acc; corrected=corrected))
end

"""
    covariance(acc::CovarianceAccumulator; corrected=true)

Return the covariance between x and y.
"""
function Statistics.cov(acc::CovarianceAccumulator; corrected::Bool=true)
    acc.count == 0 && return convert(typeof(acc.mean_x), NaN)
    acc.count == 1 && return corrected ? convert(typeof(acc.mean_x), NaN) : zero(typeof(acc.mean_x))
    
    denom = corrected ? acc.count - 1 : acc.count
    return acc.sum_cross_dev / denom
end

"""
    cor(acc::CovarianceAccumulator, var_x::VarianceAccumulator, var_y::VarianceAccumulator)

Return the correlation coefficient given covariance and individual variances.
"""
function Statistics.cor(acc::CovarianceAccumulator, var_x::VarianceAccumulator, var_y::VarianceAccumulator)
    cov_xy = covariance(acc; corrected=false)
    var_x_val = variance(var_x; corrected=false)
    var_y_val = variance(var_y; corrected=false)
    
    return cov_xy / sqrt(var_x_val * var_y_val)
end

#
# Count accessor
#

"""
    nobs(acc::MergeableStatistic)

Return the number of observations accumulated.
"""
OnlineStats.nobs(acc::VarianceAccumulator) = acc.count
OnlineStats.nobs(acc::CovarianceAccumulator) = acc.count
OnlineStats.nobs(acc::RawMomentsAccumulator) = acc.count
