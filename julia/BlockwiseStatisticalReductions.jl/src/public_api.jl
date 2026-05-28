"""
    Public API - high-level convenience functions for common use cases.

These functions provide a clean, user-friendly interface to the package's
core functionality without exposing internal implementation details.
"""

#
# Blockwise statistics API
#

"""
    blockwise_stats(data, window_sizes; stats=[:mean], corrected=true)

Compute multiple statistics on data using blockwise (non-overlapping) windows.

# Arguments
- `data`: Input array (any dimension)
- `window_sizes::NTuple{N,Int}`: Block size for each dimension
- `stats::Vector{Symbol}`: Statistics to compute [:mean, :variance, :std, :min, :max]
- `corrected::Bool`: Use Bessel's correction for variance (default true)

# Returns
Dict mapping stat name => computed array

# Example
```julia
# Compute mean and variance in 10x10 blocks
results = blockwise_stats(data, (10, 10), stats=[:mean, :variance])
mean_result = results[:mean]
var_result = results[:variance]
```
"""
function blockwise_stats(data::AbstractArray{T,N}, window_sizes::NTuple{N,Int};
                         stats::AbstractVector{Symbol}=[:mean],
                         corrected::Bool=true) where {T,N}
    
    window = WindowConfig(window_sizes, window_sizes, :valid)
    
    # Validate window divides evenly
    for i in 1:N
        size(data, i) % window_sizes[i] == 0 || 
            error("Window size $(window_sizes[i]) does not divide dimension $(size(data,i)) evenly")
    end
    
    return _compute_blockwise_stats_impl(data, window, stats; corrected=corrected)
end

"""
    blockwise_mean(data, window_sizes)

Compute blockwise mean (convenience wrapper).

# Example
```julia
means = blockwise_mean(data, (10, 10, 5))
```
"""
function blockwise_mean(data::AbstractArray{T,N}, window_sizes::NTuple{N,Int}) where {T,N}
    results = blockwise_stats(data, window_sizes; stats=[:mean])
    return results[:mean]
end

"""
    blockwise_variance(data, window_sizes; corrected=true)

Compute blockwise variance using numerically stable parallel merge algorithm.

# Example
```julia
variances = blockwise_variance(data, (10, 10, 5), corrected=true)
```
"""
function blockwise_variance(data::AbstractArray{T,N}, window_sizes::NTuple{N,Int};
                           corrected::Bool=true) where {T,N}
    results = blockwise_stats(data, window_sizes; stats=[:variance], corrected=corrected)
    return results[:variance]
end

"""
    blockwise_std(data, window_sizes; corrected=true)

Compute blockwise standard deviation.

# Example
```julia
stds = blockwise_std(data, (10, 10, 5))
```
"""
function blockwise_std(data::AbstractArray{T,N}, window_sizes::NTuple{N,Int};
                      corrected::Bool=true) where {T,N}
    results = blockwise_stats(data, window_sizes; stats=[:std], corrected=corrected)
    return results[:std]
end

"""
    blockwise_covariance(x, y, window_sizes; corrected=true)

Compute blockwise covariance between two arrays.

# Example
```julia
covs = blockwise_covariance(ql, w, (10, 10, 5))
```
"""
function blockwise_covariance(x::AbstractArray{T,N}, y::AbstractArray{T,N}, 
                              window_sizes::NTuple{N,Int};
                              corrected::Bool=true) where {T,N}
    
    size(x) == size(y) || throw(DimensionMismatch("x and y must have same shape"))
    
    window = WindowConfig(window_sizes, window_sizes, :valid)
    
    # Calculate output dimensions
    out_dims = ntuple(i -> div(size(x, i), window_sizes[i]), N)
    result = similar(x, out_dims)
    
    # Compute covariance per block using CovarianceAccumulator
    for I in CartesianIndices(result)
        start_idx = ntuple(i -> (I[i] - 1) * window_sizes[i] + 1, N)
        
        acc = CovarianceAccumulator{T}()
        
        inner_indices = CartesianIndices(ntuple(i -> start_idx[i]:start_idx[i]+window_sizes[i]-1, N))
        for J in inner_indices
            fit!(acc, x[J], y[J])
        end
        
        result[I] = Statistics.cov(acc; corrected=corrected)
    end
    
    return result
end

"""
    blockwise_moments(data, window_sizes, max_order::Int)

Compute raw moments up to `max_order` for each block.

Returns array of Dicts where each Dict contains moments 1..max_order.

# Example
```julia
# Compute first 4 moments (mean, variance-related, skewness, kurtosis)
moments = blockwise_moments(data, (10, 10), 4)
mean_val = moments[i, j][1]      # First moment = mean
second_moment = moments[i, j][2]  # E[x²]
```
"""
function blockwise_moments(data::AbstractArray{T,N}, window_sizes::NTuple{N,Int}, 
                           max_order::Int) where {T,N}
    
    max_order >= 1 || throw(ArgumentError("max_order must be >= 1"))
    
    window = WindowConfig(window_sizes, window_sizes, :valid)
    out_dims = ntuple(i -> div(size(data, i), window_sizes[i]), N)
    
    # Result is array of tuples (moments)
    result = Array{NTuple{max_order,T},N}(undef, out_dims)
    
    for I in CartesianIndices(result)
        start_idx = ntuple(i -> (I[i] - 1) * window_sizes[i] + 1, N)
        
        acc = RawMomentsAccumulator{T,max_order}()
        
        inner_indices = CartesianIndices(ntuple(i -> start_idx[i]:start_idx[i]+window_sizes[i]-1, N))
        for J in inner_indices
            fit!(acc, data[J])
        end
        
        result[I] = acc.moments
    end
    
    return result
end

#
# Internal implementation
#

function _compute_blockwise_stats_impl(data::AbstractArray{T,N}, 
                                      window::WindowConfig{N},
                                      stats::AbstractVector{Symbol};
                                      corrected::Bool=true) where {T,N}
    
    out_dims = ntuple(i -> div(size(data, i), window.sizes[i]), N)
    results = Dict{Symbol, AbstractArray}()
    
    # Single stat optimization — delegate to canonical kernels
    if length(stats) == 1
        stat = stats[1]
        if stat == :mean
            results[:mean] = blockwise_mean_kernel(data, window.sizes)
        elseif stat == :variance
            results[:variance] = blockwise_variance_kernel(data, window.sizes; corrected=corrected)
        elseif stat == :std
            results[:std] = sqrt.(blockwise_variance_kernel(data, window.sizes; corrected=corrected))
        elseif stat == :min
            out = similar(data, T, out_dims)
            results[:min] = blockwise_min!(out, data, window.sizes)
        elseif stat == :max
            out = similar(data, T, out_dims)
            results[:max] = blockwise_max!(out, data, window.sizes)
        else
            error("Unknown statistic: $stat")
        end
        return results
    end
    
    # Multiple stats — use single-pass mean+variance kernel where possible
    if :mean in stats || :variance in stats || :std in stats
        means = similar(data, T, out_dims)
        variances = similar(data, T, out_dims)
        blockwise_mean_variance!(means, variances, data, window.sizes; corrected=corrected)
        
        :mean in stats && (results[:mean] = means)
        :variance in stats && (results[:variance] = variances)
        :std in stats && (results[:std] = sqrt.(variances))
    end
    
    # Min/Max computed separately via canonical kernels
    if :min in stats
        out = similar(data, T, out_dims)
        results[:min] = blockwise_min!(out, data, window.sizes)
    end
    if :max in stats
        out = similar(data, T, out_dims)
        results[:max] = blockwise_max!(out, data, window.sizes)
    end
    
    return results
end
