"""
    Product coarsening - compute statistics of products without materializing intermediates.

These functions compute <x*y>, <x²>, <y²>, and covariance in fused loops,
avoiding temporary arrays for memory efficiency.
"""

#
# Product mean (simple <x*y> without intermediate array)
#

"""
    product_mean(x::AbstractArray{T,N}, y::AbstractArray{T,N}, 
                 window::WindowConfig{N}) where {T,N}

Compute the mean of element-wise product `<x*y>` over windows without 
materializing the intermediate product array.

This is memory-efficient for large arrays - instead of computing `x.*y` 
(which allocates) then reducing, we fuse the multiply and accumulate.

# Example
```julia
x = rand(100, 100, 50)
y = rand(100, 100, 50)
win = WindowConfig((10, 10, 5), (10, 10, 5))  # 10x10x5 blocks

result = product_mean(x, y, win)  # Returns (10, 10, 10) array of <x*y> per block
```
"""
function product_mean(x::AbstractArray{T,N}, y::AbstractArray{T,N}, 
                      window::WindowConfig{N}) where {T,N}
    size(x) == size(y) || throw(DimensionMismatch("x and y must have same shape"))
    
    # Validate window configuration
    validate_window_config(size(x), window; strict=true)
    
    # Calculate output dimensions
    out_dims = ntuple(i -> div(size(x, i), window.sizes[i]), N)
    result = similar(x, out_dims)
    
    # Compute block-wise product means
    _product_mean_impl!(result, x, y, window.sizes)
    
    return result
end

function _product_mean_impl!(out::AbstractArray{T,M}, x::AbstractArray{T,N}, 
                             y::AbstractArray{T,N}, window_sizes::NTuple{N,Int}) where {T,N,M}
    
    out_idx = CartesianIndices(out)
    
    for I in out_idx
        # Calculate window start for this output index
        start_idx = ntuple(i -> (I[i] - 1) * window_sizes[i] + 1, N)
        
        # Accumulate product sum
        s = zero(T)
        count = 0
        
        # Inner loop over window
        inner_indices = CartesianIndices(ntuple(i -> start_idx[i]:start_idx[i]+window_sizes[i]-1, N))
        for J in inner_indices
            s += x[J] * y[J]
            count += 1
        end
        
        out[I] = s / count
    end
    
    return out
end

#
# Joint moments - compute mean, variance, covariance in one pass
#

"""
    JointMomentsResult{T,N}

Result of joint moments computation containing:
- `mean_x`: Mean of x
- `mean_y`: Mean of y  
- `var_x`: Variance of x
- `var_y`: Variance of y
- `cov_xy`: Covariance between x and y
- `n_samples`: Number of samples per window
"""
struct JointMomentsResult{T,N}
    mean_x::Array{T,N}
    mean_y::Array{T,N}
    var_x::Array{T,N}
    var_y::Array{T,N}
    cov_xy::Array{T,N}
    n_samples::Array{Int,N}
end

"""
    product_moments(x::AbstractArray{T,N}, y::AbstractArray{T,N},
                    window::WindowConfig{N}; corrected::Bool=true) where {T,N}

Compute joint moments (means, variances, covariance) of two arrays over windows.

All statistics computed in one fused pass - no intermediate allocations.

# Returns
`JointMomentsResult` containing:
- `mean_x`, `mean_y`: Per-window means
- `var_x`, `var_y`: Per-window variances  
- `cov_xy`: Per-window covariances
- `n_samples`: Sample counts per window

# Example
```julia
x = rand(100, 100, 50)  # e.g., liquid water
y = rand(100, 100, 50)  # e.g., vertical velocity
win = WindowConfig((10, 10, 5))

moments = product_moments(x, y, win)

# Compute TKE contribution: <w'w'> = <w*w> - <w>^2
tke = moments.mean_y  # mean of w
w_variance = moments.var_y  # variance of w
```
"""
function product_moments(x::AbstractArray{T,N}, y::AbstractArray{T,N},
                         window::WindowConfig{N}; corrected::Bool=true) where {T,N}
    size(x) == size(y) || throw(DimensionMismatch("x and y must have same shape"))
    
    # Validate window configuration
    validate_window_config(size(x), window; strict=true)
    
    # Calculate output dimensions
    out_dims = ntuple(i -> div(size(x, i), window.sizes[i]), N)
    
    # Allocate output arrays
    mean_x = similar(x, out_dims)
    mean_y = similar(x, out_dims)
    var_x = similar(x, out_dims)
    var_y = similar(x, out_dims)
    cov_xy = similar(x, out_dims)
    n_samples = similar(x, Int, out_dims)
    
    # Compute all moments in one pass
    _product_moments_impl!(mean_x, mean_y, var_x, var_y, cov_xy, n_samples,
                          x, y, window.sizes; corrected=corrected)
    
    return JointMomentsResult(mean_x, mean_y, var_x, var_y, cov_xy, n_samples)
end

function _product_moments_impl!(mean_x::AbstractArray, mean_y::AbstractArray,
                                var_x::AbstractArray, var_y::AbstractArray,
                                cov_xy::AbstractArray, n_samples::AbstractArray{Int},
                                x::AbstractArray{T,N}, y::AbstractArray{T,N},
                                window_sizes::NTuple{N,Int}; 
                                corrected::Bool=true) where {T,N}
    
    out_idx = CartesianIndices(mean_x)
    
    for I in out_idx
        # Calculate window start
        start_idx = ntuple(i -> (I[i] - 1) * window_sizes[i] + 1, N)
        
        # Use VarianceAccumulator and CovarianceAccumulator for numerically stable computation
        acc_x = VarianceAccumulator{T}()
        acc_y = VarianceAccumulator{T}()
        acc_xy = CovarianceAccumulator{T}()
        
        # Accumulate over window
        inner_indices = CartesianIndices(ntuple(i -> start_idx[i]:start_idx[i]+window_sizes[i]-1, N))
        for J in inner_indices
            fit!(acc_x, x[J])
            fit!(acc_y, y[J])
            fit!(acc_xy, x[J], y[J])
        end
        
        # Store results
        mean_x[I] = Statistics.mean(acc_x)
        mean_y[I] = Statistics.mean(acc_y)
        var_x[I] = Statistics.var(acc_x; corrected=corrected)
        var_y[I] = Statistics.var(acc_y; corrected=corrected)
        cov_xy[I] = Statistics.cov(acc_xy; corrected=corrected)
        n_samples[I] = acc_x.count
    end
    
    return nothing
end

#
# Convenience functions for common use cases
#

"""
    product_variance(x::AbstractArray, y::AbstractArray, window::WindowConfig)

Compute variance of products: Var(x*y) over windows.

Uses identity: Var(x*y) = <(xy)²> - <xy>² computed via accumulators.
"""
function product_variance(x::AbstractArray{T,N}, y::AbstractArray{T,N}, 
                          window::WindowConfig{N}; corrected::Bool=true) where {T,N}
    # For product variance, we need E[(xy)²] and E[xy]²
    # Compute via raw moments on xy
    
    size(x) == size(y) || throw(DimensionMismatch("x and y must have same shape"))
    validate_window_config(size(x), window; strict=true)
    
    out_dims = ntuple(i -> div(size(x, i), window.sizes[i]), N)
    result = similar(x, out_dims)
    
    out_idx = CartesianIndices(result)
    
    for I in out_idx
        start_idx = ntuple(i -> (I[i] - 1) * window.sizes[i] + 1, N)
        
        # Accumulate moments of product
        acc = RawMomentsAccumulator{T,2}()
        
        inner_indices = CartesianIndices(ntuple(i -> start_idx[i]:start_idx[i]+window.sizes[i]-1, N))
        for J in inner_indices
            fit!(acc, x[J] * y[J])
        end
        
        # Variance = E[(xy)²] - E[xy]²
        E_xy = acc.moments[1]
        E_xy_sq = acc.moments[2]
        var_pop = E_xy_sq - E_xy^2
        
        if corrected && acc.count > 1
            result[I] = var_pop * acc.count / (acc.count - 1)
        else
            result[I] = var_pop
        end
    end
    
    return result
end

"""
    covariance_from_moments(mean_x, mean_y, mean_xy)

Compute covariance from means using identity: Cov(x,y) = <xy> - <x><y>

This is useful when you already have the marginal means and product mean.
"""
function covariance_from_moments(mean_x::AbstractArray, mean_y::AbstractArray, 
                                  mean_xy::AbstractArray)
    size(mean_x) == size(mean_y) == size(mean_xy) || 
        throw(DimensionMismatch("All inputs must have same shape"))
    
    return mean_xy .- mean_x .* mean_y
end

"""
    variance_from_moments(mean, mean_sq)

Compute variance from moments using identity: Var(x) = <x²> - <x>²

This is the population variance. For sample variance, multiply by n/(n-1).
"""
function variance_from_moments(mean::AbstractArray, mean_sq::AbstractArray)
    size(mean) == size(mean_sq) || throw(DimensionMismatch("Inputs must have same shape"))
    return mean_sq .- mean.^2
end

#
# Blockwise wrappers using WindowConfig
#

"""
    blockwise_product_mean(x, y; window_sizes, strict=true)

Convenience wrapper for product_mean with keyword arguments.
"""
function blockwise_product_mean(x::AbstractArray{T,N}, y::AbstractArray{T,N}; 
                                 window_sizes::NTuple{N,Int}, strict::Bool=true) where {T,N}
    window = WindowConfig(window_sizes, window_sizes, :valid)
    return product_mean(x, y, window)
end

"""
    blockwise_product_moments(x, y; window_sizes, corrected=true)

Convenience wrapper for product_moments with keyword arguments.
"""
function blockwise_product_moments(x::AbstractArray{T,N}, y::AbstractArray{T,N}; 
                                    window_sizes::NTuple{N,Int}, 
                                    corrected::Bool=true) where {T,N}
    window = WindowConfig(window_sizes, window_sizes, :valid)
    return product_moments(x, y, window; corrected=corrected)
end
