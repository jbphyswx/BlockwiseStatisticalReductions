"""
    Hybrid reduction mode - combine blockwise and sliding window operations.

Supports workflows like: blockwise coarsening followed by sliding analysis on coarsened data.
"""

#
# Hybrid reduction specification
#

"""
    HybridReductionSpec

Specification for hybrid reduction: blockwise reduction followed by sliding analysis.

# Fields
- `block_window::WindowConfig`: Blockwise reduction window
- `sliding_window::WindowConfig`: Sliding window for analysis on coarsened data
- `block_stats::Vector{Symbol}`: Statistics for blockwise phase
- `sliding_stats::Vector{Symbol}`: Statistics for sliding phase
"""
struct HybridReductionSpec{N}
    block_window::WindowConfig{N}
    sliding_window::WindowConfig{N}
    block_stats::Vector{Symbol}
    sliding_stats::Vector{Symbol}
end

"""
    HybridReductionResult

Result of hybrid reduction containing both blockwise and sliding outputs.

# Fields
- `block_result::ReductionResult`: Output from blockwise phase
- `sliding_result::ReductionResult`: Output from sliding phase on coarsened data
"""
struct HybridReductionResult
    block_result::ReductionResult
    sliding_result::ReductionResult
end

#
# Hybrid execution
#

"""
    execute_hybrid(data::AbstractArray{T,N}, spec::HybridReductionSpec{N}) where {T,N}

Execute hybrid reduction: blockwise coarsening then sliding analysis.

# Algorithm
1. Apply blockwise reduction to coarsen data
2. Apply sliding window analysis on coarsened result
3. Return both intermediate and final results

# Example
```julia
spec = HybridReductionSpec(
    WindowConfig((10, 10, 5), (10, 10, 5), :valid),    # Block: 10x10x5
    WindowConfig((3, 3, 3), (1, 1, 1), :same),         # Sliding: 3x3x3
    [:mean],                                           # Block stats
    [:variance]                                         # Sliding stats
)

result = execute_hybrid(data, spec)
block_output = result.block_result.data
sliding_output = result.sliding_result.data  # On coarsened data
```
"""
function execute_hybrid(data::AbstractArray{T,N}, spec::HybridReductionSpec{N}) where {T,N}
    
    # Phase 1: Blockwise reduction
    block_result = _execute_blockwise_phase(data, spec.block_window, spec.block_stats)
    
    # Phase 2: Sliding analysis on coarsened data
    sliding_result = _execute_sliding_phase(
        block_result.data, 
        spec.sliding_window, 
        spec.sliding_stats
    )
    
    return HybridReductionResult(block_result, sliding_result)
end

function _execute_blockwise_phase(data::AbstractArray{T,N}, 
                                   window::WindowConfig{N},
                                   stats::Vector{Symbol}) where {T,N}
    
    # Calculate output dimensions
    out_dims = ntuple(i -> div(size(data, i), window.sizes[i]), N)
    
    # Compute blockwise statistics
    if length(stats) == 1 && stats[1] == :mean
        result_data = _compute_hybrid_blockwise_mean(data, window.sizes, out_dims)
    elseif length(stats) == 1 && stats[1] == :variance
        result_data = _compute_hybrid_blockwise_variance(data, window.sizes, out_dims)
    else
        # General case - use accumulators
        result_data = _compute_hybrid_blockwise_stats(data, window.sizes, out_dims, stats)
    end
    
    # Wrap in ReductionResult
    return ReductionResult(result_data, Dict{Symbol,Any}(
        :phase => :blockwise,
        :window => window,
        :stats => stats,
        :input_shape => size(data)
    ))
end

function _compute_hybrid_blockwise_mean(data::AbstractArray{T,N}, 
                                         window_sizes::NTuple{N,Int},
                                         out_dims::NTuple{N,Int}) where {T,N}
    
    result = similar(data, out_dims)
    
    for I in CartesianIndices(result)
        start_idx = ntuple(i -> (I[i] - 1) * window_sizes[i] + 1, N)
        
        s = zero(T)
        count = 0
        
        inner_indices = CartesianIndices(ntuple(i -> start_idx[i]:start_idx[i]+window_sizes[i]-1, N))
        for J in inner_indices
            s += data[J]
            count += 1
        end
        
        result[I] = s / count
    end
    
    return result
end

function _compute_hybrid_blockwise_variance(data::AbstractArray{T,N},
                                           window_sizes::NTuple{N,Int},
                                           out_dims::NTuple{N,Int}) where {T,N}
    
    result = similar(data, out_dims)
    
    for I in CartesianIndices(result)
        start_idx = ntuple(i -> (I[i] - 1) * window_sizes[i] + 1, N)
        
        acc = VarianceAccumulator{T}()
        
        inner_indices = CartesianIndices(ntuple(i -> start_idx[i]:start_idx[i]+window_sizes[i]-1, N))
        for J in inner_indices
            fit!(acc, data[J])
        end
        
        result[I] = Statistics.var(acc)
    end
    
    return result
end

function _compute_hybrid_blockwise_stats(data::AbstractArray{T,N},
                                        window_sizes::NTuple{N,Int},
                                        out_dims::NTuple{N,Int},
                                        stats::Vector{Symbol}) where {T,N}
    
    # Return Dict for multi-stat
    results = Dict{Symbol, AbstractArray}()
    
    # For now, just compute mean (extend as needed)
    results[:mean] = _compute_hybrid_blockwise_mean(data, window_sizes, out_dims)
    
    return results
end

function _execute_sliding_phase(data::AbstractArray{T,N},
                                window::WindowConfig{N},
                                stats::Vector{Symbol}) where {T,N}
    
    # Calculate output dimensions based on padding mode
    out_dims = _compute_sliding_output_dims(size(data), window.sizes, window.strides, window.padding)
    
    # For now, implement simple mean sliding window
    if :mean in stats
        result_data = _compute_hybrid_sliding_mean(data, window, out_dims)
    else
        error("Sliding phase stats other than :mean not yet implemented")
    end
    
    # Wrap in ReductionResult
    return ReductionResult(result_data, Dict{Symbol,Any}(
        :phase => :sliding,
        :window => window,
        :stats => stats,
        :input_shape => size(data)
    ))
end

function _compute_sliding_output_dims(input_shape::NTuple{N,Int},
                                     window_sizes::NTuple{N,Int},
                                     strides::NTuple{N,Int},
                                     padding::Symbol) where N
    
    if padding == :valid
        # No padding - output is smaller
        return ntuple(i -> div(input_shape[i] - window_sizes[i], strides[i]) + 1, N)
    elseif padding == :same
        # Pad to keep size same
        return input_shape
    else
        error("Unknown padding mode: $padding")
    end
end

function _compute_hybrid_sliding_mean(data::AbstractArray{T,N},
                                     window::WindowConfig{N},
                                     out_dims::NTuple{N,Int}) where {T,N}
    
    result = similar(data, out_dims)
    
    for I in CartesianIndices(result)
        # Calculate window center based on output index and stride
        center_idx = ntuple(i -> (I[i] - 1) * window.strides[i] + 1, N)
        
        # Calculate window bounds
        half_window = ntuple(i -> div(window.sizes[i], 2), N)
        start_idx = ntuple(i -> max(1, center_idx[i] - half_window[i]), N)
        end_idx = ntuple(i -> min(size(data, i), center_idx[i] + half_window[i]), N)
        
        # Compute mean over window
        s = zero(T)
        count = 0
        
        inner_indices = CartesianIndices(ntuple(i -> start_idx[i]:end_idx[i], N))
        for J in inner_indices
            s += data[J]
            count += 1
        end
        
        result[I] = s / count
    end
    
    return result
end

#
# Convenience API
#

"""
    hybrid_reduction(data::AbstractArray;
                    block_sizes::NTuple{N,Int},
                    sliding_sizes::NTuple{N,Int},
                    block_stats=[:mean],
                    sliding_stats=[:variance]) where N

High-level API for hybrid blockwise + sliding reduction.

# Example
```julia
result = hybrid_reduction(data,
    block_sizes=(10, 10, 5),
    sliding_sizes=(3, 3, 3),
    block_stats=[:mean],
    sliding_stats=[:variance]
)

# Get results
coarsened = result.block_result.data
analysis = result.sliding_result.data  # Sliding on coarsened
```
"""
function hybrid_reduction(data::AbstractArray{T,N};
                         block_sizes::NTuple{N,Int},
                         sliding_sizes::NTuple{N,Int},
                         block_stats=[:mean],
                         sliding_stats=[:variance]) where {T,N}
    
    spec = HybridReductionSpec(
        WindowConfig(block_sizes, block_sizes, :valid),
        WindowConfig(sliding_sizes, ntuple(i -> 1, N), :same),
        block_stats,
        sliding_stats
    )
    
    return execute_hybrid(data, spec)
end
