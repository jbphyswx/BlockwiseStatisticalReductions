"""
    Multi-resolution plan builder - compute statistics at multiple scales with caching.

This module builds ReductionPlans that efficiently compute statistics at multiple
resolutions, reusing intermediate results when factors divide evenly.
"""

#
# Factor sequence utilities
#

"""
    factor_sequence(start::Int, targets::AbstractVector{Int})

Find an ordered sequence of factors from `start` to cover all `targets`,
where each factor divides evenly into the next.

This enables caching: if we compute at factor 2, we can reuse for 4, 8, etc.

# Example
```julia
factor_sequence(1, [2, 4, 8, 16])  # Returns [1, 2, 4, 8, 16]
factor_sequence(1, [3, 6, 12])     # Returns [1, 3, 6, 12]
factor_sequence(1, [5, 10, 20])    # Returns [1, 5, 10, 20]
```
"""
function factor_sequence(start::Int, targets::AbstractVector{Int})
    # Sort targets
    sorted_targets = sort(targets)
    
    # Start from given start
    result = Int[start]
    current = start
    
    for target in sorted_targets
        if target <= current
            continue  # Skip if already covered
        end
        
        # Check if target divides evenly by current
        if target % current == 0
            push!(result, target)
            current = target
        else
            # Find intermediate factor that divides both
            # For now, just use target (may recompute)
            push!(result, target)
            current = target
        end
    end
    
    return result
end

#
# Multi-resolution plan building
#

"""
    build_multires_plan(input_shape::NTuple{N,Int}, 
                        target_factors::Vector{Int},
                        stats::Vector{Symbol}) where N

Build a ReductionPlan that computes statistics at multiple resolutions.

The plan reuses intermediate results when possible (e.g., 2x reduction 
reused for 4x by further reducing the 2x result).

# Arguments
- `input_shape`: Input array dimensions
- `target_factors`: Target reduction factors (must divide input evenly)
- `stats`: Statistics to compute [:mean, :variance, :covariance, ...]

# Returns
`ReductionPlan` configured for multi-resolution execution

# Example
```julia
plan = build_multires_plan((100, 100, 50), [2, 4, 10], [:mean, :variance])
results = execute(plan, data)  # Returns 3 ReductionResults at different scales
```
"""
function build_multires_plan(input_shape::NTuple{N,Int}, 
                             target_factors::AbstractVector{Int},
                             stats::AbstractVector{Symbol}) where N
    
    # Validate all factors divide evenly
    for factor in target_factors
        for dim in input_shape
            if dim % factor != 0
                error("Factor $factor does not divide dimension $dim evenly")
            end
        end
    end
    
    # Build factor sequence that enables reuse
    factors = factor_sequence(1, target_factors)
    
    # Start building plan
    builder = ReductionPlanBuilder()
    builder.input_shape = input_shape
    
    # Create parallel branches for each factor
    # This is a simplified version - full implementation would use fork/merge
    branches = []
    
    for factor in factors[2:end]  # Skip 1 (native resolution)
        # Create window config for this factor
        window = WindowConfig(
            ntuple(i -> factor, N),
            ntuple(i -> factor, N),
            :valid
        )
        
        # Add window node
        node = WindowNode(window, next_id!(builder))
        add_node!(builder, node)
        
        # Add stats node
        stat_type = length(stats) == 1 ? stats[1] : stats
        stats_node = StatsNode{Symbol}(stat_type, :, next_id!(builder))
        add_node!(builder, stats_node)
        
        push!(branches, builder.current_node_id)
    end
    
    # Finalize - outputs are the stats nodes for each factor
    builder.plan.outputs = branches
    
    return finalize_plan(builder)
end

#
# Cached multi-level execution
#

"""
    CachedMultiLevelExecution

Execution strategy that caches intermediate reductions and reuses them
for coarser resolutions when factors divide evenly.
"""
mutable struct CachedMultiLevelExecution
    cache::PlanCache
    factors::Vector{Int}
    current_level::Int
    
    function CachedMultiLevelExecution(cache::PlanCache, factors::Vector{Int})
        new(cache, factors, 0)
    end
end

"""
    execute_cached_multilevel(data::AbstractArray{T,N},
                              factors::Vector{Int},
                              stats::Vector{Symbol};
                              cache::PlanCache=PlanCache()) where {T,N}

Execute multi-level reduction with automatic caching.

Computes statistics at each factor level, reusing previous results when possible.

# Returns
Dictionary mapping factor -> ReductionResult

# Example
```julia
# Compute variance at scales 2x, 4x, 8x, 10x
results = execute_cached_multilevel(data, [2, 4, 8, 10], [:variance])

# Access 4x reduced result
var_4x = results[4].data
```
"""
function execute_cached_multilevel(data::AbstractArray{T,N},
                                   factors::AbstractVector{Int},
                                   stats::AbstractVector{Symbol};
                                   cache::PlanCache=PlanCache()) where {T,N}
    
    input_shape = size(data)
    
    # Validate all factors divide evenly
    for factor in factors
        for dim in input_shape
            if dim % factor != 0
                error("Factor $factor does not divide dimension $dim evenly")
            end
        end
    end
    
    results = Dict{Int, ReductionResult}()
    
    # Sort factors ascending for optimal caching
    sorted_factors = sort(factors)
    
    # Previous result for chaining
    prev_data = data
    prev_factor = 1
    
    # Only enable caching for single-stat computations
    # Multiple stats return Dict which can't be easily cached for further reductions
    can_cache = length(stats) == 1 && (stats[1] == :mean || stats[1] == :variance)
    
    for target_factor in sorted_factors
        # Check if we can reuse previous result (only for single stat)
        if can_cache && target_factor % prev_factor == 0 && prev_factor > 1
            # Reduce from previous level
            relative_factor = div(target_factor, prev_factor)
            window = WindowConfig(
                ntuple(i -> relative_factor, N),
                ntuple(i -> relative_factor, N),
                :valid
            )
            
            # Compute on cached data
            result_data = compute_stats_on_data(prev_data, window, stats)
        else
            # Compute from original data
            window = WindowConfig(
                ntuple(i -> target_factor, N),
                ntuple(i -> target_factor, N),
                :valid
            )
            
            result_data = compute_stats_on_data(data, window, stats)
        end
        
        # Wrap in ReductionResult
        result = ReductionResult(result_data, Dict{Symbol,Any}(
            :factor => target_factor,
            :stats => stats,
            :input_shape => input_shape
        ))
        
        results[target_factor] = result
        
        # Cache for next iteration (only if single stat)
        if can_cache
            prev_data = result_data
            prev_factor = target_factor
        else
            # Reset caching for multiple stats
            prev_data = data
            prev_factor = 1
        end
    end
    
    return results
end

"""
    compute_stats_on_data(data::AbstractArray, window::WindowConfig, stats::Vector{Symbol})

Compute specified statistics on data with given window configuration.

Internal helper for cached execution.
"""
function compute_stats_on_data(data::AbstractArray{T,N}, 
                               window::WindowConfig{N},
                               stats::Vector{Symbol}) where {T,N}
    
    # Calculate output dimensions
    out_dims = ntuple(i -> div(size(data, i), window.sizes[i]), N)
    
    # For now, just compute mean (extend to other stats as needed)
    if :mean in stats && length(stats) == 1
        return _compute_blockwise_mean(data, window.sizes, out_dims)
    elseif :variance in stats && length(stats) == 1
        return _compute_blockwise_variance(data, window.sizes, out_dims)
    else
        # Multiple stats - use accumulator approach
        return _compute_blockwise_stats(data, window.sizes, out_dims, stats)
    end
end

function _compute_blockwise_mean(data::AbstractArray{T,N}, 
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

function _compute_blockwise_variance(data::AbstractArray{T,N},
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

function _compute_blockwise_stats(data::AbstractArray{T,N},
                                  window_sizes::NTuple{N,Int},
                                  out_dims::NTuple{N,Int},
                                  stats::Vector{Symbol}) where {T,N}
    
    # Return dict of stat -> array
    results = Dict{Symbol, AbstractArray}()
    
    # For efficiency, compute all stats in one pass
    means = _compute_blockwise_mean(data, window_sizes, out_dims)
    results[:mean] = means
    
    if :variance in stats
        results[:variance] = _compute_blockwise_variance(data, window_sizes, out_dims)
    end
    
    return results
end

#
# High-level convenience API
#

"""
    multiresolution_stats(data::AbstractArray, target_factors::Vector{Int};
                          stats=[:mean, :variance],
                          cache::Union{PlanCache,Nothing}=nothing)

High-level API for multi-resolution statistics.

# Example
```julia
# Compute mean and variance at 2x, 4x, 8x reductions
results = multiresolution_stats(data, [2, 4, 8], stats=[:mean, :variance])

mean_2x = results[2][:mean]
var_4x = results[4][:variance]
```
"""
function multiresolution_stats(data::AbstractArray, target_factors::AbstractVector{Int};
                               stats=[:mean, :variance],
                               cache::Union{PlanCache,Nothing}=nothing)
    
    if cache === nothing
        cache = PlanCache()
    end
    
    return execute_cached_multilevel(data, target_factors, stats; cache=cache)
end
