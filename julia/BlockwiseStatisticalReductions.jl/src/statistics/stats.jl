"""
    create_stat(stat_type::Symbol, T::Type=Float64)

Create an OnlineStats object for the given statistic type.

Supported types:
- :mean → Mean{T}
- :var, :variance → Variance{T}
- :std → Series(Mean{T}(), Variance{T}())
- :moments → Series(Mean{T}(), Variance{T}(), Moments{T}(3), Moments{T}(4))
- :min, :max → Extrema{T}
- :sum → Sum{T}
- :count → Count{T}
"""
function create_stat(stat_type::Symbol, T::Type=Float64)
    if stat_type == :mean
        return OnlineStats.Mean(T)
    elseif stat_type in (:var, :variance)
        return OnlineStats.Variance(T; weight=OnlineStats.EqualWeight())
    elseif stat_type == :std
        return OnlineStats.Series(OnlineStats.Mean(T), OnlineStats.Variance(T; weight=OnlineStats.EqualWeight()))
    elseif stat_type == :moments
        return OnlineStats.Series(OnlineStats.Mean(T), OnlineStats.Variance(T; weight=OnlineStats.EqualWeight()), 
                      OnlineStats.Moments(T, 3), OnlineStats.Moments(T, 4))
    elseif stat_type == :min
        return OnlineStats.Extrema(T)
    elseif stat_type == :max
        return OnlineStats.Extrema(T)
    elseif stat_type == :sum
        return OnlineStats.Sum(T)
    elseif stat_type == :count
        return OnlineStats.Count(T)
    else
        error("Unknown statistic type: $stat_type")
    end
end

# Multiple stats at once
function create_stat(stat_types::Vector{Symbol}, T::Type=Float64)
    return OnlineStats.Series([create_stat(s, T) for s in stat_types]...)
end

"""
    fit_window!(stat, view::AbstractArray)

Fit a window of data to an OnlineStats object.
"""
function fit_window!(stat, view::AbstractArray)
    # Use Welford-style online algorithm via OnlineStats
    for x in view
        OnlineStats.fit!(stat, x)
    end
    return stat
end

"""
    window_stat(view::AbstractArray, stat_type::Symbol)

Compute a statistic over a window using OnlineStats.
"""
function window_stat(view::AbstractArray, stat_type::Symbol)
    T = eltype(view)
    stat = create_stat(stat_type, T)
    fit_window!(stat, view)
    return OnlineStats.value(stat)
end

"""
    window_stat(view::AbstractArray, stat_types::Vector{Symbol})

Compute multiple statistics over a window in one pass.
"""
function window_stat(view::AbstractArray, stat_types::Vector{Symbol})
    T = eltype(view)
    stat = create_stat(stat_types, T)
    fit_window!(stat, view)
    return [OnlineStats.value(s) for s in stat.stats]
end

"""
    merge_stats(s1, s2)

Merge two OnlineStats objects (for tree reductions).
"""
merge_stats(s1, s2) = OnlineStats.merge!(s1, s2)

"""
    finalize_stat(stat)

Extract the final value(s) from a statistic object.
"""
finalize_stat(stat) = OnlineStats.value(stat)

"""
    rolling_stats(array::AbstractArray, config::WindowConfig, stat_type::Symbol)

Compute rolling window statistics over an array.

Returns a vector of `(result, metadata)` tuples.
"""
function rolling_stats(array::AbstractArray, config::WindowConfig, stat_type::Symbol)
    results = []
    for (view, meta) in rolling_views(array, config)
        stat = window_stat(view, stat_type)
        push!(results, (stat, meta))
    end
    return results
end

"""
    combine_rolling_stats(results::Vector, combine_op::Symbol)

Combine rolling statistics using a tree reduction.

`combine_op` can be:
- :merge for OnlineStats objects
- Custom function that takes two results and returns combined result
"""
function combine_rolling_stats(results::Vector{T}, combine_op::Symbol=:merge) where T
    if combine_op == :merge
        op = (a, b) -> merge_stats(a[1], b[1])
        final = tree_reduce_impl(results, op, CPUBackend())
        return final
    else
        error("Unknown combine_op: $combine_op")
    end
end

"""
    combine_rolling_stats(results::Vector{T}, op) where T

Combine with user-provided function.
"""
function combine_rolling_stats(results::Vector{T}, op::Function) where T
    return tree_reduce_impl(results, op, CPUBackend())
end

"""
    rolling_histogram(array::AbstractArray, config::WindowConfig, edges)

Compute rolling window histograms (for heatmaps).

Returns vector of `(hist, metadata)` where hist is a StatsBase.Histogram.
"""
function rolling_histogram(array::AbstractArray, config::WindowConfig, edges)
    results = []
    for (view, meta) in rolling_views(array, config)
        h = StatsBase.fit(StatsBase.Histogram, vec(view), edges)
        push!(results, (h, meta))
    end
    return results
end

"""
    adaptive_rolling_histogram(array::AbstractArray, config::WindowConfig, n_bins::Int)

Compute rolling histograms with adaptive (data-driven) binning per window.
"""
function adaptive_rolling_histogram(array::AbstractArray, config::WindowConfig, n_bins::Int)
    results = []
    for (view, meta) in rolling_views(array, config)
        # Use quantile-based binning
        v = vec(view)
        edges = Statistics.quantile(v, range(0, 1, length=n_bins+1))
        h = StatsBase.fit(StatsBase.Histogram, v, edges)
        push!(results, (h, meta))
    end
    return results
end

"""
    tiled_stats(array::AbstractArray, config::WindowConfig, stat_type::Symbol)

Compute statistics for non-overlapping (blockwise) tiles using OnlineStats merge.

This is optimized for blockwise windows where stride == window_size. Instead of
materializing all windows and computing each separately, this uses OnlineStats'
mergeable property to compute partial statistics and combine them hierarchically.

Returns a vector of `(result, metadata)` tuples.
"""
function tiled_stats(array::AbstractArray, config::WindowConfig, stat_type::Symbol)
    # Validate blockwise configuration (no overlap)
    for i in 1:ndims(config)
        if config.strides[i] != config.sizes[i]
            error("tiled_stats requires non-overlapping windows (stride == size). " *
                  "Got stride $(config.strides[i]) != size $(config.sizes[i]) for dimension $i. " *
                  "Use rolling_stats for overlapping windows.")
        end
    end
    
    # Use strict validation for exact divisibility
    BlockwiseStatisticalReductions.validate_window_config(size(array), config; strict=true)
    
    # Compute tiled statistics
    results = []
    for (view, meta) in BlockwiseStatisticalReductions.rolling_views(array, config)
        stat = create_stat(stat_type, eltype(view))
        fit_window!(stat, view)
        
        # Store the OnlineStats object (mergeable for later tree reduction)
        push!(results, (stat, meta))
    end
    
    return results
end

"""
    tiled_stats_merge(results::Vector, combine_op=:merge)

Merge tiled statistics using OnlineStats merge operations.

For tree reductions, this efficiently combines partial statistics without
recomputing from raw data.
"""
function tiled_stats_merge(results::Vector, combine_op::Symbol=:merge)
    if isempty(results)
        return nothing
    elseif length(results) == 1
        return results[1]
    end
    
    # Extract statistics from results
    stats = [r[1] for r in results]
    
    # Tree reduction using merge
    while length(stats) > 1
        new_stats = []
        for i in 1:2:length(stats)
            if i + 1 <= length(stats)
                # Merge pair
                merged = OnlineStats.merge!(deepcopy(stats[i]), stats[i+1])
                push!(new_stats, merged)
            else
                # Odd element out
                push!(new_stats, stats[i])
            end
        end
        stats = new_stats
    end
    
    # Return final merged result with metadata from first window
    return (stats[1], results[1][2])
end
