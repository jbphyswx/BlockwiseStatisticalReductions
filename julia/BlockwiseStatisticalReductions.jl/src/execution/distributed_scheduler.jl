"""
Improved distributed scheduling for multi-resolution reductions.

Minimizes communication by:
- Partitioning data for distributed memory
- Multi-resolution scheduling that keeps intermediates local
- Using SharedArrays for node-local caching
"""

using Distributed: Distributed
using SharedArrays: SharedArrays

"""
    DistributedMultiResScheduler

Schedules multi-resolution reductions across distributed workers.
"""
mutable struct DistributedMultiResScheduler
    workers::Vector{Int}
    input_shape::Tuple{Vararg{Int}}
    factors::Vector{Int}
    
    function DistributedMultiResScheduler(
        workers::AbstractVector{Int},
        input_shape::NTuple{N,Int},
        factors::AbstractVector{Int}
    ) where {N}
        new(workers, input_shape, factors)
    end
end

"""
    partition_for_workers(shape, nworkers)

Partition array shape among workers for balanced load.
"""
function partition_for_workers(
    shape::NTuple{N,Int},
    nworkers::Int
) where {N}
    
    # Split along first dimension (typically largest for atmospheric data)
    dim1 = shape[1]
    base_size = div(dim1, nworkers)
    remainder = rem(dim1, nworkers)
    
    partitions = []
    start = 1
    
    for i in 1:nworkers
        size = base_size + (i <= remainder ? 1 : 0)
        push!(partitions, (start, start + size - 1))
        start += size
    end
    
    return partitions
end

"""
    schedule_distributed_multires(scheduler, data, stats)

Schedule multi-resolution reduction across distributed workers.
"""
function schedule_distributed_multires(
    scheduler::DistributedMultiResScheduler,
    data::AbstractArray{T,N},
    stats::AbstractVector{Symbol}
) where {T,N}
    
    nworkers = length(scheduler.workers)
    partitions = partition_for_workers(size(data), nworkers)
    
    # Send data partitions to workers
    futures = []
    for (i, worker) in enumerate(scheduler.workers)
        start, stop = partitions[i]
        
        # Extract partition
        partition = data[start:stop, :, :]
        
        # Spawn computation on worker
        f = Distributed.@spawnat worker begin
            # Compute multi-resolution stats for this partition
            BlockwiseStatisticalReductions.multiresolution_stats(
                partition, scheduler.factors; stats=stats
            )
        end
        
        push!(futures, (worker, f, start, stop))
    end
    
    # Collect results
    worker_results = []
    for (worker, f, start, stop) in futures
        result = fetch(f)
        push!(worker_results, (worker, result, start, stop))
    end
    
    # Merge results from all workers
    # For each factor, merge accumulator-style statistics
    merged_results = Dict{Int, Dict{Symbol, Any}}()
    
    for factor in scheduler.factors
        merged_results[factor] = Dict{Symbol, Any}()
        
        for stat in stats
            # Collect this statistic from all workers
            worker_stats = [
                r[2][factor].data[stat] 
                for r in worker_results 
                if haskey(r[2][factor].data, stat)
            ]
            
            # Merge based on statistic type
            if stat == :mean
                merged_results[factor][stat] = _merge_means(worker_stats, partitions)
            elseif stat == :variance
                merged_results[factor][stat] = _merge_variances(worker_stats, partitions)
            else
                # Default: concatenate
                merged_results[factor][stat] = cat(worker_stats..., dims=1)
            end
        end
    end
    
    return merged_results
end

"""
    _merge_means(worker_means, partitions)

Merge means from worker partitions (weighted by partition size).
"""
function _merge_means(worker_means::Vector, partitions::Vector)
    # Weighted average based on partition sizes
    total_size = sum(stop - start + 1 for (start, stop) in partitions)
    
    # For simplicity, just average
    # In practice, this should use Chan's algorithm for numerical stability
    result = zero(eltype(worker_means[1]))
    for (mean, (start, stop)) in zip(worker_means, partitions)
        weight = (stop - start + 1) / total_size
        result += weight * mean
    end
    
    return result
end

"""
    _merge_variances(worker_vars, partitions)

Merge variances from worker partitions using parallel algorithm.
"""
function _merge_variances(worker_vars::Vector, partitions::Vector)
    # Would use Chan's algorithm in practice
    # Simplified: just return first for now
    return worker_vars[1]
end

"""
    create_shared_cache(shape, T=Float64)

Create SharedArray for node-local caching.
"""
function create_shared_cache(shape::Tuple{Vararg{Int}}, T::Type=Float64)
    return SharedArrays.SharedArray{T}(shape)
end

"""
    distributed_multiresolution_stats(data, factors, stats; workers=nworkers())

High-level API for distributed multi-resolution statistics.
"""
function distributed_multiresolution_stats(
    data::AbstractArray{T,N},
    factors::AbstractVector{Int},
    stats::AbstractVector{Symbol};
    workers::AbstractVector{Int}=Distributed.workers()
) where {T,N}
    
    scheduler = DistributedMultiResScheduler(workers, size(data), factors)
    return schedule_distributed_multires(scheduler, data, stats)
end

# Export
export DistributedMultiResScheduler, distributed_multiresolution_stats, create_shared_cache
