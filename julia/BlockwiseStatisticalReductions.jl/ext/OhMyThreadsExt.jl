module OhMyThreadsExt

using BlockwiseStatisticalReductions
using OhMyThreads: OhMyThreads

"""
    OhMyThreadsBackend

OhMyThreads-based parallel execution backend.
"""
struct OhMyThreadsBackend <: BlockwiseStatisticalReductions.AbstractExecutionBackend
    scheduler::Symbol
    nthreads::Int
end

OhMyThreadsBackend(; scheduler=:dynamic, nthreads=Base.Threads.nthreads()) = 
    OhMyThreadsBackend(scheduler, nthreads)

"""
    BlockwiseStatisticalReductions.execute_parallel(backend::OhMyThreadsBackend, f, items)

Execute function `f` over `items` using OhMyThreads.
"""
function BlockwiseStatisticalReductions.execute_parallel(backend::OhMyThreadsBackend, f, items)
    return OhMyThreads.tmap(f, items; scheduler=backend.scheduler, ntasks=backend.nthreads)
end

"""
    BlockwiseStatisticalReductions.execute_mapreduce(backend::OhMyThreadsBackend, f, op, items; init)

Map-reduce execution using OhMyThreads.
"""
function BlockwiseStatisticalReductions.execute_mapreduce(backend::OhMyThreadsBackend, f, op, items; init)
    return OhMyThreads.tmapreduce(f, op, items; 
                      scheduler=backend.scheduler, 
                      ntasks=backend.nthreads,
                      init=init)
end

"""
    BlockwiseStatisticalReductions.tree_reduce_impl(items::Vector{T}, op, backend::OhMyThreadsBackend) where T

Tree reduction using OhMyThreads.
"""
function BlockwiseStatisticalReductions.tree_reduce_impl(items::Vector{T}, op, backend::OhMyThreadsBackend) where T
    if length(items) == 0
        return nothing
    elseif length(items) == 1
        return items[1]
    end
    
    # Use OhMyThreads for tree reduction
    # First level: parallel reduction of pairs
    while length(items) > 1
        half_len = div(length(items), 2)
        is_odd = isodd(length(items))
        
        new_items = Vector{T}(undef, half_len + (is_odd ? 1 : 0))
        
        # Parallel reduction of pairs
        OhMyThreads.tmap!(new_items, 1:half_len; scheduler=backend.scheduler, ntasks=backend.nthreads) do i
            op(items[2*i - 1], items[2*i])
        end
        
        # Handle odd element
        if is_odd
            new_items[end] = items[end]
        end
        
        items = new_items
    end
    
    return items[1]
end

end
