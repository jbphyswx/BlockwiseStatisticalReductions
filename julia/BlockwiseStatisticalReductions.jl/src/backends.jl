"""
    execute_parallel(backend::CPUBackend, f, items)

Execute function `f` over `items` using CPU backend.
"""
function execute_parallel(backend::CPUBackend, f, items)
    if backend.nthreads == 1
        return map(f, items)
    else
        return Base.Threads.@threads :static for i in eachindex(items)
            f(items[i])
        end
    end
end

"""
    execute_mapreduce(backend::CPUBackend, f, op, items; init)

Map-reduce execution using CPU backend.
"""
function execute_mapreduce(backend::CPUBackend, f, op, items; init)
    if backend.nthreads == 1
        return mapreduce(f, op, items; init=init)
    else
        return Base.ThreadsX.mapreduce(f, op, items; init=init)
    end
end

"""
    partition_items(backend::DistributedBackend, items)

Partition items across distributed workers.
"""
function partition_items(backend::DistributedBackend, items)
    n_procs = length(backend.procs)
    n_items = length(items)
    chunk_size = ceil(Int, n_items / n_procs)
    
    partitions = []
    for i in 1:n_procs
        start_idx = (i-1) * chunk_size + 1
        end_idx = min(i * chunk_size, n_items)
        push!(partitions, items[start_idx:end_idx])
    end
    
    return partitions
end

"""
    best_backend(arr::AbstractArray)

Heuristic to select the best backend for an array type.
"""
best_backend(arr::AbstractArray) = CPUBackend()

# GPU backend will be extended in CUDAExt
best_backend(arr::Type) = CPUBackend()
