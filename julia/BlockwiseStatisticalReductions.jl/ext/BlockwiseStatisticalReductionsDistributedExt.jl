module BlockwiseStatisticalReductionsDistributedExt

"""
    BlockwiseStatisticalReductionsDistributedExt — distributed execution (`DistributedBackend{Inner}`)

Distributes the expensive base pass across worker processes and runs the (cheap) coarsening tower
locally. The base pass is partitioned over DISJOINT slabs of output cells: each output cell reduces
its own block of the input, so slabs never straddle and each is computed independently and
identically to the serial result — no cross-worker merge is needed and the distributed result is
bit-for-bit equal to serial. (When a single coarse block must itself be split across workers, the
exact Chan/Pebay `merge` would be used; that full-domain case is left to the local coarsening here.)
Each worker runs the package's `inner` local backend.
"""

using Distributed: Distributed, workers, nworkers, remotecall_fetch
using SharedArrays: SharedArrays
using BlockwiseStatisticalReductions: BlockwiseStatisticalReductions as BSR

# Split 1:n into up to `k` contiguous, near-equal ranges.
function _chunk_ranges(n::Int, k::Int)
    k = clamp(k, 1, n)
    base, rem = divrem(n, k)
    ranges = UnitRange{Int}[]
    start = 1
    for i in 1:k
        len = base + (i <= rem ? 1 : 0)
        push!(ranges, start:(start + len - 1))
        start += len
    end
    return ranges
end

# Distribute one base node over disjoint output-cell slabs along the largest output dimension.
function _distributed_base!(out::Array{Acc,N}, inputs::Tuple, window::NTuple{N,Int}) where {Acc,N}
    sd = argmax(size(out))
    wkrs = workers()
    ranges = _chunk_ranges(size(out, sd), length(wkrs))
    slabs = Vector{Array{Acc,N}}(undef, length(ranges))
    @sync for k in eachindex(ranges)
        rng = ranges[k]
        w = wkrs[k]
        dlo = (first(rng) - 1) * window[sd] + 1
        dhi = last(rng) * window[sd]
        islab = map(a -> copy(selectdim(a, sd, dlo:dhi)), inputs)
        @async slabs[k] = remotecall_fetch(BSR._compute_base_slab, w, Acc, islab, window)
    end
    for k in eachindex(ranges)
        copyto!(selectdim(out, sd, ranges[k]), slabs[k])
    end
    return out
end

function BSR.run!(buf::BSR.TowerBuffers{Acc,N}, plan::BSR.ReductionPlan{N}, inputs::Tuple,
                 backend::BSR.DistributedBackend) where {Acc,N}
    @boundscheck length(buf.arrays) == length(plan.steps) ||
        throw(DimensionMismatch("buffers do not match plan"))
    inner = BSR.local_backend(backend)
    can_distribute = nworkers() > 1
    for i in eachindex(plan.steps)
        s = plan.steps[i]
        out = buf.arrays[i]
        if s.source == 0
            if can_distribute && maximum(size(out)) >= 2
                _distributed_base!(out, inputs, s.window)
            else
                BSR.blockreduce!(out, inputs, s.window, inner)
            end
        else
            BSR.coarsen!(out, buf.arrays[s.source], s.window, inner)
        end
    end
    return buf
end

end # module
