module BlockwiseStatisticalReductionsOhMyThreadsExt

"""
    BlockwiseStatisticalReductionsOhMyThreadsExt — multithreaded execution (`ThreadedBackend`)

Parallelizes the per-output-cell loop of both the base reduction and the cross-scale merge with
OhMyThreads. Output cells are written to disjoint locations, so there is no contention and the
threaded result is identical to the serial one (merge is associative + commutative). Implemented by
adding `ThreadedBackend` methods to the package's internal cell drivers `_drive_base!`/`_drive_merge!`.
"""

using OhMyThreads: OhMyThreads as OMT
using BlockwiseStatisticalReductions: BlockwiseStatisticalReductions as BSR

function __init__()
    BSR._THREADING_AVAILABLE[] = true
    return nothing
end

# Per-call scheduler: honor an explicit task count, else let OhMyThreads choose.
@inline _scheduler(b::BSR.ThreadedBackend) =
    b.ntasks > 0 ? OMT.DynamicScheduler(; ntasks = b.ntasks) : OMT.DynamicScheduler()

function BSR._drive_base!(out::AbstractArray{Acc,N}, leaf::F, window::NTuple{N,Int},
                         b::BSR.ThreadedBackend) where {Acc,N,F}
    OMT.tforeach(CartesianIndices(out); scheduler = _scheduler(b)) do I
        @inbounds out[I] = BSR.treefold(leaf, BSR._block_lo(I, window), BSR._block_hi(I, window))
    end
    return nothing
end

function BSR._drive_merge!(out::AbstractArray{Acc,N}, leaf::F, window::NTuple{N,Int},
                          b::BSR.ThreadedBackend) where {Acc,N,F}
    OMT.tforeach(CartesianIndices(out); scheduler = _scheduler(b)) do I
        @inbounds out[I] = BSR.treefold(leaf, BSR._block_lo(I, window), BSR._block_hi(I, window))
    end
    return nothing
end

end # module
