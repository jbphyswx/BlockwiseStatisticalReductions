module BlockwiseStatisticalReductionsOhMyThreadsExt

"""
    BlockwiseStatisticalReductionsOhMyThreadsExt — multithreaded execution (`ThreadedBackend`)

Parallelizes the base reduction and the cross-scale merge over output cells with OhMyThreads
`tforeach` (which chunks the cells — not one task per cell). Each output cell is computed by the very
same per-cell `reduce_block`/`coarsen_block` the serial path uses, so threading inherits the
specialized bulk kernels for free and, because cells are independent (disjoint writes), the result is
bit-for-bit identical to serial. `ThreadedBackend` always threads; choosing serial vs threaded by
problem size is `AutoBackend`'s responsibility, not this backend's.
"""

using OhMyThreads: OhMyThreads as OMT
using BlockwiseStatisticalReductions: BlockwiseStatisticalReductions as BSR

function __init__()
    BSR._THREADING_AVAILABLE[] = true
    return nothing
end

# Per-call scheduler: honor an explicit task count, else let OhMyThreads pick (chunks over threads).
@inline _scheduler(b::BSR.ThreadedBackend) =
    b.ntasks > 0 ? OMT.DynamicScheduler(; ntasks = b.ntasks) : OMT.DynamicScheduler()

function BSR._drive_base!(out::AbstractArray{Acc,N}, inputs::Tuple, window::NTuple{N,Int},
                         b::BSR.ThreadedBackend) where {Acc,N}
    OMT.tforeach(CartesianIndices(out); scheduler = _scheduler(b)) do I
        @inbounds out[I] = BSR.reduce_block(Acc, inputs, BSR._block_lo(I, window), BSR._block_hi(I, window))
    end
    return nothing
end

function BSR._drive_merge!(out::AbstractArray{Acc,N}, fine::AbstractArray{Acc,N}, window::NTuple{N,Int},
                          b::BSR.ThreadedBackend) where {Acc,N}
    OMT.tforeach(CartesianIndices(out); scheduler = _scheduler(b)) do I
        @inbounds out[I] = BSR.coarsen_block(fine, BSR._block_lo(I, window), BSR._block_hi(I, window))
    end
    return nothing
end

end # module
