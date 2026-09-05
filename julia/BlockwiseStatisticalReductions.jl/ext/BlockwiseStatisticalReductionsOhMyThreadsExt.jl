module BlockwiseStatisticalReductionsOhMyThreadsExt

# Multithreaded CPU kernels. Every kernel writes disjoint output cells, so threading only changes which
# thread evaluates a cell, never the order of any merge — threaded results are bit-for-bit the serial
# ones. Work is split into consecutive index chunks so each task walks memory forwards.

using OhMyThreads: OhMyThreads as OMT
using ComputationalBackends: ComputationalBackends as CB
using BlockwiseStatisticalReductions: BlockwiseStatisticalReductions as BSR

function __init__()
    BSR.THREADS_LOADED[] = true
    return nothing
end

"Chunks per thread. Above one so a chunk that runs long cannot hold a thread idle at the end."
const CHUNKS_PER_THREAD = 4

@inline _chunks(n::Integer) =
    OMT.index_chunks(1:Int(n); n = clamp(CHUNKS_PER_THREAD * Threads.nthreads(), 1, Int(n)))
# The chunks are the unit of work, so the scheduler must not chunk them again.
@inline _scheduler() = OMT.DynamicScheduler(; chunking = false)

BSR.boxfold!(out::BSR.AccumulatorArray{A,N}, src, w::BSR.Window{N}, b::CB.AbstractThreadedBackend) where {A,N} =
    BSR.boxfold!(out, src, w, Val(BSR.static_shape(w)), b)
function BSR.boxfold!(out::BSR.AccumulatorArray{A,N}, src, w::BSR.Window{N}, ::Val{S},
                      ::CB.AbstractThreadedBackend) where {A,N,S}
    source = BSR._source(A, src)
    BSR._check_boxfold(out, source, w)
    BSR._check_shape(w, S)
    cells = CartesianIndices(size(out))
    isempty(cells) && return out
    OMT.tforeach(_chunks(length(cells)); scheduler = _scheduler()) do rng
        @inbounds for k in rng
            I = cells[k]
            out[I] = BSR.combine(A, BSR.Box(source, BSR._box_ranges(w, I, Val(S))))
        end
    end
    return out
end

function BSR.compose!(out::BSR.AccumulatorArray{A,N}, a::BSR.AccumulatorArray{A,N}, b::BSR.AccumulatorArray{A,N},
                      axis::Int, amap::AbstractVector{Int}, bmap::AbstractVector{Int},
                      ::CB.AbstractThreadedBackend) where {A,N}
    BSR.check_compose(out, a, b, axis, amap, bmap)
    cells = CartesianIndices(size(out))
    isempty(cells) && return out
    OMT.tforeach(_chunks(length(cells)); scheduler = _scheduler()) do rng
        @inbounds for k in rng
            I = cells[k]
            out[I] = BSR.compose_cell(a, b, I, axis, amap, bmap)
        end
    end
    return out
end

# The scan carries a dependency along its own axis, so the lines are the parallel unit. Each task keeps
# its own stacks: the scratch passed in belongs to the caller and cannot be shared across tasks.
function BSR.scan!(out::BSR.AccumulatorArray{A,N}, src::BSR.AccumulatorArray{A,N}, axis::Int, size::Int,
                   partial::Bool, scratch::BSR.ScanScratch{A}, ::CB.AbstractThreadedBackend) where {A,N}
    lines = BSR.scan_lines(out, src, axis, size, partial, scratch)
    isempty(lines) && return out
    local_scratch = OMT.TaskLocalValue{typeof(scratch)}(() -> BSR.ScanScratch(A, size))
    OMT.tforeach(_chunks(length(lines)); scheduler = _scheduler()) do rng
        s = local_scratch[]
        @inbounds for k in rng
            BSR._scan_line!(out, src, axis, lines[k], size, partial, s)
        end
    end
    return out
end

function BSR.finalize!(dst::AbstractArray{Tout,N}, src::BSR.AccumulatorArray{A,N}, tag::BSR.AbstractStatistic,
                       shifts::Tuple, ::CB.AbstractThreadedBackend) where {Tout,A,N}
    size(dst) == size(src) || throw(DimensionMismatch("destination $(size(dst)) does not match accumulators $(size(src))"))
    isempty(dst) && return dst
    OMT.tforeach(_chunks(length(dst)); scheduler = _scheduler()) do rng
        @inbounds for i in rng
            dst[i] = BSR.finalize(tag, BSR.unshift(src[i], shifts), Tout)
        end
    end
    return dst
end

# Threads raise the rates the planner divides by: more aggregate bandwidth, and merges (including the
# scan's, whose lines are independent) spread across cores. `min_cells` keeps the planner from creating
# nodes too small to fill the threads. Calibrated against the threaded gates.
BSR.kernel_limits(::CB.AbstractThreadedBackend, N::Int) =
    (serial = BSR.kernel_limits(CB.SerialBackend(), N);
     t = Threads.nthreads();
     BSR.KernelLimits(max(serial.min_cells, 64 * t), serial.max_tile_elements,
                      serial.bandwidth * min(t, 6), serial.merge_rate * t, serial.serial_merge_rate * t))

end
