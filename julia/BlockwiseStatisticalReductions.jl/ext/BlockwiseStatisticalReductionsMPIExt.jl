module BlockwiseStatisticalReductionsMPIExt

# Distributed execution across MPI ranks (`MPIBackend{Inner}`). SPMD: every rank calls the same
# `reduce_stats(...; backend = MPIBackend(inner))` on the (full) input and ends up with the full
# result. Work is split over DISJOINT slabs of OUTPUT cells along the last dimension: because
# non-overlapping blockwise cells partition the input cleanly, each rank reduces its own cells with
# no cross-rank merge, then the per-rank accumulator slabs are combined with `Allgatherv!` (the
# accumulators are isbits, so `MPI.Datatype` is derived automatically). Coarsening runs locally on
# every rank from the gathered base result — exactly as the Distributed backend does. Each rank runs
# the given `inner` local backend (serial/threaded/GPU) on its slab, so `MPIBackend{ThreadedBackend}`
# etc. compose (MPI across nodes, threads within).
#
# Note: this v1 divides COMPUTE (all ranks hold the full input). Scattering the input to divide
# MEMORY across ranks is a compatible refinement (output-cell-aligned slabs need no halo).

using MPI: MPI
using BlockwiseStatisticalReductions: BlockwiseStatisticalReductions as BSR

# Contiguous near-equal cell counts for `n` cells over `k` ranks.
function _rank_counts(n::Int, k::Int)
    base, rem = divrem(n, k)
    return Int[base + (i <= rem ? 1 : 0) for i in 1:k]
end
# 1-based cell range owned by 0-based `rank`, given per-rank `counts`.
function _rank_range(counts::Vector{Int}, rank::Int)
    off = 0
    for r in 0:(rank - 1)
        off += counts[r + 1]
    end
    return (off + 1):(off + counts[rank + 1])
end

function _mpi_base!(out::Array{Acc,N}, inputs::Tuple, window::NTuple{N,Int},
                    inner::BSR.AbstractExecutionBackend, comm, nr::Int, rank::Int) where {Acc,N}
    sd = N                                    # last dim: contiguous slabs on a column-major array
    ncell = size(out, sd)
    if nr == 1 || ncell < nr
        # Fewer output cells than ranks (or serial): every rank has the full input, so just compute
        # the whole node locally (identical on all ranks) — cheap, and keeps the result complete.
        BSR.blockreduce!(out, inputs, window, inner)
        return out
    end
    counts = _rank_counts(ncell, nr)
    myr = _rank_range(counts, rank)
    dlo = (first(myr) - 1) * window[sd] + 1
    dhi = last(myr) * window[sd]
    islab = map(a -> view(a, ntuple(d -> d == sd ? (dlo:dhi) : Colon(), Val(N))...), inputs)
    oshape = ntuple(d -> d == sd ? length(myr) : size(out, d), Val(N))
    myout = BSR.blockreduce!(BSR.allocate_accumulators(Acc, oshape), islab, window, inner)
    # Gather the contiguous per-rank slabs into `out` (last-dim slabs ↔ contiguous linear chunks).
    cellsz = prod(ntuple(d -> d == sd ? 1 : size(out, d), Val(N)))
    recvcounts = counts .* cellsz
    MPI.Allgatherv!(vec(myout), MPI.VBuffer(vec(out), recvcounts), comm)
    return out
end

function BSR.run!(buf::BSR.TowerBuffers{Acc,N}, plan::BSR.ReductionPlan{N}, inputs::Tuple,
                 backend::BSR.MPIBackend) where {Acc,N}
    @boundscheck length(buf.arrays) == length(plan.steps) ||
        throw(DimensionMismatch("buffers do not match plan"))
    MPI.Initialized() || MPI.Init()
    comm = MPI.COMM_WORLD
    nr = MPI.Comm_size(comm)
    rank = MPI.Comm_rank(comm)
    inner = BSR.local_backend(backend)
    for i in eachindex(plan.steps)
        s = plan.steps[i]
        out = buf.arrays[i]
        if s.source == 0
            _mpi_base!(out, inputs, s.window, inner, comm, nr, rank)
        else
            BSR.coarsen!(out, buf.arrays[s.source], s.window, inner)   # local; parent is full on every rank
        end
    end
    return buf
end

end # module
