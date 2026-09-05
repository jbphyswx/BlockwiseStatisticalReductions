module BlockwiseStatisticalReductionsMPIExt

# SPMD requests over a tensor no rank holds whole. Every rank owns a contiguous slab along one axis and
# the windows whose first cell falls in it, so a window only ever reaches forward and one halo, on one
# side, is enough. What travels is raw input cells rather than accumulators: an accumulator is wider than
# an input element, and the cells a straddling window is missing are the same cells either way.

using MPI: MPI
using ComputationalBackends: ComputationalBackends as CB
using BlockwiseStatisticalReductions: BlockwiseStatisticalReductions as BSR

_comm(b::CB.AbstractMPIBackend) = b.comm === nothing ? MPI.COMM_WORLD : b.comm

# One peer's share of the halo: which cells along the partitioned axis, and a contiguous buffer per
# travelling field. Ranges are already local — into the sender's slab, or into the receiver's staging.
struct Peer{B<:Tuple}
    rank::Int
    range::UnitRange{Int}
    buffers::B
end

"The halo one rank exchanges: what it sends, what it receives, and the part it already holds itself."
struct Exchange{B<:Tuple}
    comm::MPI.Comm
    axis::Int
    stage::B
    sends::Vector{Peer{B}}
    recvs::Vector{Peer{B}}
    from::UnitRange{Int}
    into::UnitRange{Int}
end

struct MPIPrepared{S<:BSR.SlabRequest,X<:Exchange,W}
    request::S
    exchange::X
    weights::W
    slab::BSR.Slab
end

BSR.explain(io::IO, mp::MPIPrepared; kw...) = BSR.explain(io, mp.request.interior; kw...)
"The windows this rank owns, in global coordinates."
BSR.windows(mp::MPIPrepared) = BSR.windows(mp.request.result)

_buffers(fields::Tuple, axis::Int, len::Int) = map(f -> BSR.axis_similar(f, axis, len), fields)

# Fields that travel: the data, plus per-element weights, which are partitioned like the data. Per-axis
# weight factors index the global axis instead, so every rank slices its own share out of the same vector.
_travelling(fields::Tuple, weights::AbstractArray) = (fields..., weights)
_travelling(fields::Tuple, weights) = fields

_slice_weights(w::AbstractArray, axis, r, N, dimnames) = w
_slice_weights(::Nothing, axis, r, N, dimnames) = nothing
function _slice_weights(w::Union{Tuple,NamedTuple}, axis, r, ::Val{N}, dimnames) where {N}
    f = BSR.weight_factors(w, Val(N), dimnames)
    return ntuple(d -> d == axis ? (f[d] === nothing ? nothing : f[d][r]) : f[d], Val(N))
end

function BSR.prepare_partitioned(pf::BSR.Partitioned, scales, backend::CB.AbstractMPIBackend;
                                 weights = nothing, dimnames = nothing, spacing = nothing,
                                 edge::BSR.EdgePolicy = BSR.Truncate(), kw...)
    MPI.Initialized() ||
        throw(ArgumentError("MPI is not initialized; call MPI.Init() before preparing a partitioned request"))
    comm = _comm(backend)
    axis = pf.axis
    fields = BSR._fieldtuple(pf.fields)
    BSR._check_fields(fields)
    local_shape = size(fields[1])
    N = length(local_shape)
    axis <= N || throw(ArgumentError("partition axis $axis for a $N-dimensional field"))
    BSR._check_dimnames(dimnames, N)

    extents = MPI.Allgather(local_shape[axis], comm)
    me = MPI.Comm_rank(comm) + 1
    offsets = [sum(view(extents, 1:r-1)) for r in eachindex(extents)]
    global_shape = ntuple(d -> d == axis ? sum(extents) : local_shape[d], N)
    slab = BSR.Slab(axis, offsets[me], extents[me], global_shape[axis])
    targets = BSR.resolve(scales, global_shape; edge = edge, dimnames = dimnames, spacing = spacing)
    part = BSR.partition(targets, slab)

    # Every rank publishes the cells its boundary windows read, so each also knows what to send.
    los = MPI.Allgather(first(part.staging), comm)
    his = MPI.Allgather(last(part.staging) + 1, comm)
    mine = (offsets[me] + 1):(offsets[me] + extents[me])
    travel = _travelling(fields, weights)
    stage = _buffers(travel, axis, length(part.staging))

    recvs = Peer{typeof(stage)}[]
    sends = Peer{typeof(stage)}[]
    for r in eachindex(extents)
        r == me && continue
        theirs = (offsets[r] + 1):(offsets[r] + extents[r])
        got = intersect((los[me] + 1):his[me], theirs)
        isempty(got) || push!(recvs, Peer(r - 1, got .- los[me], _buffers(travel, axis, length(got))))
        give = intersect((los[r] + 1):his[r], mine)
        isempty(give) || push!(sends, Peer(r - 1, give .- offsets[me], _buffers(travel, axis, length(give))))
    end
    held = intersect((los[me] + 1):his[me], mine)
    exchange = Exchange(comm, axis, stage, sends, recvs, held .- offsets[me], held .- los[me])

    staged = (los[me] + 1):his[me]
    staging_fields = BSR.like_fields(pf.fields, ntuple(k -> stage[k], length(fields)))
    request = BSR.slab_request(part, pf.fields, staging_fields, slab, global_shape;
                               weights = _slice_weights(weights, axis, mine, Val(N), dimnames),
                               stageweights = weights isa AbstractArray ? stage[end] :
                                              _slice_weights(weights, axis, staged, Val(N), dimnames),
                               dimnames = dimnames, spacing = spacing, edge = edge, kw...)
    return MPIPrepared(request, exchange, weights, slab)
end

# The inner plans check the shape and the field count; the exchange only needs the fields to line up with
# the buffers it was built for.
function BSR.blockstats!(mp::MPIPrepared, fields)
    travel = _travelling(BSR._fieldtuple(fields), mp.weights)
    length(travel) == length(mp.exchange.stage) ||
        throw(ArgumentError("$(length(travel)) field(s) given for a request prepared with $(length(mp.exchange.stage))"))
    _exchange!(mp.exchange, travel)
    return BSR.run_slab!(mp.request, fields)
end
BSR.blockstats!(mp::MPIPrepared, pf::BSR.Partitioned) = BSR.blockstats!(mp, pf.fields)

function _exchange!(x::Exchange, travel::Tuple)
    reqs = MPI.Request[]
    for p in x.recvs, (k, b) in enumerate(p.buffers)
        push!(reqs, MPI.Irecv!(b, x.comm; source = p.rank, tag = k))
    end
    for p in x.sends, (k, b) in enumerate(p.buffers)
        copyto!(b, BSR.axis_slice(travel[k], x.axis, p.range))
        push!(reqs, MPI.Isend(b, x.comm; dest = p.rank, tag = k))
    end
    for (k, f) in enumerate(travel)
        copyto!(BSR.axis_slice(x.stage[k], x.axis, x.into), BSR.axis_slice(f, x.axis, x.from))
    end
    MPI.Waitall(reqs)
    for p in x.recvs, (k, b) in enumerate(p.buffers)
        copyto!(BSR.axis_slice(x.stage[k], x.axis, p.range), b)
    end
    return nothing
end

end
