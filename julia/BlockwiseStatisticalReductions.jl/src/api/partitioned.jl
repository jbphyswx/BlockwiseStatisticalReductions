# A request over a tensor nobody holds in one piece. The windows a rank owns divide into those its own
# slab covers and those that reach into the next slab; the first are computed straight from the caller's
# array, the second from a small staging region the distributed backend fills. Both write into one set of
# result arrays, so the caller sees the ordinary `ScaleResults` for the windows this rank owns.

"""
    Partitioned(fields; axis)

One rank's share of a tensor split along `axis`: the global tensor is the concatenation of every rank's
`fields` along that axis, in rank order. `fields` is an array or a `NamedTuple`/`Tuple` of same-shaped
arrays, as [`prepare`](@ref) takes.
"""
struct Partitioned{F}
    fields::F
    axis::Int
end
Partitioned(fields; axis::Integer) = (axis >= 1 || throw(ArgumentError("axis must be ≥ 1, got $axis")); Partitioned(fields, Int(axis)))

prepare(pf::Partitioned, scales; backend::CB.AbstractExecutionBackend = CB.AutoBackend(), kw...) =
    prepare_partitioned(pf, scales, backend; kw...)

"""
    prepare_partitioned(::Partitioned, scales, backend; kwargs...)

Resolve a partitioned request. Defined by the backend extensions that know how to reach the other
shares of the tensor.
"""
prepare_partitioned(::Partitioned, scales, backend::CB.AbstractExecutionBackend; kw...) =
    throw(ArgumentError("a partitioned request needs a distributed backend, got $(typeof(backend))"))
prepare_partitioned(::Partitioned, scales, backend::Union{CB.AbstractMPIBackend,CB.AbstractDistributedBackend}; kw...) =
    missing_extension(backend)

blockstats(pf::Partitioned, scales; kwargs...) = blockstats!(prepare(pf, scales; kwargs...), pf.fields)

"""
    SlabRequest

A partitioned request as this rank runs it: the plan over its own slab, the plan over the staging region
its boundary windows read, one set of result arrays the two write disjoint slices of, and the presented
[`ScaleResults`](@ref) whose windows are in global coordinates.
"""
struct SlabRequest{P1,P2,R<:ScaleResults,S}
    interior::P1
    boundary::P2
    result::R
    stagefields::S
    slab::Slab
    staging::UnitRange{Int}
end

"The staging arrays the boundary plan reads; a distributed backend fills these before every run."
stagefields(sr::SlabRequest) = sr.stagefields

"""
    slab_request(part, fields, stagefields, slab, global_shape; weights, stageweights, dimnames, spacing, kwargs...)

Split a resolved request between a rank's own slab and its staging region, given the [`partition`](@ref)
the caller already computed to size the staging arrays. `fields` are the rank's local arrays and
`stagefields` the same fields over `part.staging`, both in the container form [`prepare`](@ref) takes;
the remaining keywords are `prepare`'s. Element weights are sliced like the fields, so `weights` covers
the slab and `stageweights` the staging region.
"""
function slab_request(part::NamedTuple, fields, stagefields, slab::Slab, global_shape::NTuple{N,Int};
                      weights = nothing, stageweights = nothing, dimnames = nothing, spacing = nothing,
                      kw...) where {N}
    # One cell per axis names the statistics and their output eltypes without allocating a result the size
    # of the request; neither depends on the geometry.
    local_shape = size(_fieldtuple(fields)[1])
    probe = prepare(fields, Resolved([ntuple(d -> AxisWindow(local_shape[d], 1, Progression(0, 1, min(local_shape[d], 1)), false), Val(N))]);
                    weights = weights, kw...)
    names = statnames(probe.result)
    eltypes = map(eltype, values(probe.result[1]))
    proto = _fieldtuple(fields)[1]
    full = [NamedTuple{names}(ntuple(k -> similar(proto, eltypes[k], shape(w)), length(names))) for w in part.owned]
    interior = prepare(fields, Resolved(part.interior); weights = weights,
                       into = _slices(full, part.split, slab.axis, Val(N), true), kw...)
    boundary = prepare(stagefields, Resolved(part.boundary); weights = stageweights,
                       into = _slices(full, part.split, slab.axis, Val(N), false), kw...)
    result = ScaleResults(global_shape, part.owned, full, interior.plan, names, dimnames, spacing)
    return SlabRequest(interior, boundary, result, stagefields, slab, part.staging)
end

# Views of the result arrays: the first `split` cells along the partitioned axis belong to the slab's own
# windows, the rest to the ones that reach past it.
function _slices(full, split::Vector{Int}, axis::Int, ::Val{N}, first_part::Bool) where {N}
    return [map(a -> view(a, ntuple(d -> d == axis ? (first_part ? (1:s) : (s+1:size(a, d))) : Base.Colon(), Val(N))...), nt)
            for (nt, s) in zip(full, split)]
end

"""
    run_slab!(sr::SlabRequest, fields) -> ScaleResults

Run both plans of a split request. [`stagefields`](@ref) must already hold the cells the boundary windows
read; filling them is the distributed backend's job.
"""
function run_slab!(sr::SlabRequest, fields)
    blockstats!(sr.interior, fields)
    blockstats!(sr.boundary, sr.stagefields)
    return sr.result
end
