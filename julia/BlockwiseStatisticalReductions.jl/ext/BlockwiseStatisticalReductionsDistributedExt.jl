module BlockwiseStatisticalReductionsDistributedExt

# One process owns the tensor and hands each worker a contiguous slab of the last axis, plus the few cells
# past it that its straddling windows read. A worker holds its buffers and its plan between calls, so what
# crosses a process boundary is the slab's data on the way out and its share of the results on the way
# back — never a plan, and never the whole tensor at once.

using Distributed: Distributed
using ComputationalBackends: ComputationalBackends as CB
using BlockwiseStatisticalReductions: BlockwiseStatisticalReductions as BSR

# Prepared slab requests on this worker, keyed by the owner and the id it handed out.
const SLABS = Dict{Tuple{Int,Int},Any}()
const NEXT_ID = Ref(0)

"Answers on a worker that has this extension loaded, so the owner can say so plainly when one does not."
worker_ready() = true

"One worker's share: the rows it computes from, the rows past them its windows read, and where its windows land in each result."
struct Share
    worker::Int
    rows::UnitRange{Int}
    stagerows::UnitRange{Int}
    out::Vector{UnitRange{Int}}
end

struct DistributedPrepared{R<:BSR.ScaleResults}
    id::Int
    axis::Int
    shares::Vector{Share}
    result::R
    input_shape::Tuple{Vararg{Int}}
end

BSR.windows(dp::DistributedPrepared) = BSR.windows(dp.result)
BSR.explain(io::IO, dp::DistributedPrepared; kw...) = BSR.explain(io, dp.result; kw...)

_specs(fields::Tuple, axis::Int, len::Int) =
    map(f -> (eltype(f), ntuple(d -> d == axis ? len : size(f, d), ndims(f))), fields)

# Runs on the worker: allocate this slab's buffers, build its half of the request, and report the result
# names and element types so the owner can allocate the whole ones.
function setup!(key::Tuple{Int,Int}, names::Tuple, fieldspecs, stagespecs, slab::BSR.Slab, targets,
                global_shape, weights, stageweights, kw)
    fields = NamedTuple{names}(map(s -> Array{s[1]}(undef, s[2]), fieldspecs))
    stage = NamedTuple{names}(map(s -> Array{s[1]}(undef, s[2]), stagespecs))
    part = BSR.partition(targets, slab)
    request = BSR.slab_request(part, fields, stage, slab, global_shape;
                               weights = weights, stageweights = stageweights, kw...)
    SLABS[key] = (request, values(fields), values(stage))
    return (BSR.statnames(request.result), map(eltype, values(request.result[1])))
end

# Runs on the worker: take this call's cells, run both plans, hand back plain arrays.
function run!(key::Tuple{Int,Int}, data::Tuple, stagedata::Tuple)
    request, fields, stage = SLABS[key]
    foreach(copyto!, fields, data)
    foreach(copyto!, stage, stagedata)
    r = BSR.run_slab!(request, fields)
    return [map(Array, r[i]) for i in 1:length(BSR.windows(r))]
end

"Runs on the worker: drop a request's buffers and plan."
forget!(key::Tuple{Int,Int}) = (delete!(SLABS, key); nothing)

function BSR.prepare_on(fields, scales, backend::CB.AbstractDistributedBackend;
                        stats::Union{Tuple,NamedTuple}, edge::BSR.EdgePolicy = BSR.Truncate(),
                        weights = nothing, dimnames = nothing, spacing = nothing, kw...)
    fs = BSR._fieldtuple(fields)
    BSR._check_fields(fs)
    names = BSR._fieldnames(fields)
    shape = size(fs[1])
    N = length(shape)
    axis = N          # contiguous slabs: a slab of the last axis is one run of memory per field
    BSR._check_dimnames(dimnames, N)
    targets = BSR.resolve(scales, shape; edge = edge, dimnames = dimnames, spacing = spacing)
    workers = Distributed.workers()
    (isempty(workers) || workers == [Distributed.myid()]) &&
        throw(ArgumentError("a distributed request needs worker processes; start Julia with -p or call `addprocs`"))
    inner = CB.local_backend(backend)
    key_id = (NEXT_ID[] += 1)

    shares = Share[]
    cursor = ones(Int, length(targets))
    statnames, eltypes = (), ()
    for (k, (lo, len)) in enumerate(_slabs(shape[axis], length(workers)))
        worker = workers[k]
        slab = BSR.Slab(axis, lo, len, shape[axis])
        part = BSR.partition(targets, slab)
        rows = (lo + 1):(lo + len)
        stagerows = (first(part.staging) + 1):(last(part.staging) + 1)
        _check_worker(worker)
        statnames, eltypes = Distributed.remotecall_fetch(setup!, worker, (Distributed.myid(), key_id), names,
                                                          _specs(fs, axis, len), _specs(fs, axis, length(part.staging)),
                                                          slab, targets, shape,
                                                          _slice_weights(weights, axis, rows, Val(N), dimnames),
                                                          _slice_weights(weights, axis, stagerows, Val(N), dimnames),
                                                          (; stats = stats, edge = edge, backend = inner, kw...))
        # Windows belong to the slab their first cell falls in, so the shares tile each result in order.
        out = UnitRange{Int}[]
        for (i, w) in enumerate(part.owned)
            n = BSR.shape(w)[axis]
            push!(out, cursor[i]:(cursor[i] + n - 1))
            cursor[i] += n
        end
        push!(shares, Share(worker, rows, stagerows, out))
    end
    full = [NamedTuple{statnames}(ntuple(j -> Array{eltypes[j]}(undef, BSR.shape(w)), length(statnames)))
            for w in targets]
    for (i, w) in enumerate(targets)
        cursor[i] == BSR.shape(w)[axis] + 1 ||
            error("internal: the slabs own $(cursor[i] - 1) of the $(BSR.shape(w)[axis]) windows of $(map(aw -> aw.size, w))")
    end
    plan = BSR.plan(shape, targets; backend = inner)
    return DistributedPrepared(key_id, axis, shares,
                               BSR.ScaleResults(shape, targets, full, plan, statnames, dimnames, spacing), shape)
end

# Contiguous slabs of `extent` over at most `n` workers, the remainder going to the last.
function _slabs(extent::Int, n::Int)
    n = max(1, min(n, max(extent, 1)))
    base = extent ÷ n
    out = Tuple{Int,Int}[]
    lo = 0
    for r in 1:n
        len = r < n ? base : extent - lo
        push!(out, (lo, len))
        lo += len
    end
    return out
end

function _check_worker(w::Int)
    ok = try
        Distributed.remotecall_fetch(worker_ready, w)
    catch
        false
    end
    ok || throw(ArgumentError("worker $w cannot serve this request; every worker needs `@everywhere using BlockwiseStatisticalReductions`"))
    return nothing
end

_slice_weights(::Nothing, axis, r, ::Val, dimnames) = nothing
_slice_weights(w::AbstractArray, axis, r, ::Val, dimnames) = collect(BSR.axis_slice(w, axis, r))
function _slice_weights(w::Union{Tuple,NamedTuple}, axis, r, ::Val{N}, dimnames) where {N}
    f = BSR.weight_factors(w, Val(N), dimnames)
    return ntuple(d -> d == axis ? (f[d] === nothing ? nothing : f[d][r]) : f[d], Val(N))
end

function BSR.blockstats!(dp::DistributedPrepared, fields)
    fs = BSR._fieldtuple(fields)
    BSR._check_fields(fs)
    size(fs[1]) == dp.input_shape ||
        throw(DimensionMismatch("input shape $(size(fs[1])) does not match the prepared shape $(dp.input_shape)"))
    key = (Distributed.myid(), dp.id)
    pending = map(dp.shares) do share
        Distributed.remotecall(run!, share.worker, key,
                               map(f -> collect(BSR.axis_slice(f, dp.axis, share.rows)), fs),
                               map(f -> collect(BSR.axis_slice(f, dp.axis, share.stagerows)), fs))
    end
    for (share, task) in zip(dp.shares, pending)
        for (i, nt) in enumerate(fetch(task)), name in BSR.statnames(dp.result)
            copyto!(BSR.axis_slice(dp.result.results[i][name], dp.axis, share.out[i]), nt[name])
        end
    end
    return dp.result
end

BSR.release!(dp::DistributedPrepared) =
    (foreach(s -> Distributed.remotecall_wait(forget!, s.worker, (Distributed.myid(), dp.id)), dp.shares); nothing)

end
