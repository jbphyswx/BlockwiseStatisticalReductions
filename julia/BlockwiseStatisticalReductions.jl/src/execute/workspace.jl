# Prepared kernel invocations: one per computed node, executed in plan order.
struct BaseStep{O,W,S}
    out::O
    window::W
    shape::Val{S}
end
struct CoarsenStep{O,P,W,S}
    out::O
    parent::P
    grid::W
    shape::Val{S}
end
struct ComposeStep{O,P,Q,AM,BM}
    out::O
    a::P
    b::Q
    axis::Int
    amap::AM
    bmap::BM
end
struct ScanStep{O,P,SC}
    out::O
    parent::P
    axis::Int
    size::Int
    partial::Bool
    scratch::SC
end

"""
    Workspace{N}

Storage for a plan: one `AccumulatorArray` per buffer, a storage handle per node (views for restrides),
and the prepared steps `run!` executes.
"""
struct Workspace{N,S<:Tuple}
    buffers::Vector{Any}
    storage::Vector{Any}
    steps::S
end

"Storage of node `k` (an `AccumulatorArray`, possibly a view)."
node_storage(ws::Workspace, k::Int) = ws.storage[k]

# Zero-copy view of a storage over per-axis index selections.
function subarray(aa::AccumulatorArray{A,N}, sel::NTuple{N,AbstractVector{Int}}) where {A,N}
    comps = _view_components(aa.components, sel)
    return AccumulatorArray{A,N,typeof(comps)}(comps, map(length, sel))
end
_view_components(nt::NamedTuple, sel) = map(c -> _view_components(c, sel), nt)
_view_components(c::AbstractArray, sel) = view(c, sel...)
_view_components(c::Uniform, sel) = c

"""
    allocate(plan, ::Type{A}, prototype::AbstractArray) -> Workspace

Allocate the plan's buffers for accumulator type `A` where `prototype` lives, build the restride views,
and prepare one step per computed node.
"""
function allocate(p::Plan{N}, ::Type{A}, prototype::AbstractArray) where {N,A<:AbstractAccumulator}
    nbuf = maximum(p.buffer; init = 0)
    buffers = Vector{Any}(undef, nbuf)
    owner = Dict{Int,Int}()
    for k in p.order
        b = p.buffer[k]
        haskey(owner, b) && continue
        owner[b] = k
        n = p.nodes[k]
        uniform = uniform_count(n) ? (n = volume(n.window),) : (;)
        buffers[b] = AccumulatorArray(A, prototype, n.shape; uniform)
    end
    storage = Vector{Any}(undef, length(p.nodes))
    for k in p.order
        storage[k] = buffers[p.buffer[k]]
    end
    for k in eachindex(p.nodes)
        d = p.how[k]
        is_view(d) || continue
        storage[k] = subarray(storage[d.parent], d.sel)
    end
    steps = Tuple(_step(p, k, storage, prototype) for k in p.order)
    return Workspace{N,typeof(steps)}(buffers, storage, steps)
end

# Index data a kernel dereferences must live where the kernel runs. Ranges are isbits and travel as
# they are; an explicit list is copied into the storage's array type once, here, rather than per call.
_like(proto::AbstractArray, v::AbstractRange) = v
_like(proto::AbstractArray, v::AbstractVector) = copyto!(similar(proto, eltype(v), size(v)), v)
_like(proto::AbstractArray, p::Progression) = p
_like(proto::AbstractArray, p::Origins) = (v = _like(proto, p.origins); Origins{typeof(v)}(v))
_like(proto::AbstractArray, aw::AxisWindow) = AxisWindow(aw.extent, aw.size, _like(proto, aw.pos), aw.partial)
_like(proto::AbstractArray, w::Tuple) = map(x -> _like(proto, x), w)

# One component array of a node's storage, or `nothing` when every component is uniform (a count-only
# request stores no arrays at all).
_prototype(aa::AccumulatorArray) = _find_array(aa.components)
_find_array(c::AbstractArray) = c
_find_array(::Uniform) = nothing
function _find_array(nt::NamedTuple)
    for v in values(nt)
        a = _find_array(v)
        a === nothing || return a
    end
    return nothing
end

function _step(p::Plan{N}, k::Int, storage, prototype::AbstractArray) where {N}
    d, n = p.how[k], p.nodes[k]
    proto = prototype
    if d isa Base_
        return BaseStep(storage[k], _like(proto, n.window), Val(static_shape(n.window)))
    elseif d isa Coarsen
        parent = p.nodes[d.parent]
        grid = ntuple(a -> tiled(parent.shape[a], d.k[a], n.window[a].partial ? Partial() : Truncate()), Val(N))
        return CoarsenStep(storage[k], storage[d.parent], _like(proto, grid), Val(static_shape(grid)))
    elseif d isa Compose
        return ComposeStep(storage[k], storage[d.a], storage[d.b], d.axis, _like(proto, d.amap), _like(proto, d.bmap))
    elseif d isa Scan
        A = eltype(storage[k])
        return ScanStep(storage[k], storage[d.parent], d.axis, d.size, d.partial, ScanScratch(A, d.size))
    end
    error("internal: no step for $(typeof(d))")
end
