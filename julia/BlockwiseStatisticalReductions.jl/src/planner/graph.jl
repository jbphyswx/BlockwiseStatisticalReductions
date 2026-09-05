"One window of a plan: its storage shape and whether the caller asked for it."
struct Node{N}
    window::Window{N}
    shape::NTuple{N,Int}
    requested::Bool
end
Node(w::Window{N}, requested::Bool) where {N} = Node{N}(w, shape(w), requested)
"`true` when every window of the node holds the same number of cells (its count can be stored once)."
uniform_count(n::Node) = uniform_length(n.window)
sizes(n::Node) = map(aw -> aw.size, n.window)
cells(n::Node) = prod(n.shape)

abstract type Derivation end
"Reduce raw fields inside every window."
struct Base_ <: Derivation end
"Merge `k` consecutive windows of a tiled parent along every axis."
struct Coarsen{N} <: Derivation
    parent::Int
    k::NTuple{N,Int}
end
"Merge parent `a` at the child origins with parent `b` at the origins shifted by `a`'s size along `axis`."
struct Compose <: Derivation
    a::Int
    b::Int
    axis::Int
    amap::AbstractVector{Int}
    bmap::AbstractVector{Int}
end
"Select a subset of the parent's windows along every axis (a view, no computation)."
struct Restride{N} <: Derivation
    parent::Int
    sel::NTuple{N,AbstractVector{Int}}
end
"Dense sliding windows of `size` along `axis` over the parent's cells."
struct Scan <: Derivation
    parent::Int
    axis::Int
    size::Int
    partial::Bool
end

parents(::Base_) = Int[]
parents(d::Coarsen) = Int[d.parent]
parents(d::Compose) = Int[d.a, d.b]
parents(d::Restride) = Int[d.parent]
parents(d::Scan) = Int[d.parent]
is_view(::Derivation) = false
is_view(::Restride) = true

"""
    Plan{N}

Nodes with one derivation each, an execution `order` (views excluded), a buffer id per node (`0` for
views), and the requested nodes as `outputs`, in target order.
"""
struct Plan{N}
    input_shape::NTuple{N,Int}
    nodes::Vector{Node{N}}
    how::Vector{Derivation}
    order::Vector{Int}
    buffer::Vector{Int}
    outputs::Vector{Int}
end

Base.length(p::Plan) = length(p.nodes)
"Indices of the nodes computed straight from the raw fields."
base_nodes(p::Plan) = [i for i in eachindex(p.how) if p.how[i] isa Base_]

# Index of every origin of `child` (shifted by `shift`) within the origins of `parent`; a range when both
# are progressions.
function index_map(parent::Positions, child::Positions, shift::Int)
    if parent isa Progression && child isa Progression
        child.count == 0 && return 1:0
        first = (child.offset + shift - parent.offset) ÷ parent.stride + 1
        step = child.stride ÷ parent.stride
        return range(first; step = max(step, 1), length = child.count)
    end
    po = origins(parent)
    return Int[searchsortedfirst(po, o + shift) for o in origins(child)]
end

# The window `k`-fold coarsening of `parent` produces, or `nothing` when it is not tiled.
_coarsened(parent::Window{N}, k::NTuple{N,Int}) where {N} =
    all(is_tiled, parent) ? ntuple(d -> coarsen_result(parent[d], k[d]), Val(N)) : nothing

"""
    check(plan) -> plan

Verify every derivation against the geometry predicates, that parents precede children in `order`,
and that every requested node has storage. Throws on the first violation.
"""
function check(p::Plan{N}) where {N}
    pos = Dict(i => k for (k, i) in enumerate(p.order))
    for (i, d) in enumerate(p.how)
        n = p.nodes[i]
        for q in parents(d)
            1 <= q <= length(p.nodes) || error("node $i: parent $q out of range")
            q != i || error("node $i derives from itself")
            if !is_view(d)
                haskey(pos, i) || error("node $i is computed but not scheduled")
                is_view(p.how[q]) || haskey(pos, q) || error("node $i: parent $q is not scheduled")
                is_view(p.how[q]) || pos[q] < pos[i] || error("node $i is scheduled before its parent $q")
            end
        end
        _check_derivation(p, i, d)
        (n.requested && p.buffer[i] == 0 && !is_view(d)) && error("requested node $i has no storage")
    end
    return p
end
_check_derivation(p::Plan, i, ::Base_) = nothing
function _check_derivation(p::Plan{N}, i, d::Coarsen{N}) where {N}
    parent, child = p.nodes[d.parent].window, p.nodes[i].window
    all(a -> can_coarsen(parent[a], child[a]), 1:N) || error("node $i: invalid coarsen from $(d.parent)")
    return nothing
end
function _check_derivation(p::Plan{N}, i, d::Compose) where {N}
    a, b, child = p.nodes[d.a].window, p.nodes[d.b].window, p.nodes[i].window
    for e in 1:N
        e == d.axis && continue
        (a[e] == child[e] && b[e] == child[e]) || error("node $i: compose parents differ off axis $(d.axis)")
    end
    can_compose(a[d.axis], b[d.axis], child[d.axis]) || error("node $i: invalid compose from $(d.a), $(d.b)")
    return nothing
end
function _check_derivation(p::Plan{N}, i, d::Restride{N}) where {N}
    parent, child = p.nodes[d.parent].window, p.nodes[i].window
    all(a -> can_restride(parent[a], child[a]), 1:N) || error("node $i: invalid restride from $(d.parent)")
    return nothing
end
function _check_derivation(p::Plan{N}, i, d::Scan) where {N}
    parent, child = p.nodes[d.parent].window, p.nodes[i].window
    for e in 1:N
        e == d.axis && continue
        parent[e] == child[e] || error("node $i: scan parent differs off axis $(d.axis)")
    end
    pa, ca = parent[d.axis], child[d.axis]
    (pa.size == 1 && pa.pos == Progression(0, 1, pa.extent) && ca.size == d.size && ca.partial == d.partial &&
     ca.pos == strided(ca.extent, d.size, 1, d.partial ? Partial() : Truncate()).pos) ||
        error("node $i: invalid scan from $(d.parent)")
    return nothing
end
