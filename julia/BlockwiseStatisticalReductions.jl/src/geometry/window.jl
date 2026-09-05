# Window origins along one axis are 0-based offsets from the axis' first index; a window at origin `o`
# covers 1-based indices `o+1 : min(o+size, extent)`.

abstract type Positions end

"Origins `offset, offset+stride, …` (`count` of them)."
struct Progression <: Positions
    offset::Int
    stride::Int
    count::Int
    function Progression(offset::Integer, stride::Integer, count::Integer)
        offset >= 0 || throw(ArgumentError("offset must be ≥ 0, got $offset"))
        stride >= 1 || throw(ArgumentError("stride must be ≥ 1, got $stride"))
        count >= 0 || throw(ArgumentError("count must be ≥ 0, got $count"))
        return new(offset, stride, count)
    end
end

"""
    Origins(origins::AbstractVector{<:Integer})

An explicit strictly increasing list of origins. Parametric in the vector type so a window can be moved
to a device alongside the data it addresses; the checked constructor builds the host form.
"""
struct Origins{V<:AbstractVector{Int}} <: Positions
    origins::V
    # Declared so no unchecked outer constructor is generated; the checked one below is the way in, and
    # this parametric form is for moving an already-checked list to another array type.
    Origins{V}(origins::V) where {V<:AbstractVector{Int}} = new{V}(origins)
end
function Origins(origins::AbstractVector{<:Integer})
    v = Vector{Int}(origins)
    (isempty(v) || v[1] >= 0) || throw(ArgumentError("origins must be ≥ 0"))
    all(i -> v[i] < v[i+1], 1:length(v)-1) || throw(ArgumentError("origins must be strictly increasing"))
    return Origins{Vector{Int}}(v)
end

"Origins of a `Positions` as an `AbstractVector{Int}`."
origins(p::Progression) = range(p.offset; step = p.stride, length = p.count)
origins(p::Origins) = p.origins
nwindows(p::Progression) = p.count
nwindows(p::Origins) = length(p.origins)
"The `i`-th origin."
@inline origin(p::Progression, i::Integer) = p.offset + (i - 1) * p.stride
@inline origin(p::Origins, i::Integer) = @inbounds p.origins[i]

Base.:(==)(a::Positions, b::Positions) = origins(a) == origins(b)
Base.hash(p::Positions, h::UInt) = hash(collect(origins(p)), hash(Positions, h))

"`true` when every origin of `q`, shifted by `shift`, is an origin of `p`."
function contains(p::Progression, q::Progression, shift::Int)
    q.count == 0 && return true
    first = q.offset + shift - p.offset
    (first >= 0 && first % p.stride == 0) || return false
    q.count == 1 || q.stride % p.stride == 0 || return false
    return first + q.stride * (q.count - 1) <= p.stride * (p.count - 1)
end
contains(p::Positions, q::Positions, shift::Int) = all(o -> _has_origin(p, o + shift), origins(q))
_has_origin(p::Progression, o::Int) = o >= p.offset && (o - p.offset) % p.stride == 0 && (o - p.offset) ÷ p.stride < p.count
_has_origin(p::Origins, o::Int) = insorted(o, p.origins)

"""
    AxisWindow(extent, size, pos::Positions, partial::Bool)

Windows of `size` cells at origins `pos` along an axis of `extent` cells. With `partial == false` every
window lies inside the axis; with `partial == true` the trailing windows may be clipped at the extent.
"""
struct AxisWindow{P<:Positions}
    extent::Int
    size::Int
    pos::P
    partial::Bool
    function AxisWindow(extent::Integer, size::Integer, pos::P, partial::Bool) where {P<:Positions}
        extent >= 0 || throw(ArgumentError("extent must be ≥ 0, got $extent"))
        size >= 1 || throw(ArgumentError("window size must be ≥ 1, got $size"))
        os = origins(pos)
        (isempty(os) || os[end] < extent) || throw(ArgumentError("origin $(os[end]) lies outside extent $extent"))
        (partial || isempty(os) || os[end] + size <= extent) ||
            throw(ArgumentError("window at origin $(os[end]) of size $size exceeds extent $extent; use partial = true to clip"))
        return new{P}(extent, size, pos, partial)
    end
end

const Window{N} = NTuple{N,AxisWindow}

# Explicit origins live in an array, so a window only reaches a device kernel after adapting.
Adapt.adapt_structure(to, p::Origins) = (v = Adapt.adapt(to, p.origins); Origins{typeof(v)}(v))
Adapt.adapt_structure(to, aw::AxisWindow) = AxisWindow(aw.extent, aw.size, Adapt.adapt(to, aw.pos), aw.partial)

Base.:(==)(a::AxisWindow, b::AxisWindow) =
    a.extent == b.extent && a.size == b.size && a.partial == b.partial && a.pos == b.pos
Base.hash(aw::AxisWindow, h::UInt) = hash(aw.pos, hash(aw.partial, hash(aw.size, hash(aw.extent, hash(AxisWindow, h)))))

nwindows(aw::AxisWindow) = nwindows(aw.pos)
origins(aw::AxisWindow) = origins(aw.pos)
"`true` when the positions tile the axis (stride equals size)."
is_tiled(aw::AxisWindow) = aw.pos isa Progression && aw.pos.stride == aw.size
"1-based input index range of window `i`."
@inline window_range(aw::AxisWindow, i::Integer) = (o = origin(aw.pos, i); (o + 1):min(o + aw.size, aw.extent))
@inline window_length(aw::AxisWindow, i::Integer) = length(window_range(aw, i))
"A window whose `partial` flag is dropped when no window is actually clipped."
canonicalize(aw::AxisWindow) = aw.partial && uniform_length(aw) ? AxisWindow(aw.extent, aw.size, aw.pos, false) : aw

"`true` when every window holds exactly `size` cells."
uniform_length(aw::AxisWindow) = nwindows(aw) == 0 || origins(aw)[end] + aw.size <= aw.extent
uniform_length(w::Window) = all(uniform_length, w)
"Output shape of an N-D window."
shape(w::Window{N}) where {N} = map(nwindows, w)
"Number of input cells one window covers, for windows of uniform length."
volume(w::Window) = prod(aw -> aw.size, w)

abstract type EdgePolicy end
"Only windows that lie inside the axis; the trailing remainder is dropped."
struct Truncate <: EdgePolicy end
"Trailing windows are clipped at the extent and carry their true smaller counts."
struct Partial <: EdgePolicy end
"Windows must cover the axis exactly."
struct Strict <: EdgePolicy end
"Like `Truncate` with the dropped remainder split evenly between both ends."
struct Centered <: EdgePolicy end

"Tiling of an axis of `extent` cells by windows of `size` cells under an edge policy."
tiled(extent::Integer, size::Integer, ::Truncate) = AxisWindow(extent, size, Progression(0, size, extent ÷ size), false)
tiled(extent::Integer, size::Integer, ::Partial) = AxisWindow(extent, size, Progression(0, size, cld(extent, size)), true)
tiled(extent::Integer, size::Integer, ::Centered) =
    AxisWindow(extent, size, Progression(extent < size ? 0 : (extent % size) ÷ 2, size, extent ÷ size), false)
function tiled(extent::Integer, size::Integer, ::Strict)
    extent % size == 0 || throw(ArgumentError("size $size does not tile extent $extent exactly"))
    return tiled(extent, size, Truncate())
end
tiled(extents::NTuple{N,Integer}, sizes::NTuple{N,Integer}, policy::EdgePolicy) where {N} =
    ntuple(d -> tiled(extents[d], sizes[d], policy), Val(N))

"Windows of `size` placed every `stride` cells from the axis start under an edge policy."
strided(extent::Integer, size::Integer, stride::Integer, ::Truncate) =
    AxisWindow(extent, size, Progression(0, stride, extent < size ? 0 : (extent - size) ÷ stride + 1), false)
strided(extent::Integer, size::Integer, stride::Integer, ::Partial) =
    AxisWindow(extent, size, Progression(0, stride, cld(extent, stride)), true)
function strided(extent::Integer, size::Integer, stride::Integer, ::Centered)
    extent < size && return AxisWindow(extent, size, Progression(0, stride, 0), false)
    return AxisWindow(extent, size, Progression(((extent - size) % stride) ÷ 2, stride, (extent - size) ÷ stride + 1), false)
end
function strided(extent::Integer, size::Integer, stride::Integer, ::Strict)
    (extent >= size && (extent - size) % stride == 0) ||
        throw(ArgumentError("windows of size $size every $stride cells do not end at extent $extent exactly"))
    return strided(extent, size, stride, Truncate())
end

"Windows of `size` at explicit `origins`."
anchored(extent::Integer, size::Integer, origins::AbstractVector{<:Integer}; partial::Bool = false) =
    AxisWindow(extent, size, Origins(origins), partial)

# ── Derivation predicates ────────────────────────────────────────────────────────

"`true` when `child` selects a subset of `parent`'s windows (same extent and size)."
can_restride(parent::AxisWindow, child::AxisWindow) =
    parent.extent == child.extent && parent.size == child.size && contains(parent.pos, child.pos, 0)

"`true` when every `child` window is the union of `k` consecutive windows of the tiled `parent`."
function can_coarsen(parent::AxisWindow, child::AxisWindow)
    parent.extent == child.extent && is_tiled(parent) || return false
    child.size % parent.size == 0 || return false
    k = child.size ÷ parent.size
    ps = parent.size
    return all(o -> all(j -> (p = o + j * ps; p >= parent.extent || _has_origin(parent.pos, p)), 0:k-1), origins(child))
end

"`true` when every `child` window is the window of `a` at the same origin followed by the window of `b` at origin + `a.size`."
function can_compose(a::AxisWindow, b::AxisWindow, child::AxisWindow)
    a.extent == b.extent == child.extent || return false
    child.size == a.size + b.size || return false
    contains(a.pos, child.pos, 0) || return false
    return all(o -> (q = o + a.size; q >= a.extent || _has_origin(b.pos, q)), origins(child))
end

# ── Derivation results (candidate children for the planner) ─────────────────────

"The window obtained by merging `k` consecutive windows of a tiled `parent`."
function coarsen_result(parent::AxisWindow{Progression}, k::Integer)
    is_tiled(parent) || throw(ArgumentError("coarsen needs a tiled parent"))
    count = parent.partial ? cld(parent.pos.count, k) : parent.pos.count ÷ k
    return AxisWindow(parent.extent, k * parent.size, Progression(parent.pos.offset, k * parent.size, count), parent.partial)
end

"The window of size `a.size + b.size` at every origin of `a` whose continuation exists in `b`."
function compose_result(a::AxisWindow, b::AxisWindow)
    a.extent == b.extent || throw(ArgumentError("compose needs equal extents"))
    size = a.size + b.size
    kept = filter(o -> (q = o + a.size; (a.partial && q >= a.extent) || _has_origin(b.pos, q)), collect(origins(a)))
    partial = any(o -> o + size > a.extent, kept)
    pos = _as_positions(kept, a.pos)
    return AxisWindow(a.extent, size, pos, partial)
end

function _as_positions(kept::Vector{Int}, like::Progression)
    n = length(kept)
    n == 0 && return Progression(like.offset, like.stride, 0)
    kept == collect(range(kept[1]; step = like.stride, length = n)) && return Progression(kept[1], like.stride, n)
    return Origins(kept)
end
_as_positions(kept::Vector{Int}, ::Origins) = Origins(kept)
