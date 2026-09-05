"""
    Shifted(field, shift)

An input field read as `field[J] - shift`. Accumulating shifted observations keeps a large mean out of
every difference the moment kernels take, so the accumulation eltype only has to resolve the spread of
the data rather than its offset. [`unshift`](@ref) puts the offset back at finalize.
"""
struct Shifted{A<:AbstractArray,S}
    field::A
    shift::S
end
@inline value(f::AbstractArray, J::CartesianIndex) = @inbounds f[J]
@inline value(f::Shifted, J::CartesianIndex) = (@inbounds f.field[J]) - f.shift
Base.size(f::Shifted) = size(f.field)
Base.eltype(::Type{Shifted{A,S}}) where {A,S} = promote_type(eltype(A), S)

"Raw input fields read at an index and lifted into accumulators of type `A`."
struct Lift{A<:AbstractAccumulator,F<:Tuple}
    fields::F
end
Lift{A}(fields::F) where {A,F<:Tuple} = Lift{A,F}(fields)

@inline leaf(src::Lift{A}, J::CartesianIndex) where {A} = lift(A, map(f -> value(f, J), src.fields))
@inline leaf(src::AccumulatorArray, J::CartesianIndex) = @inbounds src[J]

"`L` consecutive indices from `start`; the length is part of the type so box loops unroll."
struct StaticRange{L}
    start::Int
end
Base.length(::StaticRange{L}) where {L} = L
Base.first(r::StaticRange) = r.start
Base.last(r::StaticRange{L}) where {L} = r.start + L - 1
Base.iterate(r::StaticRange{L}, i::Int = r.start) where {L} = i > r.start + L - 1 ? nothing : (i, i + 1)
Base.eltype(::Type{<:StaticRange}) = Int
_dynamic(r::UnitRange{Int}) = r
_dynamic(r::StaticRange) = first(r):last(r)
_static_length(::Type{StaticRange{L}}) where {L} = L
_static_length(::Type{UnitRange{Int}}) = 0

"""
    Box(src, ranges)

The leaves of `src` inside the Cartesian product of per-axis index `ranges`; iterates with the first
axis fastest. Kernels combine one box per output cell.
"""
struct Box{S,N,R<:Tuple}
    src::S
    ranges::R
    Box(src::S, ranges::R) where {S,N,R<:NTuple{N,Union{UnitRange{Int},StaticRange}}} = new{S,N,R}(src, ranges)
end
Base.IteratorSize(::Type{<:Box}) = Base.HasLength()
Base.length(b::Box) = prod(length, b.ranges)
Base.eltype(::Type{Box{Lift{A,F},N,R}}) where {A,F,N,R} = A
Base.eltype(::Type{Box{S,N,R}}) where {A,S<:AccumulatorArray{A},N,R} = A
_indices(b::Box) = CartesianIndices(map(_dynamic, b.ranges))
@inline function Base.iterate(b::Box, state = nothing)
    inds = _indices(b)
    it = state === nothing ? iterate(inds) : iterate(inds, state)
    it === nothing && return nothing
    return leaf(b.src, it[1]), it[2]
end
@inline _outer(b::Box) = CartesianIndices(map(_dynamic, Base.tail(b.ranges)))

# Box kinds: every axis static and small (fully unrolled), first axis static (unrolled rows), or dynamic rows.
struct FullyStatic end
struct StaticRows end
struct DynamicRows end
"Largest box (in cells) that is unrolled as one expression tree; tuples this long still specialize."
const FULLY_STATIC_CELLS = 32
function _kind_of(R)
    lengths = Int[]
    for d in 1:fieldcount(R)
        push!(lengths, _static_length(fieldtype(R, d)))
    end
    all(>(0), lengths) && prod(lengths) <= FULLY_STATIC_CELLS && return :(FullyStatic())
    lengths[1] > 0 && return :(StaticRows())
    return :(DynamicRows())
end
@generated _kind(::Type{R}) where {R<:Tuple} = _kind_of(R)

# Balanced expression tree of `op` over the elements of a tuple (short dependency chains).
function _tree_expr(op, lo, hi)
    lo == hi && return :(t[$lo])
    mid = (lo + hi) ÷ 2
    return :($op($(_tree_expr(op, lo, mid)), $(_tree_expr(op, mid + 1, hi))))
end
_treereduce_expr(n) = quote
    Base.@_inline_meta
    $(_tree_expr(:op, 1, n))
end
@generated _treereduce(op, t::NTuple{n,Any}) where {n} = _treereduce_expr(n)

# Every index of a fully static box, as a tuple of `CartesianIndex` (first axis fastest).
function _box_indices_expr(R)
    dims = Int[]
    for d in 1:fieldcount(R)
        push!(dims, _static_length(fieldtype(R, d)))
    end
    indices = Expr[]
    for I in CartesianIndices(Tuple(dims))
        offsets = Tuple(I) .- 1
        push!(indices, Expr(:call, :CartesianIndex, [:(first(b.ranges[$d]) + $(offsets[d])) for d in 1:length(dims)]...))
    end
    return quote
        Base.@_inline_meta
        ($(indices...),)
    end
end
@generated _box_indices(b::Box{S,N,R}) where {S,N,R} = _box_indices_expr(R)
@inline _row_indices(start::Int, K, ::Val{L}) where {L} = ntuple(i -> CartesianIndex(start + i - 1, K), Val(L))

"""
    boxreduce(contrib, op, init, b::Box)

Reduce `contrib(J)` over the indices `J` of `b` with the associative `op`, starting from `init`.
Static axes unroll into balanced trees; dynamic rows run a SIMD loop along the contiguous axis.
"""
@inline boxreduce(contrib::C, op::O, init, b::Box) where {C,O} = _boxreduce(contrib, op, init, b, _kind(typeof(b.ranges)))
@inline _boxreduce(contrib::C, op::O, init, b::Box, ::FullyStatic) where {C,O} =
    op(init, _treereduce(op, map(contrib, _box_indices(b))))
@inline function _boxreduce(contrib::C, op::O, init, b::Box, ::StaticRows) where {C,O}
    r1 = b.ranges[1]
    acc = init
    @inbounds for K in _outer(b)
        acc = op(acc, _treereduce(op, map(contrib, _row_indices(r1.start, K, Val(length(r1))))))
    end
    return acc
end
@inline function _boxreduce(contrib::C, op::O, init, b::Box, ::DynamicRows) where {C,O}
    r1 = b.ranges[1]
    acc = init
    @inbounds for K in _outer(b)
        @simd for i in r1
            acc = op(acc, contrib(CartesianIndex(i, K)))
        end
    end
    return acc
end

# Phase functors carry the accumulator type in their own type, so no closure captures a type parameter.
struct Phase1Lift{A,S}
    src::S
end
struct Phase1Merge{A} end
struct Phase2Lift{A,S,Mt}
    src::S
    m::Mt
end
struct Phase2Merge{A} end
@inline (f::Phase1Lift{A})(J) where {A} = p1lift(A, leaf(f.src, J))
@inline (::Phase1Merge{A})(s, t) where {A} = p1merge(A, s, t)
@inline (f::Phase2Lift{A})(J) where {A} = p2lift(A, leaf(f.src, J), f.m)
@inline (::Phase2Merge{A})(s, t) where {A} = p2merge(A, s, t)

"""
    combine(::Type{A}, b::Box) -> A

The two-phase protocol over the leaves of a box; for composites every member's phases run fused.
"""
@inline function combine(::Type{A}, b::Box{S}) where {A<:AbstractAccumulator,S}
    s1 = boxreduce(Phase1Lift{A,S}(b.src), Phase1Merge{A}(), p1init(A), b)
    phases(A) == 1 && return finish(A, s1, nothing, nothing)
    m = mid(A, s1)
    s2 = boxreduce(Phase2Lift{A,S,typeof(m)}(b.src, m), Phase2Merge{A}(), p2init(A, m), b)
    return finish(A, s1, m, s2)
end

_source(::Type{A}, fields::NamedTuple) where {A} = Lift{A}(values(fields))
_source(::Type{A}, fields::Tuple) where {A} = Lift{A}(fields)
_source(::Type{A}, field::AbstractArray) where {A} = Lift{A}((field,))
_source(::Type{A}, parent::AccumulatorArray{A}) where {A} = parent

function _check_boxfold(out::AccumulatorArray{A,N}, src::Lift, w::Window{N}) where {A,N}
    size(out) == shape(w) || throw(DimensionMismatch("output $(size(out)) does not match window shape $(shape(w))"))
    extents = map(aw -> aw.extent, w)
    for f in src.fields
        size(f) == extents || throw(DimensionMismatch("field of size $(size(f)) does not match window extents $extents"))
    end
    length(src.fields) == arity(A) || throw(ArgumentError("$(length(src.fields)) field(s) for an accumulator of arity $(arity(A))"))
    return nothing
end
function _check_boxfold(out::AccumulatorArray{A,N}, parent::AccumulatorArray{A,N}, w::Window{N}) where {A,N}
    size(out) == shape(w) || throw(DimensionMismatch("output $(size(out)) does not match window shape $(shape(w))"))
    size(parent) == map(aw -> aw.extent, w) || throw(DimensionMismatch("parent $(size(parent)) does not match window extents $(map(aw -> aw.extent, w))"))
    return nothing
end

"Axes whose windows have at most this many cells are unrolled at compile time."
const STATIC_LIMIT = 8

"""
    static_shape(w::Window) -> NTuple{N,Int}

Per-axis compile-time box lengths for `w`: the window size on axes with uniform length ≤ `STATIC_LIMIT`,
`0` where the length stays dynamic. Pass it as `Val(static_shape(w))` to [`boxfold!`](@ref) to avoid a
dynamic dispatch per call.
"""
static_shape(w::Window{N}) where {N} = ntuple(d -> uniform_length(w[d]) && w[d].size <= STATIC_LIMIT ? w[d].size : 0, Val(N))

function _box_ranges_expr(N, S)
    parts = Expr[]
    for d in 1:N
        push!(parts, S[d] == 0 ? :(window_range(w[$d], I[$d])) : :(StaticRange{$(S[d])}(origin(w[$d].pos, I[$d]) + 1)))
    end
    return quote
        Base.@_inline_meta
        ($(parts...),)
    end
end
@generated _box_ranges(w::Window{N}, I::CartesianIndex{N}, ::Val{S}) where {N,S} = _box_ranges_expr(N, S)

function _check_shape(w::Window{N}, S::NTuple{N,Int}) where {N}
    for d in 1:N
        S[d] == 0 && continue
        (uniform_length(w[d]) && w[d].size == S[d]) ||
            throw(ArgumentError("static shape $S does not describe axis $d of the window (size $(w[d].size), uniform $(uniform_length(w[d])))"))
    end
    return nothing
end

"""
    boxfold!(out, src, w::Window, [shape::Val,] backend) -> out

Fill every cell `I` of `out` with the combination of the leaves of `src` inside window `w` at `I`.
`src` is a field array, a tuple or `NamedTuple` of field arrays (base pass: leaves are lifted
observations), or an `AccumulatorArray` (coarsen: leaves are its cells, `w` addressed in its cell grid).
`shape` is `Val(static_shape(w))`; without it, it is computed and dispatched on once per call.
"""
boxfold!(out::AccumulatorArray{A,N}, src, w::Window{N}, backend::CB.AbstractSerialBackend) where {A,N} =
    boxfold!(out, src, w, Val(static_shape(w)), backend)
function boxfold!(out::AccumulatorArray{A,N}, src, w::Window{N}, ::Val{S}, ::CB.AbstractSerialBackend) where {A,N,S}
    source = _source(A, src)
    _check_boxfold(out, source, w)
    _check_shape(w, S)
    @inbounds for I in CartesianIndices(size(out))
        out[I] = combine(A, Box(source, _box_ranges(w, I, Val(S))))
    end
    return out
end
boxfold!(out::AccumulatorArray{A,N}, src, w::Window{N}, b::CB.AbstractExecutionBackend) where {A,N} = missing_extension(b)
boxfold!(out::AccumulatorArray{A,N}, src, w::Window{N}, ::Val, b::CB.AbstractExecutionBackend) where {A,N} = missing_extension(b)

"""
    compose!(out, a, b, axis, amap, bmap, backend) -> out

`out[I] = merge(a[Ia], b[Ib])` where `Ia`/`Ib` are `I` with the `axis` index replaced by
`amap[I[axis]]`/`bmap[I[axis]]`; the other axes must match between `out`, `a` and `b`.
"""
function check_compose(out::AccumulatorArray{A,N}, a::AccumulatorArray{A,N}, b::AccumulatorArray{A,N},
                      axis::Int, amap::AbstractVector{Int}, bmap::AbstractVector{Int}) where {A,N}
    1 <= axis <= N || throw(ArgumentError("axis $axis out of range"))
    n = size(out, axis)
    length(amap) == n || throw(DimensionMismatch("$(length(amap)) first-parent indices for $n cells"))
    length(bmap) == n || throw(DimensionMismatch("$(length(bmap)) second-parent indices for $n cells"))
    for d in 1:N
        d == axis && continue
        size(a, d) == size(out, d) == size(b, d) ||
            throw(DimensionMismatch("parents $(size(a)), $(size(b)) do not match output $(size(out)) off axis $axis"))
    end
    (isempty(amap) || (1 <= minimum(amap) && maximum(amap) <= size(a, axis))) || throw(BoundsError(a, maximum(amap)))
    (isempty(bmap) || (1 <= minimum(bmap) && maximum(bmap) <= size(b, axis))) || throw(BoundsError(b, maximum(bmap)))
    return nothing
end

"`out[I]` for one cell of a compose: the merge of each parent at its mapped index along `axis`."
@inline function compose_cell(a::AccumulatorArray{A,N}, b::AccumulatorArray{A,N}, I::CartesianIndex{N},
                              axis::Int, amap::AbstractVector{Int}, bmap::AbstractVector{Int}) where {A,N}
    i = I[axis]
    @inbounds Ia = CartesianIndex(Base.setindex(Tuple(I), amap[i], axis))
    @inbounds Ib = CartesianIndex(Base.setindex(Tuple(I), bmap[i], axis))
    return @inbounds merge(a[Ia], b[Ib])
end

function compose!(out::AccumulatorArray{A,N}, a::AccumulatorArray{A,N}, b::AccumulatorArray{A,N},
                  axis::Int, amap::AbstractVector{Int}, bmap::AbstractVector{Int}, ::CB.AbstractSerialBackend) where {A,N}
    check_compose(out, a, b, axis, amap, bmap)
    @inbounds for I in CartesianIndices(size(out))
        out[I] = compose_cell(a, b, I, axis, amap, bmap)
    end
    return out
end
compose!(out::AccumulatorArray{A,N}, a::AccumulatorArray{A,N}, b::AccumulatorArray{A,N}, axis::Int,
         amap::AbstractVector{Int}, bmap::AbstractVector{Int}, bk::CB.AbstractExecutionBackend) where {A,N} = missing_extension(bk)
