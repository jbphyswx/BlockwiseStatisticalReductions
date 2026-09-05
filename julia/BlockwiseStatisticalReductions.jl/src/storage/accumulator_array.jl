using Adapt: Adapt

"A component whose value is the same in every cell; occupies no memory and ignores writes."
struct Uniform{T}
    value::T
end

"""
    AccumulatorArray{A,N,C}(components::C, dims)

Struct-of-arrays storage for accumulators of type `A`: one component array (or [`Uniform`](@ref)) per
leaf field, nested as a `NamedTuple` mirroring the accumulator's fields; tuple fields (composite
members, power sums) appear as `m1, m2, …`. Indexing rebuilds the accumulator; writing scatters its fields.
"""
struct AccumulatorArray{A<:AbstractAccumulator,N,C<:NamedTuple} <: AbstractArray{A,N}
    components::C
    dims::NTuple{N,Int}
end

"""
    AccumulatorArray(::Type{A}, prototype::AbstractArray, dims; uniform = (;))

Allocate storage for `A` with every leaf array created by `similar(prototype, T, dims)` (so the
storage lives where `prototype` lives). Leaf fields named in `uniform` become [`Uniform`](@ref)
components with the given values at every nesting level.
"""
function AccumulatorArray(::Type{A}, prototype::AbstractArray, dims::Dims{N}; uniform::NamedTuple = (;)) where {A<:AbstractAccumulator,N}
    components = _allocate(A, prototype, dims, uniform)
    return AccumulatorArray{A,N,typeof(components)}(components, dims)
end

_allocate(::Type{A}, proto, dims, uniform) where {A<:AbstractAccumulator} =
    NamedTuple{fieldnames(A)}(ntuple(k -> _allocate_field(fieldname(A, k), fieldtype(A, k), proto, dims, uniform), Val(fieldcount(A))))
_allocate(::Type{M}, proto, dims, uniform) where {M<:Tuple} =
    NamedTuple{_member_names(Val(fieldcount(M)))}(ntuple(k -> _allocate(fieldtype(M, k), proto, dims, uniform), Val(fieldcount(M))))
_allocate(::Type{T}, proto, dims, uniform) where {T} = similar(proto, T, dims)
_allocate_field(name::Symbol, ::Type{T}, proto, dims, uniform) where {T} =
    haskey(uniform, name) && !(T <: Union{AbstractAccumulator,Tuple}) ? Uniform{T}(uniform[name]) : _allocate(T, proto, dims, uniform)
_member_names(::Val{K}) where {K} = ntuple(k -> Symbol(:m, k), Val(K))

Base.size(aa::AccumulatorArray) = aa.dims
Base.IndexStyle(::Type{<:AccumulatorArray}) = IndexLinear()

@inline function Base.getindex(aa::AccumulatorArray{A}, i::Int) where {A}
    @boundscheck checkbounds(aa, i)
    return _read(A, aa.components, i)
end
@inline function Base.setindex!(aa::AccumulatorArray{A}, a::A, i::Int) where {A}
    @boundscheck checkbounds(aa, i)
    _write!(A, aa.components, a, i)
    return aa
end

# Field loops are generated so every field type and index is a compile-time constant.
function _field_exprs(T, expr)
    out = Expr[]
    for k in 1:fieldcount(T)
        push!(out, expr(k, fieldtype(T, k)))
    end
    return out
end
_read_expr(T, wrap) = quote
    Base.@_inline_meta
    $(wrap(_field_exprs(T, (k, Tk) -> :(_read($Tk, comps[$k], i)))))
end
_write_expr(T, getter) = quote
    Base.@_inline_meta
    $(_field_exprs(T, (k, Tk) -> :(_write!($Tk, comps[$k], $(getter(k)), i)))...)
    return nothing
end

@generated _read(::Type{A}, comps::NamedTuple, i) where {A<:AbstractAccumulator} = _read_expr(A, fields -> :($A($(fields...))))
@generated _read(::Type{M}, comps::NamedTuple, i) where {M<:Tuple} = _read_expr(M, fields -> :(($(fields...),)))
@inline _read(::Type{T}, c::AbstractArray, i) where {T} = @inbounds c[i]
@inline _read(::Type{T}, c::Uniform, i) where {T} = c.value

@generated _write!(::Type{A}, comps::NamedTuple, a, i) where {A<:AbstractAccumulator} = _write_expr(A, k -> :(getfield(a, $k)))
@generated _write!(::Type{M}, comps::NamedTuple, t::Tuple, i) where {M<:Tuple} = _write_expr(M, k -> :(t[$k]))
@inline _write!(::Type{T}, c::AbstractArray, v, i) where {T} = (@inbounds c[i] = v; nothing)
@inline _write!(::Type{T}, ::Uniform, v, i) where {T} = nothing

"""
    component(aa::AccumulatorArray, path::Symbol...) -> AbstractArray or Uniform

The storage of one leaf field, e.g. `component(aa, :mean)` or `component(aa, :members, :m2, :M2)`.
"""
component(aa::AccumulatorArray, path::Symbol...) = foldl(getproperty, path; init = aa.components)

"Zero-copy storage of member `k` of an array of composites."
@inline function member_array(aa::AccumulatorArray{Composite{M,B},N}, ::Val{k}) where {M,B,N,k}
    c = aa.components.members[k]
    return AccumulatorArray{fieldtype(M, k),N,typeof(c)}(c, aa.dims)
end

Base.similar(aa::AccumulatorArray{A,N}) where {A,N} =
    AccumulatorArray{A,N,typeof(aa.components)}(_similar_components(aa.components), aa.dims)
_similar_components(nt::NamedTuple) = map(_similar_components, nt)
_similar_components(c::AbstractArray) = similar(c)
_similar_components(c::Uniform) = c

Adapt.adapt_structure(to, aa::AccumulatorArray{A,N}) where {A,N} =
    (components = Adapt.adapt(to, aa.components); AccumulatorArray{A,N,typeof(components)}(components, aa.dims))
