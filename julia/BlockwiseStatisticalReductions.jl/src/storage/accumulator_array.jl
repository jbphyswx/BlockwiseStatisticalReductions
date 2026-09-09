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

# The nesting is walked while the code is generated, so one body reaches every leaf and nothing between
# the accumulator and a component array is ever an argument the compiler has to materialize.
_has_leaves(::Type{T}) where {T} = T <: AbstractAccumulator || T <: Tuple
_component(::Type{T}, comps, k) where {T} =
    Expr(:., comps, QuoteNode(T <: Tuple ? _member_names(Val(fieldcount(T)))[k] : fieldname(T, k)))

function _read_expr(::Type{T}, comps) where {T}
    _has_leaves(T) || return :(_load($comps, i))
    fields = Expr[_read_expr(fieldtype(T, k), _component(T, comps, k)) for k in 1:fieldcount(T)]
    return T <: Tuple ? Expr(:tuple, fields...) : Expr(:call, T, fields...)
end
function _write_exprs!(out, ::Type{T}, comps, val) where {T}
    if _has_leaves(T)
        for k in 1:fieldcount(T)
            _write_exprs!(out, fieldtype(T, k), _component(T, comps, k), :(getfield($val, $k)))
        end
    else
        push!(out, :(_store!($comps, $val, i)))
    end
    return out
end

@generated function _read(::Type{A}, comps::NamedTuple, i) where {A<:AbstractAccumulator}
    return quote
        Base.@_inline_meta
        $(_read_expr(A, :comps))
    end
end
@generated function _write!(::Type{A}, comps::NamedTuple, a, i) where {A<:AbstractAccumulator}
    return quote
        Base.@_inline_meta
        $(_write_exprs!(Expr[], A, :comps, :a)...)
        return nothing
    end
end
@inline _load(c::AbstractArray, i) = @inbounds c[i]
@inline _load(c::Uniform, i) = c.value
@inline _store!(c::AbstractArray, v, i) = (@inbounds c[i] = v; nothing)
@inline _store!(::Uniform, v, i) = nothing

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
