"""
    AbstractAccumulator

Sufficient statistics of a set of observations. Concrete subtypes are immutable isbits structs with the
default positional constructor and implement [`neutral`](@ref), [`lift`](@ref) and `Base.merge`.
"""
abstract type AbstractAccumulator end

"""
    AbstractStatistic

A requested statistic. A tag selects an accumulator type ([`accumulator_type`](@ref)), reads input
fields ([`bindings`](@ref)) and turns one accumulator into a value ([`finalize`](@ref)).
"""
abstract type AbstractStatistic end

"""
    neutral(::Type{A}) -> A

Monoid identity: `merge(neutral(A), a) == a == merge(a, neutral(A))`.
"""
function neutral end

"""
    lift(::Type{A}, xs::Tuple) -> A

Accumulator of one observation; `xs` holds the `arity(A)` bound raw values in binding order.
"""
function lift end

"""
    combine(::Type{A}, children) -> A

Combine an iterable of accumulators; the k-ary counterpart of `merge(a::A, b::A)`. Runs the two-phase
protocol: phase 1 folds [`p1lift`](@ref) with [`p1merge`](@ref), [`mid`](@ref) derives what phase 2
needs, phase 2 folds [`p2lift`](@ref) with [`p2merge`](@ref), and [`finish`](@ref) builds the result.
Kernels run the same protocol over boxes and fuse the phases of all members of a composite.
"""
function combine(::Type{A}, children) where {A<:AbstractAccumulator}
    s1 = p1init(A)
    for c in children
        s1 = p1merge(A, s1, p1lift(A, c))
    end
    phases(A) == 1 && return finish(A, s1, nothing, nothing)
    m = mid(A, s1)
    s2 = p2init(A, m)
    for c in children
        s2 = p2merge(A, s2, p2lift(A, c, m))
    end
    return finish(A, s1, m, s2)
end

"Number of passes over the children `combine` needs: 1 or 2."
phases(::Type{<:AbstractAccumulator}) = 1
"Phase-1 state before any child."
p1init(::Type{A}) where {A<:AbstractAccumulator} = neutral(A)
"Phase-1 contribution of one child."
@inline p1lift(::Type{A}, c) where {A<:AbstractAccumulator} = c
"Associative combination of two phase-1 states."
@inline p1merge(::Type{A}, s, t) where {A<:AbstractAccumulator} = merge(s, t)
"Quantities phase 2 needs, from the completed phase-1 state."
function mid end
"Phase-2 state before any child."
function p2init end
"Phase-2 contribution of one child given the phase-1 quantities `m`."
function p2lift end
"Associative combination of two phase-2 states."
@inline p2merge(::Type{A}, s, t) where {A<:AbstractAccumulator} = s + t
"The accumulator from the phase states (`m` and `s2` are `nothing` for single-phase types)."
@inline finish(::Type{A}, s1, m, s2) where {A<:AbstractAccumulator} = s1

"""
    unmerge(ab::A, b::A) -> A

Group inverse: the `a` with `merge(a, b) == ab`. Defined only when [`is_invertible`](@ref).
"""
function unmerge end

"`true` when [`unmerge`](@ref) is defined for the accumulator type."
is_invertible(::Type{<:AbstractAccumulator}) = false

"Number of input fields one observation of the accumulator binds."
arity(::Type{<:AbstractAccumulator}) = 1

"Element type of the accumulator's moment fields."
function acc_eltype end

"""
    accumulation_eltype(::Type{Tin}) -> Type

Element type in which moments accumulate for input eltype `Tin`: `Float16 → Float32`,
`Float32 → Float64`, integers and `Bool → Float64`, other floats unchanged.
"""
accumulation_eltype(::Type{T}) where {T<:AbstractFloat} = T
accumulation_eltype(::Type{Float16}) = Float32
accumulation_eltype(::Type{Float32}) = Float64
accumulation_eltype(::Type{T}) where {T<:Integer} = Float64
accumulation_eltype(::Type{Bool}) = Float64
accumulation_eltype(::Type{T}) where {T} = float(T)

"""
    accumulator_type(tag, ::Type{Tin}, ::Type{Tacc}) -> Type{<:AbstractAccumulator}

Accumulator a tag materializes for input eltype `Tin` accumulating in `Tacc`.
"""
function accumulator_type end

"Input fields a tag reads, as positions or names, in the order its accumulator binds them."
bindings(::AbstractStatistic) = (1,)

"""
    finalize(tag, a::AbstractAccumulator, ::Type{Tout}) -> Tout

Value of the statistic for one accumulator. Methods exist for every accumulator that can serve the tag.
"""
function finalize end

"Accumulator field that holds the finalized value verbatim, or `nothing`."
component_view(::AbstractStatistic, ::Type{<:AbstractAccumulator}) = nothing

"Default output eltype of a tag for input eltype `Tin`."
result_eltype(::AbstractStatistic, ::Type{Tin}) where {Tin} = Tin

"`true` when an `A` can finalize every statistic a `B` can, for identical bindings."
subsumes(::Type{A}, ::Type{A}) where {A<:AbstractAccumulator} = true
subsumes(::Type{<:AbstractAccumulator}, ::Type{<:AbstractAccumulator}) = false

"Result key of a tag; field names are appended when the binding is not the default."
function name end

_suffix(fields::Tuple, default::Tuple) =
    fields == default ? Symbol("") : Symbol(("_" * string(f) for f in fields)...)
_named(base::Symbol, fields::Tuple, default::Tuple) = Symbol(base, _suffix(fields, default))
