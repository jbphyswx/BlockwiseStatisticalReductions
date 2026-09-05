"""
    Composite{M<:Tuple,B}(members::M)

Product of member accumulators over one observation of several input fields. `B` is a tuple with one
`NTuple{K,Int}` per member naming the input-field positions that member binds. Satisfies the whole
accumulator interface member-wise; its two-phase protocol fuses the phases of all members, so a
composite costs at most two passes over its children.
"""
struct Composite{M<:Tuple,B} <: AbstractAccumulator
    members::M
end

"Member accumulators of a composite."
@inline members(c::Composite) = c.members

"Per-member input-field positions of a composite type."
bindings(::Type{Composite{M,B}}) where {M,B} = B

# Member-wise operations are generated so every member type and field index is a compile-time constant.
function _member_exprs(M, member_expr)
    out = Any[]
    for k in 1:fieldcount(M)
        push!(out, member_expr(k, fieldtype(M, k)))
    end
    return out
end
_inline_tuple(exprs) = quote
    Base.@_inline_meta
    ($(exprs...),)
end
_inline_composite(M, B, exprs) = quote
    Base.@_inline_meta
    Composite{$M,$B}(($(exprs...),))
end
_phase2_members(M) = [k for k in 1:fieldcount(M) if phases(fieldtype(M, k)) == 2]

_neutral_expr(M, B) = _inline_composite(M, B, _member_exprs(M, (k, Mk) -> :(neutral($Mk))))
_lift_expr(M, B) = _inline_composite(M, B, _member_exprs(M, (k, Mk) -> :(lift($Mk, ($(map(j -> :(xs[$j]), B[k])...),)))))
_merge_expr(M, B) = _inline_composite(M, B, _member_exprs(M, (k, Mk) -> :(merge(a.members[$k], b.members[$k]))))
_unmerge_expr(M, B) = _inline_composite(M, B, _member_exprs(M, (k, Mk) -> :(unmerge(ab.members[$k], b.members[$k]))))
function _lift_skipping_expr(M, B)
    exprs = Any[]
    for k in 1:fieldcount(M)
        vals = Expr(:tuple, [:(xs[$j]) for j in B[k]]...)
        push!(exprs, :(lift_skipping($(fieldtype(M, k)), $vals)))
    end
    return _inline_composite(M, B, exprs)
end
_p1init_expr(M) = _inline_tuple(_member_exprs(M, (k, Mk) -> :(p1init($Mk))))
_p1lift_expr(M) = _inline_tuple(_member_exprs(M, (k, Mk) -> :(p1lift($Mk, c.members[$k]))))
_p1merge_expr(M) = _inline_tuple(_member_exprs(M, (k, Mk) -> :(p1merge($Mk, s[$k], t[$k]))))
_mid_expr(M) = _inline_tuple(_member_exprs(M, (k, Mk) -> phases(Mk) == 2 ? :(mid($Mk, s[$k])) : :(nothing)))
function _p2init_expr(M)
    exprs = Expr[]
    for k in _phase2_members(M)
        push!(exprs, :(p2init($(fieldtype(M, k)), m[$k])))
    end
    return _inline_tuple(exprs)
end
function _p2lift_expr(M)
    exprs = Expr[]
    for k in _phase2_members(M)
        push!(exprs, :(p2lift($(fieldtype(M, k)), c.members[$k], m[$k])))
    end
    return _inline_tuple(exprs)
end
function _p2merge_expr(M)
    exprs = Expr[]
    for (j, k) in enumerate(_phase2_members(M))
        push!(exprs, :(p2merge($(fieldtype(M, k)), s[$j], t[$j])))
    end
    return _inline_tuple(exprs)
end
function _finish_expr(M, B)
    two = _phase2_members(M)
    exprs = Expr[]
    for k in 1:fieldcount(M)
        j = findfirst(==(k), two)
        push!(exprs, j === nothing ? :(finish($(fieldtype(M, k)), s1[$k], nothing, nothing)) :
                                     :(finish($(fieldtype(M, k)), s1[$k], m[$k], s2[$j])))
    end
    return _inline_composite(M, B, exprs)
end
_finish1_expr(M, B) = _inline_composite(M, B, _member_exprs(M, (k, Mk) -> :(finish($Mk, s1[$k], nothing, nothing))))
_unshift_expr(M, B) = _inline_composite(M, B, _member_exprs(M, (k, Mk) -> :(unshift(a.members[$k], ($(map(j -> :(s[$j]), B[k])...),)))))

@generated neutral(::Type{Composite{M,B}}) where {M,B} = _neutral_expr(M, B)
@generated lift(::Type{Composite{M,B}}, xs::Tuple) where {M,B} = _lift_expr(M, B)
@generated lift_skipping(::Type{Composite{M,B}}, xs::Tuple) where {M,B} = _lift_skipping_expr(M, B)
@generated Base.merge(a::Composite{M,B}, b::Composite{M,B}) where {M,B} = _merge_expr(M, B)
@generated unmerge(ab::Composite{M,B}, b::Composite{M,B}) where {M,B} = _unmerge_expr(M, B)
@generated phases(::Type{Composite{M,B}}) where {M,B} = isempty(_phase2_members(M)) ? 1 : 2
@generated p1init(::Type{Composite{M,B}}) where {M,B} = _p1init_expr(M)
@generated p1lift(::Type{Composite{M,B}}, c) where {M,B} = _p1lift_expr(M)
@generated p1merge(::Type{Composite{M,B}}, s, t) where {M,B} = _p1merge_expr(M)
@generated mid(::Type{Composite{M,B}}, s) where {M,B} = _mid_expr(M)
@generated p2init(::Type{Composite{M,B}}, m) where {M,B} = _p2init_expr(M)
@generated p2lift(::Type{Composite{M,B}}, c, m) where {M,B} = _p2lift_expr(M)
@generated p2merge(::Type{Composite{M,B}}, s, t) where {M,B} = _p2merge_expr(M)
@generated finish(::Type{Composite{M,B}}, s1, ::Nothing, ::Nothing) where {M,B} = _finish1_expr(M, B)
@generated finish(::Type{Composite{M,B}}, s1, m, s2) where {M,B} = _finish_expr(M, B)
@generated unshift(a::Composite{M,B}, s::Tuple{Vararg{Real}}) where {M,B} = _unshift_expr(M, B)
shiftable(::Type{Composite{M,B}}) where {M,B} = all(shiftable, fieldtypes(M))
is_invertible(::Type{Composite{M,B}}) where {M,B} = all(is_invertible, fieldtypes(M))
arity(::Type{Composite{M,B}}) where {M,B} = maximum(maximum, B)

# ── Assembly: statistics × field names → composite type + routing ─────────────

_field_index(f::Integer, fieldnames::Tuple) =
    (1 <= f <= length(fieldnames) || throw(ArgumentError("field position $f out of range for $(length(fieldnames)) field(s)")); Int(f))
function _field_index(f::Symbol, fieldnames::Tuple)
    i = findfirst(==(f), fieldnames)
    i === nothing && throw(ArgumentError("unknown field $f; fields are $fieldnames"))
    return i
end
resolve_bindings(tag::AbstractStatistic, fieldnames::Tuple) = map(f -> _field_index(f, fieldnames), bindings(tag))

"""
    assemble(stats, fieldnames::Tuple, ::Type{Tin}, ::Type{Tacc})
        -> (CompositeType, routing::Tuple{Vararg{Val}}, names::Tuple{Vararg{Symbol}}, result_eltypes::Tuple)

Composite accumulator type serving every tag in `stats` over input fields `fieldnames`, with each tag
routed (by `Val` member index) to the member that finalizes it. Members with identical bindings that
another member subsumes are dropped. A `NamedTuple` of tags takes its result names from the keys, a
`Tuple` from each tag's own [`name`](@ref).
"""
assemble(stats::Tuple, fieldnames::Tuple, ::Type{Tin}, ::Type{Tacc}) where {Tin,Tacc} =
    _assemble(stats, map(name, stats), fieldnames, Tin, Tacc)
assemble(stats::NamedTuple, fieldnames::Tuple, ::Type{Tin}, ::Type{Tacc}) where {Tin,Tacc} =
    _assemble(values(stats), keys(stats), fieldnames, Tin, Tacc)

function _assemble(stats::Tuple, names::Tuple, fieldnames::Tuple, ::Type{Tin}, ::Type{Tacc}) where {Tin,Tacc}
    isempty(stats) && throw(ArgumentError("at least one statistic is required"))
    natural = map(s -> accumulator_type(s, Tin, Tacc), stats)
    bound = map(s -> resolve_bindings(s, fieldnames), stats)
    for (s, a, b) in zip(stats, natural, bound)
        length(b) == arity(a) || throw(ArgumentError("$(s) binds $(length(b)) field(s) but its accumulator $(a) has arity $(arity(a))"))
    end
    candidates = unique(collect(zip(natural, bound)))
    kept = filter(p -> !any(q -> q !== p && q[2] == p[2] && subsumes(q[1], p[1]), candidates), candidates)
    M = Tuple{(p[1] for p in kept)...}
    B = Tuple(p[2] for p in kept)
    routing = ntuple(k -> Val(_member_for(natural[k], bound[k], kept)), length(stats))
    allunique(names) || throw(ArgumentError("duplicate result names $(names); pass stats as a NamedTuple to name them"))
    outs = map(s -> result_eltype(s, Tin), stats)
    return Composite{M,B}, routing, names, outs
end

function _member_for(acc::Type, binding::Tuple, kept::Vector)
    for (i, (a, b)) in enumerate(kept)
        b == binding && subsumes(a, acc) && return i
    end
    error("internal: no composite member serves $acc bound to $binding")
end
