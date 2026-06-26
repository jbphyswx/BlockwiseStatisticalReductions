# ─────────────────────────────────────────────────────────────────────────────
# Composite accumulator: several accumulators over the same field(s), as one accumulator
# ─────────────────────────────────────────────────────────────────────────────
#
# A `CompositeAccumulator` packs several member accumulators that all read the SAME field(s)
# (the same "binding") and combines them into one accumulator. Because it itself satisfies the
# accumulator interface (`empty_acc`/`lift`/`merge`/…), every kernel and the whole multi-scale
# tower stay generic: a node's buffer is `Array{CompositeAccumulator{…}}`, computed and merged
# in a single pass over the data, yet each member stays its own concrete isbits accumulator.
#
# Statistics that read *different* fields (e.g. a covariance pair vs a univariate variance)
# have different bindings and live in different composites / towers — that separation is made by
# the request layer (see the public API), not here.

"""
    CompositeAccumulator{T<:Tuple}(members)

An accumulator composed of several member accumulators that share the same field binding and
arity. Implements the full accumulator interface by acting member-wise, so it flows through the
kernels and the multi-scale merge tree like any single accumulator. isbits whenever its members
are. Use [`members`](@ref) to access the tuple of member accumulators.
"""
struct CompositeAccumulator{T<:Tuple} <: AbstractAccumulator
    members::T
end

"""
    members(c::CompositeAccumulator) -> Tuple

The tuple of member accumulators inside a composite.
"""
@inline members(c::CompositeAccumulator) = c.members

@inline empty_acc(::Type{CompositeAccumulator{T}}) where {T} =
    CompositeAccumulator(map(empty_acc, fieldtypes(T)))
@inline lift(::Type{CompositeAccumulator{T}}, x) where {T} =
    CompositeAccumulator(map(A -> lift(A, x), fieldtypes(T)))
@inline lift(::Type{CompositeAccumulator{T}}, x, y) where {T} =
    CompositeAccumulator(map(A -> lift(A, x, y), fieldtypes(T)))
@inline Base.merge(a::CompositeAccumulator{T}, b::CompositeAccumulator{T}) where {T} =
    CompositeAccumulator(map(merge, a.members, b.members))

arity(::Type{CompositeAccumulator{T}}) where {T} = arity(fieldtype(T, 1))
is_invertible(::Type{CompositeAccumulator{T}}) where {T} = all(is_invertible, fieldtypes(T))
@inline inverse_merge(ab::CompositeAccumulator{T}, b::CompositeAccumulator{T}) where {T} =
    CompositeAccumulator(map(inverse_merge, ab.members, b.members))

# ── Building the minimal member set from a list of accumulator types ───────────

"""
    minimal_accumulator_set(acc_types) -> Vector{DataType}

Reduce a collection of accumulator types to the minimal set that still [`subsumes`](@ref) all of
them: drop any type strictly subsumed by another (e.g. a `MeanAcc` when a `VarAcc` is present).
Build-time helper (not type-stable / not hot path).
"""
function minimal_accumulator_set(acc_types)
    uniq = unique(collect(acc_types))
    keep = DataType[]
    for a in uniq
        if !any(b -> b !== a && subsumes(b, a), uniq)
            push!(keep, a)
        end
    end
    return keep
end

"""
    member_for(tag, member_types, Tin) -> Int

Index of the member accumulator (within a composite's `member_types`) that can finalize `tag`,
preferring an exact match, else any member that [`subsumes`](@ref) the tag's natural accumulator.
Returns `0` if none qualifies. Build-time helper.
"""
function member_for(tag::AbstractStatistic, member_types, ::Type{Tin}) where {Tin}
    natural = accumulator_type(tag, Tin)
    for (i, m) in enumerate(member_types)
        m === natural && return i
    end
    for (i, m) in enumerate(member_types)
        subsumes(m, natural) && return i
    end
    return 0
end
