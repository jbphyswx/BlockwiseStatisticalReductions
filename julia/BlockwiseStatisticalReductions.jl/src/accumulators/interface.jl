# ─────────────────────────────────────────────────────────────────────────────
# The mergeable-monoid accumulator interface
# ─────────────────────────────────────────────────────────────────────────────
#
# Every statistic BSR computes is a *mergeable monoid over sufficient statistics*.
# An accumulator is an immutable, isbits struct carrying the sufficient statistics of
# a set of observations. It supports:
#
#   empty_acc(Acc)      -> the identity accumulator (monoid identity)
#   lift(Acc, x[, y])   -> the accumulator of a single observation
#   merge(a, b)         -> combine two accumulators (ASSOCIATIVE + COMMUTATIVE)
#
# and, optionally, for accumulators that form a commutative *group* (used by the
# summed-area-table sliding path):
#
#   is_invertible(Acc)  -> Bool
#   inverse_merge(ab, b) -> a   such that merge(a, b) == ab
#
# A *statistic tag* (e.g. `Mean()`, `Var()`) is the user-facing request. It maps to an
# accumulator type and finalizes it to a reported value:
#
#   accumulator_type(tag, Tin)        -> the accumulator type to materialize
#   result_value(tag, acc, Tout)      -> the finalized statistic, in output eltype Tout
#   default_output_eltype(tag, Tin)   -> default output eltype (Tin for most; Int for Count)
#   subsumes(AccA, AccB)              -> AccA can finalize everything AccB can (optimization)
#
# This split is what makes the package both type-stable (a fixed concrete accumulator
# flows through every kernel) and extensible (a user adds a statistic with a struct +
# a handful of methods, zero edits to BSR).

"""
    AbstractAccumulator

Supertype for all sufficient-statistic accumulators. Concrete subtypes must be immutable
and isbits, and must implement [`empty_acc`](@ref), [`lift`](@ref) and `Base.merge`.
"""
abstract type AbstractAccumulator end

"""
    AbstractStatistic

Supertype for user-facing statistic tags (e.g. `Mean`, `Var`). A tag selects an accumulator
type via [`accumulator_type`](@ref) and finalizes it via [`result_value`](@ref).
"""
abstract type AbstractStatistic end

# ── Accumulator interface (methods a custom accumulator implements) ───────────

"""
    empty_acc(::Type{Acc}) -> Acc

The identity accumulator: `merge(empty_acc(Acc), a) == a == merge(a, empty_acc(Acc))`.
"""
function empty_acc end

"""
    lift(::Type{Acc}, x)        # arity-1 statistics
    lift(::Type{Acc}, x, y)     # arity-2 statistics (e.g. covariance)

The accumulator of a single observation. `x`/`y` are raw data elements; the accumulator's
fields are stored in its accumulation eltype.
"""
function lift end

"""
    arity(::Type{Acc}) -> Int

Number of input fields an accumulator consumes per observation (`1` for most, `2` for
covariance-like statistics). Defaults to `1`.
"""
arity(::Type{<:AbstractAccumulator}) = 1
arity(a::AbstractAccumulator) = arity(typeof(a))

"""
    is_invertible(::Type{Acc}) -> Bool

`true` if the accumulator forms a commutative group, i.e. [`inverse_merge`](@ref) is defined.
Enables the O(1) summed-area-table sliding-window path. Defaults to `false`.
"""
is_invertible(::Type{<:AbstractAccumulator}) = false
is_invertible(a::AbstractAccumulator) = is_invertible(typeof(a))

"""
    inverse_merge(ab::Acc, b::Acc) -> a

Remove a disjoint sub-population: returns `a` such that `merge(a, b) == ab`. Only defined when
[`is_invertible`](@ref) is `true`.
"""
function inverse_merge end

# ── Numeric widening rule ─────────────────────────────────────────────────────

"""
    accumulation_eltype(::Type{Tin}) -> Type

The eltype in which moment accumulation happens. Float32 inputs accumulate in Float64 (and
Float16 in Float32) to avoid catastrophic cancellation; the reported output is narrowed back to
the input eltype unless overridden. Integers accumulate in Float64.
"""
accumulation_eltype(::Type{T}) where {T<:AbstractFloat} = T
accumulation_eltype(::Type{Float16}) = Float32
accumulation_eltype(::Type{Float32}) = Float64
accumulation_eltype(::Type{T}) where {T<:Integer} = Float64
accumulation_eltype(::Type{Bool}) = Float64
accumulation_eltype(::Type{T}) where {T} = float(T)

# ── Statistic-tag interface (methods a custom tag implements) ─────────────────

"""
    accumulator_type(tag::AbstractStatistic, ::Type{Tin}) -> Type{<:AbstractAccumulator}

The accumulator type to materialize for `tag` over input eltype `Tin`.
"""
function accumulator_type end

"""
    result_value(tag::AbstractStatistic, acc::AbstractAccumulator, ::Type{Tout})

Finalize `acc` into the statistic named by `tag`, in output eltype `Tout`. Methods exist for
every `(tag, acc)` pair the tag can be computed from (including accumulators that *subsume* the
tag's natural accumulator).
"""
function result_value end

"""
    default_output_eltype(tag::AbstractStatistic, ::Type{Tin}) -> Type

Default output eltype for `tag` given input eltype `Tin`. Defaults to `Tin`.
"""
default_output_eltype(::AbstractStatistic, ::Type{Tin}) where {Tin} = Tin

"""
    subsumes(::Type{AccA}, ::Type{AccB}) -> Bool

`true` if an `AccA` carries enough information to finalize any statistic that an `AccB` could
(e.g. a variance accumulator subsumes a mean accumulator). Used to materialize the minimal set
of accumulators per field. Reflexive; defaults to `false` for distinct types.
"""
subsumes(::Type{A}, ::Type{A}) where {A<:AbstractAccumulator} = true
subsumes(::Type{<:AbstractAccumulator}, ::Type{<:AbstractAccumulator}) = false

"""
    stat_arity(tag::AbstractStatistic) -> Int

Number of input fields the statistic binds to (1 or 2). Derived from its accumulator type via a
representative eltype; defaults via `arity(accumulator_type(tag, Float64))`.
"""
stat_arity(tag::AbstractStatistic) = arity(accumulator_type(tag, Float64))

# ── check_monoid: verify a (custom) accumulator obeys the monoid/group laws ────

"""
    check_monoid(::Type{Acc}; samples=randn(64), atol=…, rtol=…) -> Bool

Verify that accumulator type `Acc` obeys the monoid laws (identity, commutativity,
associativity) and, when [`is_invertible`](@ref), the group inverse law, on `samples`. Intended
for tests and for validating user-defined accumulators. Returns `true` or throws an
informative error. For arity-2 accumulators pass `samples` as a vector of `(x, y)` tuples.
"""
function check_monoid(::Type{Acc};
                      samples = randn(64),
                      atol::Real = 0,
                      rtol::Real = 1e-8) where {Acc<:AbstractAccumulator}
    lifted = [_lift_sample(Acc, s) for s in samples]
    e = empty_acc(Acc)
    # Effective absolute floor: tiny residuals against an exact-zero field cannot satisfy a
    # relative tolerance, so floor `atol` at `rtol * (scale of the accumulator fields)`.
    scale = maximum(_acc_scale, lifted; init = one(Float64))
    eff_atol = max(atol, rtol * scale)
    _approx(p, q) = _accs_approx(p, q; atol = eff_atol, rtol = rtol)

    # Identity
    for a in lifted
        _approx(merge(e, a), a) || error("check_monoid($Acc): left identity failed")
        _approx(merge(a, e), a) || error("check_monoid($Acc): right identity failed")
    end

    # Commutativity
    for a in lifted, b in lifted
        _approx(merge(a, b), merge(b, a)) ||
            error("check_monoid($Acc): commutativity failed")
    end

    # Associativity
    a, b, c = lifted[begin], lifted[begin+1], lifted[begin+2]
    _approx(merge(merge(a, b), c), merge(a, merge(b, c))) ||
        error("check_monoid($Acc): associativity failed")

    # Fold-equivalence: merging in any grouping equals folding left-to-right
    full = foldl(merge, lifted)
    half = length(lifted) ÷ 2
    grouped = merge(foldl(merge, lifted[1:half]), foldl(merge, lifted[half+1:end]))
    _approx(full, grouped) || error("check_monoid($Acc): fold/merge-tree equivalence failed")

    # Group inverse (if claimed)
    if is_invertible(Acc)
        ab = merge(a, b)
        _approx(inverse_merge(ab, b), a) ||
            error("check_monoid($Acc): inverse_merge(merge(a,b), b) != a")
    end
    return true
end

_lift_sample(::Type{Acc}, s) where {Acc} =
    s isa Tuple ? lift(Acc, s...) : lift(Acc, s)

# Magnitude scale of an accumulator's fields (max abs over float fields, recursing into tuples
# and nested accumulators such as `BundleAcc`).
_value_scale(x::AbstractFloat) = abs(float(x))
_value_scale(x::AbstractAccumulator) = _acc_scale(x)
_value_scale(x::Tuple) = isempty(x) ? 0.0 : maximum(_value_scale, x)
_value_scale(::Any) = 0.0

function _acc_scale(a::AbstractAccumulator)
    s = 0.0
    for f in fieldnames(typeof(a))
        s = max(s, _value_scale(getfield(a, f)))
    end
    return s
end

# Field-wise approximate comparison of two accumulators. Handles Int, float, tuple, and nested
# accumulator fields (e.g. `BundleAcc`).
_value_approx(x::Integer, y::Integer; atol, rtol) = x == y
_value_approx(x::AbstractAccumulator, y::AbstractAccumulator; atol, rtol) =
    _accs_approx(x, y; atol = atol, rtol = rtol)
_value_approx(x::Tuple, y::Tuple; atol, rtol) =
    all(_value_approx(p, q; atol = atol, rtol = rtol) for (p, q) in zip(x, y))
_value_approx(x, y; atol, rtol) = isapprox(x, y; atol = atol, rtol = rtol)

function _accs_approx(a::Acc, b::Acc; atol::Real, rtol::Real) where {Acc<:AbstractAccumulator}
    for f in fieldnames(Acc)
        _value_approx(getfield(a, f), getfield(b, f); atol = atol, rtol = rtol) || return false
    end
    return true
end
