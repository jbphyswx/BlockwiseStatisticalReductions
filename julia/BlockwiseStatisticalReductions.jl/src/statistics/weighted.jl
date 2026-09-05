# Weighted accumulators: West's weighted Welford lift with the Chan/Pébay merge run on weight totals
# instead of counts. A weighted accumulator binds one field more than its unweighted counterpart — the
# weight, last. Weights must be finite and non-negative; a zero weight contributes nothing.

# ── Weighted mean (ΣW, mean) ───────────────────────────────────────────────────

struct WMeanAcc{T} <: AbstractAccumulator
    W::T
    mean::T
end
arity(::Type{<:WMeanAcc}) = 2
neutral(::Type{WMeanAcc{T}}) where {T} = WMeanAcc(zero(T), zero(T))
@inline lift(::Type{WMeanAcc{T}}, xs::Tuple) where {T} = WMeanAcc(T(xs[2]), T(xs[1]))
@inline function Base.merge(a::WMeanAcc{T}, b::WMeanAcc{T}) where {T}
    iszero(a.W) && return b
    iszero(b.W) && return a
    W = a.W + b.W
    return WMeanAcc(W, a.mean + (b.mean - a.mean) * (b.W / W))
end
p1init(::Type{WMeanAcc{T}}) where {T} = (zero(T), zero(T))
@inline p1lift(::Type{WMeanAcc{T}}, c) where {T} = (c.W, c.W * c.mean)
@inline p1merge(::Type{WMeanAcc{T}}, s, t) where {T} = (s[1] + t[1], s[2] + t[2])
@inline finish(::Type{WMeanAcc{T}}, s, ::Nothing, ::Nothing) where {T} =
    iszero(s[1]) ? neutral(WMeanAcc{T}) : WMeanAcc(s[1], s[2] / s[1])
acc_eltype(::Type{WMeanAcc{T}}) where {T} = T
shiftable(::Type{<:WMeanAcc}) = true
@inline unshift(a::WMeanAcc, s::Tuple{Vararg{Real}}) = WMeanAcc(a.W, a.mean + s[1])

# ── Weighted variance (ΣW, ΣW², mean, M2) ──────────────────────────────────────

struct WVarAcc{T} <: AbstractAccumulator
    W::T
    W2::T
    mean::T
    M2::T
end
arity(::Type{<:WVarAcc}) = 2
neutral(::Type{WVarAcc{T}}) where {T} = WVarAcc(zero(T), zero(T), zero(T), zero(T))
@inline lift(::Type{WVarAcc{T}}, xs::Tuple) where {T} = (w = T(xs[2]); WVarAcc(w, w * w, T(xs[1]), zero(T)))
@inline function Base.merge(a::WVarAcc{T}, b::WVarAcc{T}) where {T}
    iszero(a.W) && return b
    iszero(b.W) && return a
    W = a.W + b.W
    δ = b.mean - a.mean
    f = b.W / W
    return WVarAcc(W, a.W2 + b.W2, a.mean + δ * f, a.M2 + b.M2 + δ * δ * (a.W * f))
end
phases(::Type{<:WVarAcc}) = 2
p1init(::Type{WVarAcc{T}}) where {T} = (zero(T), zero(T), zero(T))
@inline p1lift(::Type{WVarAcc{T}}, c) where {T} = (c.W, c.W2, c.W * c.mean)
@inline p1merge(::Type{WVarAcc{T}}, s, t) where {T} = (s[1] + t[1], s[2] + t[2], s[3] + t[3])
@inline mid(::Type{WVarAcc{T}}, s) where {T} = iszero(s[1]) ? zero(T) : s[3] / s[1]
p2init(::Type{WVarAcc{T}}, m) where {T} = zero(T)
@inline p2lift(::Type{WVarAcc{T}}, c, m) where {T} = (d = c.mean - m; c.M2 + c.W * d * d)
@inline finish(::Type{WVarAcc{T}}, s1, m, s2) where {T} = WVarAcc(s1[1], s1[2], m, s2)
acc_eltype(::Type{WVarAcc{T}}) where {T} = T
shiftable(::Type{<:WVarAcc}) = true
@inline unshift(a::WVarAcc, s::Tuple{Vararg{Real}}) = WVarAcc(a.W, a.W2, a.mean + s[1], a.M2)
subsumes(::Type{WVarAcc{T}}, ::Type{WMeanAcc{T}}) where {T} = true

# ── Weighted covariance (ΣW, ΣW², mean1, mean2, C) ─────────────────────────────

struct WCovAcc{T} <: AbstractAccumulator
    W::T
    W2::T
    mean1::T
    mean2::T
    C::T
end
arity(::Type{<:WCovAcc}) = 3
neutral(::Type{WCovAcc{T}}) where {T} = WCovAcc(zero(T), zero(T), zero(T), zero(T), zero(T))
@inline lift(::Type{WCovAcc{T}}, xs::Tuple) where {T} = (w = T(xs[3]); WCovAcc(w, w * w, T(xs[1]), T(xs[2]), zero(T)))
@inline function Base.merge(a::WCovAcc{T}, b::WCovAcc{T}) where {T}
    iszero(a.W) && return b
    iszero(b.W) && return a
    W = a.W + b.W
    δ1 = b.mean1 - a.mean1
    δ2 = b.mean2 - a.mean2
    f = b.W / W
    return WCovAcc(W, a.W2 + b.W2, a.mean1 + δ1 * f, a.mean2 + δ2 * f, a.C + b.C + δ1 * δ2 * (a.W * f))
end
phases(::Type{<:WCovAcc}) = 2
p1init(::Type{WCovAcc{T}}) where {T} = (zero(T), zero(T), zero(T), zero(T))
@inline p1lift(::Type{WCovAcc{T}}, c) where {T} = (c.W, c.W2, c.W * c.mean1, c.W * c.mean2)
@inline p1merge(::Type{WCovAcc{T}}, s, t) where {T} = (s[1] + t[1], s[2] + t[2], s[3] + t[3], s[4] + t[4])
@inline mid(::Type{WCovAcc{T}}, s) where {T} = iszero(s[1]) ? (zero(T), zero(T)) : (s[3] / s[1], s[4] / s[1])
p2init(::Type{WCovAcc{T}}, m) where {T} = zero(T)
@inline p2lift(::Type{WCovAcc{T}}, c, m) where {T} = c.C + c.W * (c.mean1 - m[1]) * (c.mean2 - m[2])
@inline finish(::Type{WCovAcc{T}}, s1, m, s2) where {T} = WCovAcc(s1[1], s1[2], m[1], m[2], s2)
acc_eltype(::Type{WCovAcc{T}}) where {T} = T
shiftable(::Type{<:WCovAcc}) = true
@inline unshift(a::WCovAcc, s::Tuple{Vararg{Real}}) = WCovAcc(a.W, a.W2, a.mean1 + s[1], a.mean2 + s[2], a.C)

# ── Weighted correlation (ΣW, ΣW², both means, both M2, C) ─────────────────────

struct WCorrAcc{T} <: AbstractAccumulator
    W::T
    W2::T
    mean1::T
    mean2::T
    M2_1::T
    M2_2::T
    C::T
end
arity(::Type{<:WCorrAcc}) = 3
neutral(::Type{WCorrAcc{T}}) where {T} = WCorrAcc(zero(T), zero(T), zero(T), zero(T), zero(T), zero(T), zero(T))
@inline lift(::Type{WCorrAcc{T}}, xs::Tuple) where {T} =
    (w = T(xs[3]); WCorrAcc(w, w * w, T(xs[1]), T(xs[2]), zero(T), zero(T), zero(T)))
@inline function Base.merge(a::WCorrAcc{T}, b::WCorrAcc{T}) where {T}
    iszero(a.W) && return b
    iszero(b.W) && return a
    W = a.W + b.W
    δ1 = b.mean1 - a.mean1
    δ2 = b.mean2 - a.mean2
    f = b.W / W
    g = a.W * f
    return WCorrAcc(W, a.W2 + b.W2, a.mean1 + δ1 * f, a.mean2 + δ2 * f,
                    a.M2_1 + b.M2_1 + δ1 * δ1 * g, a.M2_2 + b.M2_2 + δ2 * δ2 * g, a.C + b.C + δ1 * δ2 * g)
end
phases(::Type{<:WCorrAcc}) = 2
p1init(::Type{WCorrAcc{T}}) where {T} = (zero(T), zero(T), zero(T), zero(T))
@inline p1lift(::Type{WCorrAcc{T}}, c) where {T} = (c.W, c.W2, c.W * c.mean1, c.W * c.mean2)
@inline p1merge(::Type{WCorrAcc{T}}, s, t) where {T} = (s[1] + t[1], s[2] + t[2], s[3] + t[3], s[4] + t[4])
@inline mid(::Type{WCorrAcc{T}}, s) where {T} = iszero(s[1]) ? (zero(T), zero(T)) : (s[3] / s[1], s[4] / s[1])
p2init(::Type{WCorrAcc{T}}, m) where {T} = (zero(T), zero(T), zero(T))
@inline function p2lift(::Type{WCorrAcc{T}}, c, m) where {T}
    d1 = c.mean1 - m[1]
    d2 = c.mean2 - m[2]
    return (c.M2_1 + c.W * d1 * d1, c.M2_2 + c.W * d2 * d2, c.C + c.W * d1 * d2)
end
@inline p2merge(::Type{<:WCorrAcc}, s, t) = map(+, s, t)
@inline finish(::Type{WCorrAcc{T}}, s1, m, s2) where {T} = WCorrAcc(s1[1], s1[2], m[1], m[2], s2[1], s2[2], s2[3])
acc_eltype(::Type{WCorrAcc{T}}) where {T} = T
shiftable(::Type{<:WCorrAcc}) = true
@inline unshift(a::WCorrAcc, s::Tuple{Vararg{Real}}) =
    WCorrAcc(a.W, a.W2, a.mean1 + s[1], a.mean2 + s[2], a.M2_1, a.M2_2, a.C)
subsumes(::Type{WCorrAcc{T}}, ::Type{WCovAcc{T}}) where {T} = true

# ── The tag ────────────────────────────────────────────────────────────────────

"""
    Weighted(stat, W)

`stat` over weighted observations, binding input field `W` as the weight after the fields `stat` binds
itself. The `weights` keyword of [`prepare`](@ref) builds these, binding the weight source it resolves.
"""
struct Weighted{S<:AbstractStatistic,W} <: AbstractStatistic
    stat::S
end
Weighted(stat::AbstractStatistic, w) = Weighted{typeof(stat),w}(stat)
bindings(w::Weighted{S,W}) where {S,W} = (bindings(w.stat)..., W)
name(w::Weighted) = name(w.stat)
result_eltype(w::Weighted, ::Type{Tin}) where {Tin} = result_eltype(w.stat, Tin)
accumulator_type(w::Weighted, ::Type{Tin}, ::Type{Tacc}) where {Tin,Tacc} = weighted_accumulator(w.stat, Tacc)

"""
    weighted_accumulator(stat, ::Type{Tacc}) -> Type{<:AbstractAccumulator}

Accumulator the weighted form of `stat` uses. Statistics whose weighted form is not defined throw here,
so a weighted request never silently computes an unweighted quantity.
"""
weighted_accumulator(s::AbstractStatistic, ::Type) =
    throw(ArgumentError("$(name(s)) has no weighted form; request it in a call without `weights`"))
weighted_accumulator(::Union{Mean,Sum}, ::Type{T}) where {T} = WMeanAcc{T}
weighted_accumulator(::Union{Var,Std}, ::Type{T}) where {T} = WVarAcc{T}
weighted_accumulator(::Union{Cov,ProductMean}, ::Type{T}) where {T} = WCovAcc{T}
weighted_accumulator(::Corr, ::Type{T}) where {T} = WCorrAcc{T}
weighted_accumulator(c::Component, ::Type{T}) where {T} = weighted_accumulator(c.tag, T)

"""
    weight_stat(stat, W::Int) -> AbstractStatistic

`stat` rewritten to read input field `W` as a weight. `Count` is returned unchanged: the number of
observations in a window does not depend on their weights.
"""
weight_stat(s::AbstractStatistic, W::Int) = Weighted(s, W)
weight_stat(s::Count, ::Int) = s

const _WSingle = Union{WMeanAcc,WVarAcc}
const _WPair = Union{WCovAcc,WCorrAcc}
const _WAcc = Union{_WSingle,_WPair}

# Effective sample size: the total weight, less one weight for a corrected estimate. Frequency weights
# count repeated observations, so one whole observation comes off; reliability weights measure relative
# precision, so the correction is the total's own second moment. Unit weights make the two identical.
@inline _wdenominator(::Val{false}, a::_WAcc) = a.W
@inline _wdenominator(::Val{true}, a::_WAcc) = max(a.W - one(a.W), zero(a.W))
@inline _wdenominator(::Val{:frequency}, a::_WAcc) = max(a.W - one(a.W), zero(a.W))
@inline _wdenominator(::Val{:reliability}, a::_WAcc) = a.W - a.W2 / a.W

@inline finalize(w::Weighted, a::AbstractAccumulator, ::Type{Tout}) where {Tout} = _wfinalize(w.stat, a, Tout)
@inline _wfinalize(::Mean, a::_WSingle, ::Type{Tout}) where {Tout} = Tout(_mean_or_nan(a.mean, a.W))
@inline _wfinalize(::Sum, a::_WSingle, ::Type{Tout}) where {Tout} = Tout(a.W * a.mean)
@inline _wfinalize(::Var{C}, a::WVarAcc, ::Type{Tout}) where {C,Tout} = Tout(a.M2 / _wdenominator(Val(C), a))
@inline _wfinalize(::Std{C}, a::WVarAcc, ::Type{Tout}) where {C,Tout} = Tout(sqrt(a.M2 / _wdenominator(Val(C), a)))
@inline _wfinalize(::Cov{C}, a::_WPair, ::Type{Tout}) where {C,Tout} = Tout(a.C / _wdenominator(Val(C), a))
@inline _wfinalize(::Corr, a::WCorrAcc, ::Type{Tout}) where {Tout} = Tout(a.C / sqrt(a.M2_1 * a.M2_2))
@inline _wfinalize(::ProductMean, a::_WPair, ::Type{Tout}) where {Tout} = Tout(a.C / a.W + a.mean1 * a.mean2)
@inline _wfinalize(::Component{S,F}, a::_WAcc, ::Type{Tout}) where {S,F,Tout} = Tout(getfield(a, F))
