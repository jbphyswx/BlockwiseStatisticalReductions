# ─────────────────────────────────────────────────────────────────────────────
# Built-in accumulators and statistic tags
# ─────────────────────────────────────────────────────────────────────────────
#
# All accumulators are immutable and isbits (so an `Array{Acc}` is a dense, contiguous,
# GPU-uploadable buffer and `merge` stays in registers). The numerically-stable merge math
# is Welford (lift) + Chan (variance) + Pebay (covariance); raw moments are stored as
# additive power sums Σxᵏ so their merge is *exact* addition.

# Eltype chosen for sum/moment accumulation (wider than Float32; Float64 for ints).
_sum_eltype(::Type{Tin}) where {Tin} = accumulation_eltype(Tin)

# ═══════════════════════════════════════════════════════════════════════════════
# Count
# ═══════════════════════════════════════════════════════════════════════════════

"Accumulator counting observations."
struct CountAcc <: AbstractAccumulator
    n::Int
end
empty_acc(::Type{CountAcc}) = CountAcc(0)
@inline lift(::Type{CountAcc}, x) = CountAcc(1)
@inline Base.merge(a::CountAcc, b::CountAcc) = CountAcc(a.n + b.n)
is_invertible(::Type{CountAcc}) = true
@inline inverse_merge(ab::CountAcc, b::CountAcc) = CountAcc(ab.n - b.n)

"`Count()` — the number of observations per block."
struct Count <: AbstractStatistic end
accumulator_type(::Count, ::Type{Tin}) where {Tin} = CountAcc
default_output_eltype(::Count, ::Type{Tin}) where {Tin} = Int
@inline result_value(::Count, a::CountAcc, ::Type{Tout}) where {Tout} = Tout(a.n)

# ═══════════════════════════════════════════════════════════════════════════════
# Sum
# ═══════════════════════════════════════════════════════════════════════════════

"Accumulator holding a running sum."
struct SumAcc{T} <: AbstractAccumulator
    s::T
end
empty_acc(::Type{SumAcc{T}}) where {T} = SumAcc(zero(T))
@inline lift(::Type{SumAcc{T}}, x) where {T} = SumAcc(T(x))
@inline Base.merge(a::SumAcc{T}, b::SumAcc{T}) where {T} = SumAcc(a.s + b.s)
is_invertible(::Type{<:SumAcc}) = true
@inline inverse_merge(ab::SumAcc{T}, b::SumAcc{T}) where {T} = SumAcc(ab.s - b.s)

"`Sum()` — the sum over each block."
struct Sum <: AbstractStatistic end
accumulator_type(::Sum, ::Type{Tin}) where {Tin} = SumAcc{_sum_eltype(Tin)}
@inline result_value(::Sum, a::SumAcc{T}, ::Type{Tout}) where {T,Tout} = Tout(a.s)

# ═══════════════════════════════════════════════════════════════════════════════
# Mean  (n, mean) — Welford lift, Chan merge
# ═══════════════════════════════════════════════════════════════════════════════

"Accumulator holding count and running mean."
struct MeanAcc{T} <: AbstractAccumulator
    n::Int
    mean::T
end
empty_acc(::Type{MeanAcc{T}}) where {T} = MeanAcc(0, zero(T))
@inline lift(::Type{MeanAcc{T}}, x) where {T} = MeanAcc(1, T(x))
@inline function Base.merge(a::MeanAcc{T}, b::MeanAcc{T}) where {T}
    a.n == 0 && return b
    b.n == 0 && return a
    n = a.n + b.n
    δ = b.mean - a.mean
    return MeanAcc(n, a.mean + δ * (T(b.n) / T(n)))
end
is_invertible(::Type{<:MeanAcc}) = true
@inline function inverse_merge(ab::MeanAcc{T}, b::MeanAcc{T}) where {T}
    na = ab.n - b.n
    na == 0 && return empty_acc(MeanAcc{T})
    return MeanAcc(na, (T(ab.n) * ab.mean - T(b.n) * b.mean) / T(na))
end

"`Mean()` — the arithmetic mean over each block."
struct Mean <: AbstractStatistic end
accumulator_type(::Mean, ::Type{Tin}) where {Tin} = MeanAcc{accumulation_eltype(Tin)}
@inline result_value(::Mean, a::MeanAcc{T}, ::Type{Tout}) where {T,Tout} = Tout(a.mean)

# Mean/Sum/Count derivable from a MeanAcc:
@inline result_value(::Sum, a::MeanAcc{T}, ::Type{Tout}) where {T,Tout} = Tout(a.mean * a.n)
@inline result_value(::Count, a::MeanAcc, ::Type{Tout}) where {Tout} = Tout(a.n)
subsumes(::Type{MeanAcc{T}}, ::Type{SumAcc{T}}) where {T} = true
subsumes(::Type{MeanAcc{T}}, ::Type{CountAcc}) where {T} = true

# ═══════════════════════════════════════════════════════════════════════════════
# Variance / Std  (n, mean, M2) — Welford lift, Chan merge
# ═══════════════════════════════════════════════════════════════════════════════

"Accumulator holding count, mean, and M2 = Σ(x-mean)² (the variance numerator)."
struct VarAcc{T} <: AbstractAccumulator
    n::Int
    mean::T
    M2::T
end
empty_acc(::Type{VarAcc{T}}) where {T} = VarAcc(0, zero(T), zero(T))
@inline lift(::Type{VarAcc{T}}, x) where {T} = VarAcc(1, T(x), zero(T))
@inline function Base.merge(a::VarAcc{T}, b::VarAcc{T}) where {T}
    a.n == 0 && return b
    b.n == 0 && return a
    n = a.n + b.n
    δ = b.mean - a.mean
    mean = a.mean + δ * (T(b.n) / T(n))
    M2 = a.M2 + b.M2 + δ * δ * (T(a.n) * T(b.n) / T(n))   # Chan
    return VarAcc(n, mean, M2)
end
is_invertible(::Type{<:VarAcc}) = true
@inline function inverse_merge(ab::VarAcc{T}, b::VarAcc{T}) where {T}
    na = ab.n - b.n
    na == 0 && return empty_acc(VarAcc{T})
    mean = (T(ab.n) * ab.mean - T(b.n) * b.mean) / T(na)
    δ = b.mean - mean
    M2 = ab.M2 - b.M2 - δ * δ * (T(na) * T(b.n) / T(ab.n))
    return VarAcc(na, mean, M2)
end

"`Var(; corrected=true)` — variance (sample `corrected=true`, ÷(n-1); population `false`, ÷n)."
struct Var{corrected} <: AbstractStatistic end
Var(; corrected::Bool = true) = Var{corrected}()
"`Std(; corrected=true)` — standard deviation (see [`Var`](@ref) for `corrected`)."
struct Std{corrected} <: AbstractStatistic end
Std(; corrected::Bool = true) = Std{corrected}()

accumulator_type(::Var, ::Type{Tin}) where {Tin} = VarAcc{accumulation_eltype(Tin)}
accumulator_type(::Std, ::Type{Tin}) where {Tin} = VarAcc{accumulation_eltype(Tin)}

@inline _var_denom(::Var{true}, n, ::Type{T}) where {T} = T(n - 1)
@inline _var_denom(::Var{false}, n, ::Type{T}) where {T} = T(n)
@inline result_value(s::Var, a::VarAcc{T}, ::Type{Tout}) where {T,Tout} =
    Tout(a.M2 / _var_denom(s, a.n, T))
@inline result_value(::Std{C}, a::VarAcc{T}, ::Type{Tout}) where {C,T,Tout} =
    Tout(sqrt(result_value(Var{C}(), a, T)))

# Mean/Sum/Count derivable from a VarAcc:
@inline result_value(::Mean, a::VarAcc{T}, ::Type{Tout}) where {T,Tout} = Tout(a.mean)
@inline result_value(::Sum, a::VarAcc{T}, ::Type{Tout}) where {T,Tout} = Tout(a.mean * a.n)
@inline result_value(::Count, a::VarAcc, ::Type{Tout}) where {Tout} = Tout(a.n)
subsumes(::Type{VarAcc{T}}, ::Type{MeanAcc{T}}) where {T} = true
subsumes(::Type{VarAcc{T}}, ::Type{SumAcc{T}}) where {T} = true
subsumes(::Type{VarAcc{T}}, ::Type{CountAcc}) where {T} = true

# ═══════════════════════════════════════════════════════════════════════════════
# Covariance  (n, meanx, meany, C) — Pebay merge   [arity 2]
# ═══════════════════════════════════════════════════════════════════════════════

"Accumulator holding count, both means, and C = Σ(x-meanx)(y-meany) (the covariance numerator)."
struct CovAcc{T} <: AbstractAccumulator
    n::Int
    meanx::T
    meany::T
    C::T
end
arity(::Type{<:CovAcc}) = 2
empty_acc(::Type{CovAcc{T}}) where {T} = CovAcc(0, zero(T), zero(T), zero(T))
@inline lift(::Type{CovAcc{T}}, x, y) where {T} = CovAcc(1, T(x), T(y), zero(T))
@inline function Base.merge(a::CovAcc{T}, b::CovAcc{T}) where {T}
    a.n == 0 && return b
    b.n == 0 && return a
    n = a.n + b.n
    δx = b.meanx - a.meanx
    δy = b.meany - a.meany
    f = T(a.n) * T(b.n) / T(n)
    return CovAcc(n,
                  a.meanx + δx * (T(b.n) / T(n)),
                  a.meany + δy * (T(b.n) / T(n)),
                  a.C + b.C + δx * δy * f)               # Pebay
end
is_invertible(::Type{<:CovAcc}) = true
@inline function inverse_merge(ab::CovAcc{T}, b::CovAcc{T}) where {T}
    na = ab.n - b.n
    na == 0 && return empty_acc(CovAcc{T})
    meanx = (T(ab.n) * ab.meanx - T(b.n) * b.meanx) / T(na)
    meany = (T(ab.n) * ab.meany - T(b.n) * b.meany) / T(na)
    δx = b.meanx - meanx
    δy = b.meany - meany
    C = ab.C - b.C - δx * δy * (T(na) * T(b.n) / T(ab.n))
    return CovAcc(na, meanx, meany, C)
end

"`Cov(; corrected=true)` — covariance of a field pair (see [`Var`](@ref) for `corrected`)."
struct Cov{corrected} <: AbstractStatistic end
Cov(; corrected::Bool = true) = Cov{corrected}()
accumulator_type(::Cov, ::Type{Tin}) where {Tin} = CovAcc{accumulation_eltype(Tin)}
@inline _cov_denom(::Cov{true}, n, ::Type{T}) where {T} = T(n - 1)
@inline _cov_denom(::Cov{false}, n, ::Type{T}) where {T} = T(n)
@inline result_value(s::Cov, a::CovAcc{T}, ::Type{Tout}) where {T,Tout} =
    Tout(a.C / _cov_denom(s, a.n, T))

# ═══════════════════════════════════════════════════════════════════════════════
# Min / Max  (not invertible)
# ═══════════════════════════════════════════════════════════════════════════════

"Accumulator holding a running minimum."
struct MinAcc{T} <: AbstractAccumulator
    m::T
end
empty_acc(::Type{MinAcc{T}}) where {T} = MinAcc(typemax(T))
@inline lift(::Type{MinAcc{T}}, x) where {T} = MinAcc(T(x))
@inline Base.merge(a::MinAcc{T}, b::MinAcc{T}) where {T} = MinAcc(ifelse(b.m < a.m, b.m, a.m))

"Accumulator holding a running maximum."
struct MaxAcc{T} <: AbstractAccumulator
    m::T
end
empty_acc(::Type{MaxAcc{T}}) where {T} = MaxAcc(typemin(T))
@inline lift(::Type{MaxAcc{T}}, x) where {T} = MaxAcc(T(x))
@inline Base.merge(a::MaxAcc{T}, b::MaxAcc{T}) where {T} = MaxAcc(ifelse(b.m > a.m, b.m, a.m))

"`Min()` — the minimum over each block."
struct Min <: AbstractStatistic end
"`Max()` — the maximum over each block."
struct Max <: AbstractStatistic end
accumulator_type(::Min, ::Type{Tin}) where {Tin} = MinAcc{Tin}
accumulator_type(::Max, ::Type{Tin}) where {Tin} = MaxAcc{Tin}
@inline result_value(::Min, a::MinAcc{T}, ::Type{Tout}) where {T,Tout} = Tout(a.m)
@inline result_value(::Max, a::MaxAcc{T}, ::Type{Tout}) where {T,Tout} = Tout(a.m)

# ═══════════════════════════════════════════════════════════════════════════════
# Raw moments up to order K  (n, NTuple{K,T} of power sums Σxᵏ) — exact additive merge
# ═══════════════════════════════════════════════════════════════════════════════

"Accumulator holding count and the additive power sums `S[k] = Σ xᵏ`, k=1..K."
struct RawMomentsAcc{K,T} <: AbstractAccumulator
    n::Int
    S::NTuple{K,T}
end
empty_acc(::Type{RawMomentsAcc{K,T}}) where {K,T} = RawMomentsAcc{K,T}(0, ntuple(_ -> zero(T), Val(K)))
@inline function lift(::Type{RawMomentsAcc{K,T}}, x) where {K,T}
    xT = T(x)
    return RawMomentsAcc{K,T}(1, ntuple(k -> xT^k, Val(K)))
end
@inline Base.merge(a::RawMomentsAcc{K,T}, b::RawMomentsAcc{K,T}) where {K,T} =
    RawMomentsAcc{K,T}(a.n + b.n, ntuple(k -> a.S[k] + b.S[k], Val(K)))
is_invertible(::Type{<:RawMomentsAcc}) = true
@inline inverse_merge(ab::RawMomentsAcc{K,T}, b::RawMomentsAcc{K,T}) where {K,T} =
    RawMomentsAcc{K,T}(ab.n - b.n, ntuple(k -> ab.S[k] - b.S[k], Val(K)))

"`Moments(K)` — the raw moments `E[xᵏ]`, k=1..K, returned per block as an `NTuple{K}`."
struct Moments{K} <: AbstractStatistic end
Moments(K::Integer) = Moments{Int(K)}()
accumulator_type(::Moments{K}, ::Type{Tin}) where {K,Tin} = RawMomentsAcc{K,accumulation_eltype(Tin)}
@inline result_value(::Moments{K}, a::RawMomentsAcc{K,T}, ::Type{Tout}) where {K,T,Tout} =
    ntuple(k -> Tout(a.S[k] / a.n), Val(K))

# Mean (k=1) / Count derivable from raw moments; a higher-order acc subsumes a lower one:
@inline result_value(::Mean, a::RawMomentsAcc{K,T}, ::Type{Tout}) where {K,T,Tout} = Tout(a.S[1] / a.n)
@inline result_value(::Count, a::RawMomentsAcc, ::Type{Tout}) where {Tout} = Tout(a.n)
@inline result_value(::Moments{J}, a::RawMomentsAcc{K,T}, ::Type{Tout}) where {J,K,T,Tout} =
    ntuple(k -> Tout(a.S[k] / a.n), Val(J))   # J <= K
subsumes(::Type{RawMomentsAcc{K,T}}, ::Type{RawMomentsAcc{J,T}}) where {K,J,T} = K >= J
subsumes(::Type{RawMomentsAcc{K,T}}, ::Type{MeanAcc{T}}) where {K,T} = true
subsumes(::Type{RawMomentsAcc{K,T}}, ::Type{CountAcc}) where {K,T} = true
