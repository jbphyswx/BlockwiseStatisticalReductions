# ── Count ──────────────────────────────────────────────────────────────────────

struct CountAcc <: AbstractAccumulator
    n::Int
end
neutral(::Type{CountAcc}) = CountAcc(0)
@inline lift(::Type{CountAcc}, ::Tuple) = CountAcc(1)
@inline Base.merge(a::CountAcc, b::CountAcc) = CountAcc(a.n + b.n)
is_invertible(::Type{CountAcc}) = true
@inline unmerge(ab::CountAcc, b::CountAcc) = CountAcc(ab.n - b.n)
acc_eltype(::Type{CountAcc}) = Int
shiftable(::Type{CountAcc}) = true
@inline unshift(a::CountAcc, ::Tuple{Vararg{Real}}) = a

struct Count{F} <: AbstractStatistic end
Count(f = 1) = Count{f}()
bindings(::Count{F}) where {F} = (F,)
accumulator_type(::Count, ::Type{Tin}, ::Type{Tacc}) where {Tin,Tacc} = CountAcc
result_eltype(::Count, ::Type{Tin}) where {Tin} = Int
name(::Count{F}) where {F} = _named(:count, (F,), (1,))
@inline finalize(::Count, a::CountAcc, ::Type{Tout}) where {Tout} = Tout(a.n)
component_view(::Count, ::Type{CountAcc}) = :n

# ── Sum ────────────────────────────────────────────────────────────────────────

struct SumAcc{T} <: AbstractAccumulator
    s::T
end
neutral(::Type{SumAcc{T}}) where {T} = SumAcc(zero(T))
@inline lift(::Type{SumAcc{T}}, xs::Tuple) where {T} = SumAcc(T(xs[1]))
@inline Base.merge(a::SumAcc{T}, b::SumAcc{T}) where {T} = SumAcc(a.s + b.s)
is_invertible(::Type{<:SumAcc}) = true
@inline unmerge(ab::SumAcc{T}, b::SumAcc{T}) where {T} = SumAcc(ab.s - b.s)
acc_eltype(::Type{SumAcc{T}}) where {T} = T

struct Sum{F} <: AbstractStatistic end
Sum(f = 1) = Sum{f}()
bindings(::Sum{F}) where {F} = (F,)
accumulator_type(::Sum, ::Type{Tin}, ::Type{Tacc}) where {Tin,Tacc} = SumAcc{Tacc}
name(::Sum{F}) where {F} = _named(:sum, (F,), (1,))
@inline finalize(::Sum, a::SumAcc, ::Type{Tout}) where {Tout} = Tout(a.s)
component_view(::Sum, ::Type{<:SumAcc}) = :s

# ── Mean (n, mean): Welford lift, Chan merge; phase 1 pools (Σn, Σn·mean) ───────

struct MeanAcc{T} <: AbstractAccumulator
    n::Int
    mean::T
end
neutral(::Type{MeanAcc{T}}) where {T} = MeanAcc(0, zero(T))
@inline lift(::Type{MeanAcc{T}}, xs::Tuple) where {T} = MeanAcc(1, T(xs[1]))
@inline function Base.merge(a::MeanAcc{T}, b::MeanAcc{T}) where {T}
    a.n == 0 && return b
    b.n == 0 && return a
    n = a.n + b.n
    return MeanAcc(n, a.mean + (b.mean - a.mean) * (T(b.n) / T(n)))
end
p1init(::Type{MeanAcc{T}}) where {T} = (0, zero(T))
@inline p1lift(::Type{MeanAcc{T}}, c) where {T} = (c.n, T(c.n) * c.mean)
@inline p1merge(::Type{MeanAcc{T}}, s, t) where {T} = (s[1] + t[1], s[2] + t[2])
@inline finish(::Type{MeanAcc{T}}, s, ::Nothing, ::Nothing) where {T} = s[1] == 0 ? neutral(MeanAcc{T}) : MeanAcc(s[1], s[2] / T(s[1]))
is_invertible(::Type{<:MeanAcc}) = true
@inline function unmerge(ab::MeanAcc{T}, b::MeanAcc{T}) where {T}
    na = ab.n - b.n
    na == 0 && return neutral(MeanAcc{T})
    return MeanAcc(na, (T(ab.n) * ab.mean - T(b.n) * b.mean) / T(na))
end
acc_eltype(::Type{MeanAcc{T}}) where {T} = T
shiftable(::Type{<:MeanAcc}) = true
@inline unshift(a::MeanAcc, s::Tuple{Vararg{Real}}) = MeanAcc(a.n, a.mean + s[1])

struct Mean{F} <: AbstractStatistic end
Mean(f = 1) = Mean{f}()
bindings(::Mean{F}) where {F} = (F,)
accumulator_type(::Mean, ::Type{Tin}, ::Type{Tacc}) where {Tin,Tacc} = MeanAcc{Tacc}
result_eltype(::Mean, ::Type{Tin}) where {Tin} = ratio_eltype(Tin)
name(::Mean{F}) where {F} = _named(:mean, (F,), (1,))
@inline finalize(::Mean, a::MeanAcc, ::Type{Tout}) where {Tout} = Tout(a.mean)
@inline finalize(::Sum, a::MeanAcc, ::Type{Tout}) where {Tout} = Tout(a.mean * a.n)
@inline finalize(::Count, a::MeanAcc, ::Type{Tout}) where {Tout} = Tout(a.n)
component_view(::Mean, ::Type{<:MeanAcc}) = :mean
component_view(::Count, ::Type{<:MeanAcc}) = :n
subsumes(::Type{MeanAcc{T}}, ::Type{SumAcc{T}}) where {T} = true
subsumes(::Type{<:MeanAcc}, ::Type{CountAcc}) = true

# ── Variance (n, mean, M2): Welford lift, Chan merge; two-phase pooled combine ──

struct VarAcc{T} <: AbstractAccumulator
    n::Int
    mean::T
    M2::T
end
neutral(::Type{VarAcc{T}}) where {T} = VarAcc(0, zero(T), zero(T))
@inline lift(::Type{VarAcc{T}}, xs::Tuple) where {T} = VarAcc(1, T(xs[1]), zero(T))
@inline function Base.merge(a::VarAcc{T}, b::VarAcc{T}) where {T}
    a.n == 0 && return b
    b.n == 0 && return a
    n = a.n + b.n
    δ = b.mean - a.mean
    w = T(b.n) / T(n)
    return VarAcc(n, a.mean + δ * w, a.M2 + b.M2 + δ * δ * (T(a.n) * w))
end
phases(::Type{<:VarAcc}) = 2
p1init(::Type{VarAcc{T}}) where {T} = (0, zero(T))
@inline p1lift(::Type{VarAcc{T}}, c) where {T} = (c.n, T(c.n) * c.mean)
@inline p1merge(::Type{VarAcc{T}}, s, t) where {T} = (s[1] + t[1], s[2] + t[2])
@inline mid(::Type{VarAcc{T}}, s) where {T} = s[1] == 0 ? zero(T) : s[2] / T(s[1])
p2init(::Type{VarAcc{T}}, m) where {T} = zero(T)
@inline p2lift(::Type{VarAcc{T}}, c, m) where {T} = (d = c.mean - m; c.M2 + T(c.n) * d * d)
@inline finish(::Type{VarAcc{T}}, s1, m, s2) where {T} = VarAcc(s1[1], m, s2)
is_invertible(::Type{<:VarAcc}) = true
@inline function unmerge(ab::VarAcc{T}, b::VarAcc{T}) where {T}
    na = ab.n - b.n
    na == 0 && return neutral(VarAcc{T})
    mean = (T(ab.n) * ab.mean - T(b.n) * b.mean) / T(na)
    δ = b.mean - mean
    return VarAcc(na, mean, ab.M2 - b.M2 - δ * δ * (T(na) * T(b.n) / T(ab.n)))
end
acc_eltype(::Type{VarAcc{T}}) where {T} = T
shiftable(::Type{<:VarAcc}) = true
@inline unshift(a::VarAcc, s::Tuple{Vararg{Real}}) = VarAcc(a.n, a.mean + s[1], a.M2)

struct Var{C,F} <: AbstractStatistic end
Var(f = 1; corrected::Bool = true) = Var{corrected,f}()
struct Std{C,F} <: AbstractStatistic end
Std(f = 1; corrected::Bool = true) = Std{corrected,f}()
bindings(::Var{C,F}) where {C,F} = (F,)
bindings(::Std{C,F}) where {C,F} = (F,)
accumulator_type(::Union{Var,Std}, ::Type{Tin}, ::Type{Tacc}) where {Tin,Tacc} = VarAcc{Tacc}
result_eltype(::Union{Var,Std}, ::Type{Tin}) where {Tin} = ratio_eltype(Tin)
name(::Var{C,F}) where {C,F} = _named(:var, (F,), (1,))
name(::Std{C,F}) where {C,F} = _named(:std, (F,), (1,))
@inline _denominator(::Val{true}, n::Int, ::Type{T}) where {T} = T(n - 1)
@inline _denominator(::Val{false}, n::Int, ::Type{T}) where {T} = T(n)
@inline finalize(::Var{C}, a::VarAcc{T}, ::Type{Tout}) where {C,T,Tout} = Tout(a.M2 / _denominator(Val(C), a.n, T))
@inline finalize(::Std{C}, a::VarAcc{T}, ::Type{Tout}) where {C,T,Tout} = Tout(sqrt(a.M2 / _denominator(Val(C), a.n, T)))
@inline finalize(::Mean, a::VarAcc, ::Type{Tout}) where {Tout} = Tout(a.mean)
@inline finalize(::Sum, a::VarAcc, ::Type{Tout}) where {Tout} = Tout(a.mean * a.n)
@inline finalize(::Count, a::VarAcc, ::Type{Tout}) where {Tout} = Tout(a.n)
component_view(::Mean, ::Type{<:VarAcc}) = :mean
component_view(::Count, ::Type{<:VarAcc}) = :n
subsumes(::Type{VarAcc{T}}, ::Type{MeanAcc{T}}) where {T} = true
subsumes(::Type{VarAcc{T}}, ::Type{SumAcc{T}}) where {T} = true
subsumes(::Type{<:VarAcc}, ::Type{CountAcc}) = true

# ── Central moments to order 4 (n, mean, M2, M3, M4): Pébay merge; two-phase combine ──

struct CentralMomentsAcc{T} <: AbstractAccumulator
    n::Int
    mean::T
    M2::T
    M3::T
    M4::T
end
neutral(::Type{CentralMomentsAcc{T}}) where {T} = CentralMomentsAcc(0, zero(T), zero(T), zero(T), zero(T))
@inline lift(::Type{CentralMomentsAcc{T}}, xs::Tuple) where {T} = CentralMomentsAcc(1, T(xs[1]), zero(T), zero(T), zero(T))
@inline function Base.merge(a::CentralMomentsAcc{T}, b::CentralMomentsAcc{T}) where {T}
    a.n == 0 && return b
    b.n == 0 && return a
    na, nb = T(a.n), T(b.n)
    n = na + nb
    δ = b.mean - a.mean
    δ2 = δ * δ
    mean = a.mean + δ * (nb / n)
    M2 = a.M2 + b.M2 + δ2 * (na * nb / n)
    M3 = a.M3 + b.M3 + δ2 * δ * (na * nb * (na - nb) / (n * n)) + 3 * δ * (na * b.M2 - nb * a.M2) / n
    M4 = a.M4 + b.M4 + δ2 * δ2 * (na * nb * (na * na - na * nb + nb * nb) / (n * n * n)) +
         6 * δ2 * (na * na * b.M2 + nb * nb * a.M2) / (n * n) + 4 * δ * (na * b.M3 - nb * a.M3) / n
    return CentralMomentsAcc(a.n + b.n, mean, M2, M3, M4)
end
phases(::Type{<:CentralMomentsAcc}) = 2
p1init(::Type{CentralMomentsAcc{T}}) where {T} = (0, zero(T))
@inline p1lift(::Type{CentralMomentsAcc{T}}, c) where {T} = (c.n, T(c.n) * c.mean)
@inline p1merge(::Type{CentralMomentsAcc{T}}, s, t) where {T} = (s[1] + t[1], s[2] + t[2])
@inline mid(::Type{CentralMomentsAcc{T}}, s) where {T} = s[1] == 0 ? zero(T) : s[2] / T(s[1])
p2init(::Type{CentralMomentsAcc{T}}, m) where {T} = (zero(T), zero(T), zero(T))
@inline function p2lift(::Type{CentralMomentsAcc{T}}, c, m) where {T}
    d = c.mean - m
    d2 = d * d
    n = T(c.n)
    return (c.M2 + n * d2, c.M3 + 3 * d * c.M2 + n * d2 * d, c.M4 + 4 * d * c.M3 + 6 * d2 * c.M2 + n * d2 * d2)
end
@inline p2merge(::Type{<:CentralMomentsAcc}, s, t) = map(+, s, t)
@inline finish(::Type{CentralMomentsAcc{T}}, s1, m, s2) where {T} = CentralMomentsAcc(s1[1], m, s2[1], s2[2], s2[3])
acc_eltype(::Type{CentralMomentsAcc{T}}) where {T} = T
shiftable(::Type{<:CentralMomentsAcc}) = true
@inline unshift(a::CentralMomentsAcc, s::Tuple{Vararg{Real}}) = CentralMomentsAcc(a.n, a.mean + s[1], a.M2, a.M3, a.M4)

struct CentralMoments{K,F} <: AbstractStatistic end
CentralMoments(K::Integer, f = 1) = (2 <= K <= 4 || throw(ArgumentError("CentralMoments order must be 2, 3 or 4")); CentralMoments{Int(K),f}())
struct Skewness{F} <: AbstractStatistic end
Skewness(f = 1) = Skewness{f}()
struct Kurtosis{E,F} <: AbstractStatistic end
Kurtosis(f = 1; excess::Bool = true) = Kurtosis{excess,f}()
bindings(::CentralMoments{K,F}) where {K,F} = (F,)
bindings(::Skewness{F}) where {F} = (F,)
bindings(::Kurtosis{E,F}) where {E,F} = (F,)
accumulator_type(::Union{CentralMoments,Skewness,Kurtosis}, ::Type{Tin}, ::Type{Tacc}) where {Tin,Tacc} = CentralMomentsAcc{Tacc}
result_eltype(::Union{Skewness,Kurtosis}, ::Type{Tin}) where {Tin} = ratio_eltype(Tin)
result_eltype(::CentralMoments{K}, ::Type{Tin}) where {K,Tin} = NTuple{K - 1,ratio_eltype(Tin)}
name(::CentralMoments{K,F}) where {K,F} = _named(Symbol(:central_moments, K), (F,), (1,))
name(::Skewness{F}) where {F} = _named(:skewness, (F,), (1,))
name(::Kurtosis{E,F}) where {E,F} = _named(:kurtosis, (F,), (1,))
@inline _moments(a::CentralMomentsAcc, ::Val{2}) = (a.M2,)
@inline _moments(a::CentralMomentsAcc, ::Val{3}) = (a.M2, a.M3)
@inline _moments(a::CentralMomentsAcc, ::Val{4}) = (a.M2, a.M3, a.M4)
@inline finalize(::CentralMoments{K}, a::CentralMomentsAcc{T}, ::Type{Tout}) where {K,T,Tout<:Tuple} =
    map(m -> eltype(Tout)(m / T(a.n)), _moments(a, Val(K)))
@inline finalize(::Skewness, a::CentralMomentsAcc{T}, ::Type{Tout}) where {T,Tout} =
    Tout(sqrt(T(a.n)) * a.M3 / (a.M2 * sqrt(a.M2)))
@inline finalize(::Kurtosis{E}, a::CentralMomentsAcc{T}, ::Type{Tout}) where {E,T,Tout} =
    Tout(T(a.n) * a.M4 / (a.M2 * a.M2) - (E ? T(3) : zero(T)))
@inline finalize(s::Var, a::CentralMomentsAcc{T}, ::Type{Tout}) where {T,Tout} = finalize(s, VarAcc(a.n, a.mean, a.M2), Tout)
@inline finalize(s::Std, a::CentralMomentsAcc{T}, ::Type{Tout}) where {T,Tout} = finalize(s, VarAcc(a.n, a.mean, a.M2), Tout)
@inline finalize(::Mean, a::CentralMomentsAcc, ::Type{Tout}) where {Tout} = Tout(a.mean)
@inline finalize(::Sum, a::CentralMomentsAcc, ::Type{Tout}) where {Tout} = Tout(a.mean * a.n)
@inline finalize(::Count, a::CentralMomentsAcc, ::Type{Tout}) where {Tout} = Tout(a.n)
component_view(::Mean, ::Type{<:CentralMomentsAcc}) = :mean
component_view(::Count, ::Type{<:CentralMomentsAcc}) = :n
subsumes(::Type{CentralMomentsAcc{T}}, ::Type{VarAcc{T}}) where {T} = true
subsumes(::Type{CentralMomentsAcc{T}}, ::Type{MeanAcc{T}}) where {T} = true
subsumes(::Type{CentralMomentsAcc{T}}, ::Type{SumAcc{T}}) where {T} = true
subsumes(::Type{<:CentralMomentsAcc}, ::Type{CountAcc}) = true

# ── Raw moments (n, Σxᵏ for k = 1..K): additive merge ──────────────────────────

struct RawMomentsAcc{K,T} <: AbstractAccumulator
    n::Int
    S::NTuple{K,T}
end
neutral(::Type{RawMomentsAcc{K,T}}) where {K,T} = RawMomentsAcc{K,T}(0, ntuple(_ -> zero(T), Val(K)))
@inline function lift(::Type{RawMomentsAcc{K,T}}, xs::Tuple) where {K,T}
    x = T(xs[1])
    return RawMomentsAcc{K,T}(1, ntuple(k -> x^k, Val(K)))
end
@inline Base.merge(a::RawMomentsAcc{K,T}, b::RawMomentsAcc{K,T}) where {K,T} =
    RawMomentsAcc{K,T}(a.n + b.n, ntuple(k -> a.S[k] + b.S[k], Val(K)))
is_invertible(::Type{<:RawMomentsAcc}) = true
@inline unmerge(ab::RawMomentsAcc{K,T}, b::RawMomentsAcc{K,T}) where {K,T} =
    RawMomentsAcc{K,T}(ab.n - b.n, ntuple(k -> ab.S[k] - b.S[k], Val(K)))
acc_eltype(::Type{RawMomentsAcc{K,T}}) where {K,T} = T

struct Moments{K,F} <: AbstractStatistic end
Moments(K::Integer, f = 1) = (K >= 1 || throw(ArgumentError("Moments order must be ≥ 1")); Moments{Int(K),f}())
bindings(::Moments{K,F}) where {K,F} = (F,)
accumulator_type(::Moments{K}, ::Type{Tin}, ::Type{Tacc}) where {K,Tin,Tacc} = RawMomentsAcc{K,Tacc}
result_eltype(::Moments{K}, ::Type{Tin}) where {K,Tin} = NTuple{K,ratio_eltype(Tin)}
name(::Moments{K,F}) where {K,F} = _named(Symbol(:moments, K), (F,), (1,))
@inline finalize(::Moments{J}, a::RawMomentsAcc{K,T}, ::Type{Tout}) where {J,K,T,Tout<:Tuple} =
    ntuple(k -> eltype(Tout)(a.S[k] / T(a.n)), Val(J))
@inline finalize(::Mean, a::RawMomentsAcc{K,T}, ::Type{Tout}) where {K,T,Tout} = Tout(a.S[1] / T(a.n))
@inline finalize(::Sum, a::RawMomentsAcc, ::Type{Tout}) where {Tout} = Tout(a.S[1])
@inline finalize(::Count, a::RawMomentsAcc, ::Type{Tout}) where {Tout} = Tout(a.n)
component_view(::Count, ::Type{<:RawMomentsAcc}) = :n
subsumes(::Type{RawMomentsAcc{K,T}}, ::Type{RawMomentsAcc{J,T}}) where {K,J,T} = K >= J
subsumes(::Type{RawMomentsAcc{K,T}}, ::Type{MeanAcc{T}}) where {K,T} = true
subsumes(::Type{RawMomentsAcc{K,T}}, ::Type{SumAcc{T}}) where {K,T} = true
subsumes(::Type{<:RawMomentsAcc}, ::Type{CountAcc}) = true

# ── Extrema ────────────────────────────────────────────────────────────────────

struct MinAcc{T} <: AbstractAccumulator
    m::T
end
struct MaxAcc{T} <: AbstractAccumulator
    m::T
end
struct ExtremaAcc{T} <: AbstractAccumulator
    lo::T
    hi::T
end
neutral(::Type{MinAcc{T}}) where {T} = MinAcc(typemax(T))
neutral(::Type{MaxAcc{T}}) where {T} = MaxAcc(typemin(T))
neutral(::Type{ExtremaAcc{T}}) where {T} = ExtremaAcc(typemax(T), typemin(T))
@inline lift(::Type{MinAcc{T}}, xs::Tuple) where {T} = MinAcc(T(xs[1]))
@inline lift(::Type{MaxAcc{T}}, xs::Tuple) where {T} = MaxAcc(T(xs[1]))
@inline lift(::Type{ExtremaAcc{T}}, xs::Tuple) where {T} = (x = T(xs[1]); ExtremaAcc(x, x))
@inline _lesser(a, b) = ifelse(b < a, b, a)
@inline _greater(a, b) = ifelse(b > a, b, a)
@inline Base.merge(a::MinAcc{T}, b::MinAcc{T}) where {T} = MinAcc(_lesser(a.m, b.m))
@inline Base.merge(a::MaxAcc{T}, b::MaxAcc{T}) where {T} = MaxAcc(_greater(a.m, b.m))
@inline Base.merge(a::ExtremaAcc{T}, b::ExtremaAcc{T}) where {T} = ExtremaAcc(_lesser(a.lo, b.lo), _greater(a.hi, b.hi))
acc_eltype(::Type{MinAcc{T}}) where {T} = T
shiftable(::Type{<:MinAcc}) = true
@inline unshift(a::MinAcc, s::Tuple{Vararg{Real}}) = MinAcc(a.m + s[1])
acc_eltype(::Type{MaxAcc{T}}) where {T} = T
shiftable(::Type{<:MaxAcc}) = true
@inline unshift(a::MaxAcc, s::Tuple{Vararg{Real}}) = MaxAcc(a.m + s[1])
acc_eltype(::Type{ExtremaAcc{T}}) where {T} = T
shiftable(::Type{<:ExtremaAcc}) = true
@inline unshift(a::ExtremaAcc, s::Tuple{Vararg{Real}}) = ExtremaAcc(a.lo + s[1], a.hi + s[1])

struct Min{F} <: AbstractStatistic end
Min(f = 1) = Min{f}()
struct Max{F} <: AbstractStatistic end
Max(f = 1) = Max{f}()
struct Extrema{F} <: AbstractStatistic end
Extrema(f = 1) = Extrema{f}()
bindings(::Min{F}) where {F} = (F,)
bindings(::Max{F}) where {F} = (F,)
bindings(::Extrema{F}) where {F} = (F,)
accumulator_type(::Min, ::Type{Tin}, ::Type{Tacc}) where {Tin,Tacc} = MinAcc{Tin}
accumulator_type(::Max, ::Type{Tin}, ::Type{Tacc}) where {Tin,Tacc} = MaxAcc{Tin}
accumulator_type(::Extrema, ::Type{Tin}, ::Type{Tacc}) where {Tin,Tacc} = ExtremaAcc{Tin}
result_eltype(::Extrema, ::Type{Tin}) where {Tin} = Tuple{Tin,Tin}
name(::Min{F}) where {F} = _named(:min, (F,), (1,))
name(::Max{F}) where {F} = _named(:max, (F,), (1,))
name(::Extrema{F}) where {F} = _named(:extrema, (F,), (1,))
@inline finalize(::Min, a::MinAcc, ::Type{Tout}) where {Tout} = Tout(a.m)
@inline finalize(::Max, a::MaxAcc, ::Type{Tout}) where {Tout} = Tout(a.m)
@inline finalize(::Min, a::ExtremaAcc, ::Type{Tout}) where {Tout} = Tout(a.lo)
@inline finalize(::Max, a::ExtremaAcc, ::Type{Tout}) where {Tout} = Tout(a.hi)
@inline finalize(::Extrema, a::ExtremaAcc, ::Type{Tuple{T,T}}) where {T} = (T(a.lo), T(a.hi))
component_view(::Min, ::Type{<:MinAcc}) = :m
component_view(::Max, ::Type{<:MaxAcc}) = :m
component_view(::Min, ::Type{<:ExtremaAcc}) = :lo
component_view(::Max, ::Type{<:ExtremaAcc}) = :hi
subsumes(::Type{ExtremaAcc{T}}, ::Type{MinAcc{T}}) where {T} = true
subsumes(::Type{ExtremaAcc{T}}, ::Type{MaxAcc{T}}) where {T} = true

# ── Product sum (n, Σxy) ───────────────────────────────────────────────────────

struct ProductSumAcc{T} <: AbstractAccumulator
    n::Int
    s::T
end
arity(::Type{<:ProductSumAcc}) = 2
neutral(::Type{ProductSumAcc{T}}) where {T} = ProductSumAcc(0, zero(T))
@inline lift(::Type{ProductSumAcc{T}}, xs::Tuple) where {T} = ProductSumAcc(1, T(xs[1]) * T(xs[2]))
@inline Base.merge(a::ProductSumAcc{T}, b::ProductSumAcc{T}) where {T} = ProductSumAcc(a.n + b.n, a.s + b.s)
is_invertible(::Type{<:ProductSumAcc}) = true
@inline unmerge(ab::ProductSumAcc{T}, b::ProductSumAcc{T}) where {T} = ProductSumAcc(ab.n - b.n, ab.s - b.s)
acc_eltype(::Type{ProductSumAcc{T}}) where {T} = T

struct ProductMean{F1,F2} <: AbstractStatistic end
ProductMean(f1 = 1, f2 = 2) = ProductMean{f1,f2}()
bindings(::ProductMean{F1,F2}) where {F1,F2} = (F1, F2)
accumulator_type(::ProductMean, ::Type{Tin}, ::Type{Tacc}) where {Tin,Tacc} = ProductSumAcc{Tacc}
result_eltype(::ProductMean, ::Type{Tin}) where {Tin} = ratio_eltype(Tin)
name(::ProductMean{F1,F2}) where {F1,F2} = _named(:product_mean, (F1, F2), (1, 2))
@inline finalize(::ProductMean, a::ProductSumAcc{T}, ::Type{Tout}) where {T,Tout} = Tout(a.s / T(a.n))

# ── Covariance (n, mean1, mean2, C): Pébay merge; two-phase pooled combine ─────

struct CovAcc{T} <: AbstractAccumulator
    n::Int
    mean1::T
    mean2::T
    C::T
end
arity(::Type{<:CovAcc}) = 2
neutral(::Type{CovAcc{T}}) where {T} = CovAcc(0, zero(T), zero(T), zero(T))
@inline lift(::Type{CovAcc{T}}, xs::Tuple) where {T} = CovAcc(1, T(xs[1]), T(xs[2]), zero(T))
@inline function Base.merge(a::CovAcc{T}, b::CovAcc{T}) where {T}
    a.n == 0 && return b
    b.n == 0 && return a
    n = a.n + b.n
    δ1 = b.mean1 - a.mean1
    δ2 = b.mean2 - a.mean2
    w = T(b.n) / T(n)
    return CovAcc(n, a.mean1 + δ1 * w, a.mean2 + δ2 * w, a.C + b.C + δ1 * δ2 * (T(a.n) * w))
end
phases(::Type{<:CovAcc}) = 2
p1init(::Type{CovAcc{T}}) where {T} = (0, zero(T), zero(T))
@inline p1lift(::Type{CovAcc{T}}, c) where {T} = (c.n, T(c.n) * c.mean1, T(c.n) * c.mean2)
@inline p1merge(::Type{CovAcc{T}}, s, t) where {T} = (s[1] + t[1], s[2] + t[2], s[3] + t[3])
@inline mid(::Type{CovAcc{T}}, s) where {T} = s[1] == 0 ? (zero(T), zero(T)) : (s[2] / T(s[1]), s[3] / T(s[1]))
p2init(::Type{CovAcc{T}}, m) where {T} = zero(T)
@inline p2lift(::Type{CovAcc{T}}, c, m) where {T} = c.C + T(c.n) * (c.mean1 - m[1]) * (c.mean2 - m[2])
@inline finish(::Type{CovAcc{T}}, s1, m, s2) where {T} = CovAcc(s1[1], m[1], m[2], s2)
is_invertible(::Type{<:CovAcc}) = true
@inline function unmerge(ab::CovAcc{T}, b::CovAcc{T}) where {T}
    na = ab.n - b.n
    na == 0 && return neutral(CovAcc{T})
    mean1 = (T(ab.n) * ab.mean1 - T(b.n) * b.mean1) / T(na)
    mean2 = (T(ab.n) * ab.mean2 - T(b.n) * b.mean2) / T(na)
    return CovAcc(na, mean1, mean2, ab.C - b.C - (b.mean1 - mean1) * (b.mean2 - mean2) * (T(na) * T(b.n) / T(ab.n)))
end
acc_eltype(::Type{CovAcc{T}}) where {T} = T
shiftable(::Type{<:CovAcc}) = true
@inline unshift(a::CovAcc, s::Tuple{Vararg{Real}}) = CovAcc(a.n, a.mean1 + s[1], a.mean2 + s[2], a.C)

struct Cov{C,F1,F2} <: AbstractStatistic end
Cov(f1 = 1, f2 = 2; corrected::Bool = true) = Cov{corrected,f1,f2}()
bindings(::Cov{C,F1,F2}) where {C,F1,F2} = (F1, F2)
accumulator_type(::Cov, ::Type{Tin}, ::Type{Tacc}) where {Tin,Tacc} = CovAcc{Tacc}
result_eltype(::Cov, ::Type{Tin}) where {Tin} = ratio_eltype(Tin)
name(::Cov{C,F1,F2}) where {C,F1,F2} = _named(:cov, (F1, F2), (1, 2))
@inline finalize(::Cov{C}, a::CovAcc{T}, ::Type{Tout}) where {C,T,Tout} = Tout(a.C / _denominator(Val(C), a.n, T))
@inline finalize(::ProductMean, a::CovAcc{T}, ::Type{Tout}) where {T,Tout} = Tout(a.C / T(a.n) + a.mean1 * a.mean2)
subsumes(::Type{CovAcc{T}}, ::Type{ProductSumAcc{T}}) where {T} = true

# ── Correlation (n, mean1, mean2, M2_1, M2_2, C): two-phase pooled combine ─────

struct CorrAcc{T} <: AbstractAccumulator
    n::Int
    mean1::T
    mean2::T
    M2_1::T
    M2_2::T
    C::T
end
arity(::Type{<:CorrAcc}) = 2
neutral(::Type{CorrAcc{T}}) where {T} = CorrAcc(0, zero(T), zero(T), zero(T), zero(T), zero(T))
@inline lift(::Type{CorrAcc{T}}, xs::Tuple) where {T} = CorrAcc(1, T(xs[1]), T(xs[2]), zero(T), zero(T), zero(T))
@inline function Base.merge(a::CorrAcc{T}, b::CorrAcc{T}) where {T}
    a.n == 0 && return b
    b.n == 0 && return a
    n = a.n + b.n
    δ1 = b.mean1 - a.mean1
    δ2 = b.mean2 - a.mean2
    w = T(b.n) / T(n)
    f = T(a.n) * w
    return CorrAcc(n, a.mean1 + δ1 * w, a.mean2 + δ2 * w,
                   a.M2_1 + b.M2_1 + δ1 * δ1 * f, a.M2_2 + b.M2_2 + δ2 * δ2 * f, a.C + b.C + δ1 * δ2 * f)
end
phases(::Type{<:CorrAcc}) = 2
p1init(::Type{CorrAcc{T}}) where {T} = (0, zero(T), zero(T))
@inline p1lift(::Type{CorrAcc{T}}, c) where {T} = (c.n, T(c.n) * c.mean1, T(c.n) * c.mean2)
@inline p1merge(::Type{CorrAcc{T}}, s, t) where {T} = (s[1] + t[1], s[2] + t[2], s[3] + t[3])
@inline mid(::Type{CorrAcc{T}}, s) where {T} = s[1] == 0 ? (zero(T), zero(T)) : (s[2] / T(s[1]), s[3] / T(s[1]))
p2init(::Type{CorrAcc{T}}, m) where {T} = (zero(T), zero(T), zero(T))
@inline function p2lift(::Type{CorrAcc{T}}, c, m) where {T}
    d1 = c.mean1 - m[1]
    d2 = c.mean2 - m[2]
    n = T(c.n)
    return (c.M2_1 + n * d1 * d1, c.M2_2 + n * d2 * d2, c.C + n * d1 * d2)
end
@inline p2merge(::Type{<:CorrAcc}, s, t) = map(+, s, t)
@inline finish(::Type{CorrAcc{T}}, s1, m, s2) where {T} = CorrAcc(s1[1], m[1], m[2], s2[1], s2[2], s2[3])
acc_eltype(::Type{CorrAcc{T}}) where {T} = T
shiftable(::Type{<:CorrAcc}) = true
@inline unshift(a::CorrAcc, s::Tuple{Vararg{Real}}) = CorrAcc(a.n, a.mean1 + s[1], a.mean2 + s[2], a.M2_1, a.M2_2, a.C)

struct Corr{F1,F2} <: AbstractStatistic end
Corr(f1 = 1, f2 = 2) = Corr{f1,f2}()
bindings(::Corr{F1,F2}) where {F1,F2} = (F1, F2)
accumulator_type(::Corr, ::Type{Tin}, ::Type{Tacc}) where {Tin,Tacc} = CorrAcc{Tacc}
result_eltype(::Corr, ::Type{Tin}) where {Tin} = ratio_eltype(Tin)
name(::Corr{F1,F2}) where {F1,F2} = _named(:corr, (F1, F2), (1, 2))
@inline finalize(::Corr, a::CorrAcc{T}, ::Type{Tout}) where {T,Tout} = Tout(a.C / sqrt(a.M2_1 * a.M2_2))
@inline finalize(::Cov{C}, a::CorrAcc{T}, ::Type{Tout}) where {C,T,Tout} = Tout(a.C / _denominator(Val(C), a.n, T))
@inline finalize(::ProductMean, a::CorrAcc{T}, ::Type{Tout}) where {T,Tout} = Tout(a.C / T(a.n) + a.mean1 * a.mean2)
subsumes(::Type{CorrAcc{T}}, ::Type{CovAcc{T}}) where {T} = true
subsumes(::Type{CorrAcc{T}}, ::Type{ProductSumAcc{T}}) where {T} = true

# ── Component: any accumulator field as an output ──────────────────────────────

"`Component(tag, field)` reports the raw accumulator field `field` of `tag`'s accumulator (e.g. `:M2`, `:C`)."
struct Component{S<:AbstractStatistic,C} <: AbstractStatistic
    tag::S
end
Component(tag::AbstractStatistic, field::Symbol) = Component{typeof(tag),field}(tag)
bindings(c::Component) = bindings(c.tag)
accumulator_type(c::Component, ::Type{Tin}, ::Type{Tacc}) where {Tin,Tacc} = accumulator_type(c.tag, Tin, Tacc)
name(c::Component{S,C}) where {S,C} = Symbol(C, _suffix(bindings(c.tag), ntuple(identity, length(bindings(c.tag)))))
@inline finalize(::Component{S,C}, a::AbstractAccumulator, ::Type{Tout}) where {S,C,Tout} = Tout(getfield(a, C))
component_view(::Component{S,C}, ::Type{<:AbstractAccumulator}) where {S,C} = C
