# Defining your own statistic: an isbits accumulator + a few methods, no changes to the package.
# Here: the geometric mean (a mergeable monoid over log-sums).
#
#   julia --project=examples examples/custom_statistic.jl

using BlockwiseStatisticalReductions
import BlockwiseStatisticalReductions as BSR
using Random: Random
using Statistics: Statistics

# 1) The accumulator: immutable + isbits. Sufficient statistic = (count, Σ log x).
struct GeoMeanAcc{T} <: BSR.AbstractAccumulator
    n::Int
    logsum::T
end
BSR.empty_acc(::Type{GeoMeanAcc{T}}) where {T} = GeoMeanAcc(0, zero(T))
BSR.lift(::Type{GeoMeanAcc{T}}, x) where {T} = GeoMeanAcc(1, log(T(x)))
Base.merge(a::GeoMeanAcc, b::GeoMeanAcc) = GeoMeanAcc(a.n + b.n, a.logsum + b.logsum)

# 2) The tag: maps to the accumulator, and finalizes it.
struct GeoMean <: BSR.AbstractStatistic end
BSR.accumulator_type(::GeoMean, ::Type{Tin}) where {Tin} = GeoMeanAcc{BSR.accumulation_eltype(Tin)}
BSR.result_value(::GeoMean, a::GeoMeanAcc{T}, ::Type{Tout}) where {T,Tout} = Tout(exp(a.logsum / a.n))
BSR.stat_name(::GeoMean) = :geomean

# 3) Verify it obeys the monoid laws, then use it at every scale — in any backend.
@assert check_monoid(GeoMeanAcc{Float64}; samples = abs.(randn(64)) .+ 0.1)

Random.seed!(3)
data = abs.(randn(128, 128)) .+ 0.5            # positive field
r = reduce_stats(data, [4, 8, 16]; stats = (GeoMean(), Mean()))

# cross-check against a brute-force geometric mean of one block
block = data[1:4, 1:4]
println("geomean[1,1] @4×: ", round(r[(4, 4)].geomean[1, 1]; digits = 5),
        "   brute: ", round(exp(Statistics.mean(log.(block))); digits = 5))
println("(geometric mean ≤ arithmetic mean: ",
        all(r[(4, 4)].geomean .<= r[(4, 4)].mean .+ 1e-9), ")")
