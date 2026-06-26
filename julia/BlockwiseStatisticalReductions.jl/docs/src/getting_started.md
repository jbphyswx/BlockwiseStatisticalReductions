# Getting Started

```julia
using BlockwiseStatisticalReductions
```

## One field, several statistics, several scales

```julia
data = randn(256, 256)
r = reduce_stats(data, [4, 8, 16]; stats = (Mean(), Var(), Min(), Max()))

factors(r)          # [(4,4), (8,8), (16,16)]  — finest first
r[(8, 8)].mean      # array of means at 8× coarsening
r((8, 8), Var())    # the variance array (callable accessor)
```

Requesting `Mean` and `Var` together costs no more than `Var` alone: both are read from a single
variance accumulator. `Count`, `Sum` and `Mean` are likewise free alongside `Var`.

## Choosing scales

```julia
reduce_stats(data, 8; stats = (Mean(),))                  # a single scale
reduce_stats(data, [4, 8, 16]; stats = (Mean(),))         # isotropic block sizes
reduce_stats(data, [(4, 4), (8, 2)]; stats = (Mean(),))   # per-dimension (anisotropic)

# leave a dimension unreduced with factor 1
reduce_stats(randn(64, 64, 10), [(8, 8, 1)]; stats = (Mean(),))

# a full tower: finest block, per-level multipliers, coarsest block
reduce_stats(data, Tower(base_factor = 2, steps = [2, 3], maxfactor = 64); stats = (Mean(), Std()))
```

## Covariance (two fields)

```julia
x = randn(128, 128); y = randn(128, 128)
rc = reduce_stats(x, y, [8, 16]; stats = (Cov(),))
rc[(8, 8)].cov
```

## Overlapping (sliding) windows

```julia
# 16×16 windows every 4 cells
rs = reduce_stats(data, [Sliding((16, 16); stride = (4, 4))]; stats = (Mean(), Var()))
rs[(16, 16)].var
```

With `stride == window` (the default) a `Sliding` reduction is exactly the non-overlapping blockwise
reduction.

## Backends

```julia
using OhMyThreads                              # enables ThreadedBackend
reduce_stats(data, [4, 8]; stats = (Var(),), backend = ThreadedBackend())

using Distributed, SharedArrays                # enables DistributedBackend
reduce_stats(data, [4, 8]; stats = (Var(),), backend = DistributedBackend())
```

## Zero-allocation repeated execution

```julia
plan = tower_plan(size(data); base_factor = (2, 2), steps = ([2], [2]), maxfactor = (64, 64))
buf  = allocate_tower(plan, VarAcc{Float64})
run!(buf, plan, data)                          # 0 bytes at steady state
materialize(step_result(buf, plan.output_steps[end]), Var(), Float64)
```

## A custom statistic

```julia
import BlockwiseStatisticalReductions as BSR
struct GeoMeanAcc{T} <: BSR.AbstractAccumulator
    n::Int
    logsum::T
end
BSR.empty_acc(::Type{GeoMeanAcc{T}}) where {T} = GeoMeanAcc(0, zero(T))
BSR.lift(::Type{GeoMeanAcc{T}}, x) where {T} = GeoMeanAcc(1, log(T(x)))
Base.merge(a::GeoMeanAcc, b::GeoMeanAcc) = GeoMeanAcc(a.n + b.n, a.logsum + b.logsum)
struct GeoMean <: BSR.AbstractStatistic end
BSR.accumulator_type(::GeoMean, ::Type{T}) where {T} = GeoMeanAcc{BSR.accumulation_eltype(T)}
BSR.result_value(::GeoMean, a::GeoMeanAcc{T}, ::Type{O}) where {T,O} = O(exp(a.logsum / a.n))
BSR.stat_name(::GeoMean) = :geomean

reduce_stats(abs.(data) .+ 1, [4, 8]; stats = (GeoMean(),))
```

`check_monoid(GeoMeanAcc{Float64})` checks the monoid laws on random data.
