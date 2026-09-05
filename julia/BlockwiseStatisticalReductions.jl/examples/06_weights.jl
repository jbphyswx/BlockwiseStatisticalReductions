# Weighted statistics, and dropping gaps in the data.
using BlockwiseStatisticalReductions
using Random

Random.seed!(6)
x = randn(64, 64, 20)

# Cell thickness varies with height: a per-axis factor, never materialized as a 3-D array.
dz = [fill(20.0, 5); fill(40.0, 5); fill(80.0, 10)]
r = blockstats(x, [(8, 8, 20)]; stats = (Mean(), Var()), weights = (nothing, nothing, dz))
println("thickness-weighted mean: ", r[(8, 8, 20)].mean[1, 1])

# The total weight is a statistic of its own.
rw = blockstats(x, [(8, 8, 20)]; stats = (m = Mean(), W = Component(Mean(), :W)),
                weights = (nothing, nothing, dz))
println("total weight per tile:    ", rw[(8, 8, 20)].W[1, 1], "  (= 8 * 8 * sum(dz) = ", 64 * sum(dz), ")")

# Reliability weights use a different denominator from frequency weights.
w = rand(64, 64, 20) .+ 0.5
freq = blockstats(x, [(8, 8, 20)]; stats = (v = Var(; corrected = :frequency),), weights = w)
rel  = blockstats(x, [(8, 8, 20)]; stats = (v = Var(; corrected = :reliability),), weights = w)
println("frequency vs reliability variance: ", freq[(8, 8, 20)].v[1, 1], "  ", rel[(8, 8, 20)].v[1, 1])

# Gaps. The skip is per statistic: a hole in `x` does not remove that element from `Mean(:y)`.
xn = copy(x)
xn[3, 3, 1] = NaN
kept = blockstats(xn, [(8, 8, 20)]; stats = (n = Count(), m = Mean()), skipnan = true)
println("observations in the first tile: ", kept[(8, 8, 20)].n[1, 1], " of ", 8 * 8 * 20)
