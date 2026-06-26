# Basic usage: several statistics at several scales in one pass.
#
#   julia --project=examples examples/basic_usage.jl

using BlockwiseStatisticalReductions
using Random: Random

Random.seed!(1)
data = randn(256, 256)

# Mean, variance, and extrema at 4×, 8×, 16× coarsening — computed together, reusing intermediates.
r = reduce_stats(data, [4, 8, 16]; stats = (Mean(), Var(), Min(), Max()))

@show r                       # rich display: resolutions, statistics, shapes
for f in factors(r)           # finest first
    nt = r[f]
    println("factor $f → $(size(nt.mean)) :  mean[1,1]=$(round(nt.mean[1,1]; digits=4)), ",
            "var[1,1]=$(round(nt.var[1,1]; digits=4))")
end

# Callable accessor for a single statistic array:
mean8 = r((8, 8), Mean())
println("\nmean at 8×: ", size(mean8), " array")

# Anisotropic / per-dimension factors (factor 1 leaves a dimension unreduced):
r2 = reduce_stats(randn(128, 64, 8), [(8, 8, 1), (16, 16, 2)]; stats = (Mean(),))
@show factors(r2)

# Inspect the reduction plan (the DAG that gets executed) directly:
plan = tower_plan((256, 256); base_factor = (2, 2), steps = ([2], [2]), maxfactor = (64, 64))
display(plan)                 # shows nodes, parents, windows, and work vs naive
