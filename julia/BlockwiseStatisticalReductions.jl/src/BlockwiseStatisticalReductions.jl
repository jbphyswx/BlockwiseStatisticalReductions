module BlockwiseStatisticalReductions

include("backends.jl")

include("statistics/interface.jl")
include("statistics/builtin.jl")
include("statistics/weighted.jl")
include("statistics/composite.jl")
include("statistics/checks.jl")

include("storage/accumulator_array.jl")

include("geometry/window.jl")
include("geometry/coordinates.jl")

include("scales/spec.jl")
include("scales/resolve.jl")

include("planner/graph.jl")
include("planner/cost.jl")
include("planner/candidates.jl")
include("planner/solve.jl")
include("planner/explain.jl")

include("kernels/boxfold.jl")
include("kernels/scan.jl")
include("kernels/fused.jl")
include("kernels/finalize.jl")

include("execute/workspace.jl")
include("execute/run.jl")

include("api/results.jl")
include("api/blockstats.jl")
include("api/show.jl")

export blockstats, blockstats!, prepare, explain
export ScaleResults, windows, scales, geometry
export Count, Sum, Mean, Var, Std, CentralMoments, Skewness, Kurtosis, Moments, Min, Max, Extrema,
       ProductMean, Cov, Corr, Component
export ScaleSet, Sizes, Dyadic, Smooth, Every, Divisors, Fixed, Subsample, Length
export Tiled, Stride, Overlap, Dense, Anchors, Spread, Isotropic, Product, Zip
export Truncate, Partial, Centered, Strict, Regular, Edges

end
