module BlockwiseStatisticalReductions

using StatsBase: StatsBase
using OnlineStats: OnlineStats
using RollingFunctions: RollingFunctions

# Stdlib imports (no Project.toml entry needed)
using Serialization: Serialization
using Statistics: Statistics
using Distributed: Distributed

export WindowConfig, ReductionPlan, ReductionResult
export AbstractExecutionBackend, CPUBackend, DistributedBackend, GPUBackend
export AbstractStorage, MemoryStorage, DiskStorage, PlanCache
export rolling_views, tiled_blocks, validate_window_config
export execute, build_plan
export stats, rolling_window, tree_reduce
export fork, merge_branches!, parallel_reduce
export tiled_stats, tiled_stats_merge

include("types.jl")
include("backends.jl")
include("storage.jl")
include("cache.jl")
include("windows.jl")
include("statistics/stats.jl")
include("plan.jl")
include("execution.jl")

end
