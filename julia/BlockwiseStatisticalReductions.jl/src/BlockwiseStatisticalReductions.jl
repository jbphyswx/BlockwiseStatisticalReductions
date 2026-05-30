module BlockwiseStatisticalReductions

using StatsBase: StatsBase
using OnlineStats: OnlineStats
using RollingFunctions: RollingFunctions
using LoopVectorization: LoopVectorization

# Stdlib imports (no Project.toml entry needed)
using Serialization: Serialization
using Statistics: Statistics
using Distributed: Distributed

export WindowConfig, ReductionPlan, ReductionResult, ReductionNode, SufficientStatsNode
export AbstractExecutionBackend, CPUBackend, DistributedBackend, GPUBackend
export AbstractStorage, MemoryStorage, DiskStorage, PlanCache
export rolling_views, tiled_blocks, validate_window_config
export execute, build_plan
export stats, rolling_window, tree_reduce
export fork, merge_branches!, parallel_reduce
export tiled_stats, tiled_stats_merge

# New mergeable statistics
export MergeableStatistic, VarianceAccumulator, CovarianceAccumulator, RawMomentsAccumulator
export merge, merge_many, merge_all
export fit!, nobs

# Product coarsening
export product_mean, product_moments, product_variance
export JointMomentsResult
export covariance_from_moments, variance_from_moments
export blockwise_product_mean, blockwise_product_moments

# Multi-resolution planning
export build_optimal_multires_plan, multiresolution_stats, multiresolution_products
export build_multires_plan_groups

# Tower construction
export seed_factor_ladder, build_factor_schedule
export build_tower_plan, build_tower_plan_from_outputs

# Buffer pool
export BufferPool, LevelBufferPool
export acquire!, release!, with_buffer!
export register_level!, acquire_level!, release_level!
export create_buffer_pool_for_factors

# Zero-allocation execution
export ExecutionBuffers, allocate_buffers, execute!


# Public API
export blockwise_stats, blockwise_mean, blockwise_variance, blockwise_std
export blockwise_covariance, blockwise_moments

# Canonical in-place kernels
export blockwise_mean!, blockwise_variance!, blockwise_mean_variance!
export blockwise_min!, blockwise_max!, blockwise_product_mean!
export blockwise_joint_moments!

# Merge kernels (hierarchical sufficient-statistics composition)
export blockwise_sum!
export blockwise_mean_M2!, blockwise_merge_mean_M2!
export blockwise_mean_M2_M3!, blockwise_merge_mean_M2_M3!
export blockwise_mean_C!, blockwise_merge_covariance!
export blockwise_merge_raw_moments!
export variance_from_M2, std_from_M2, skewness_from_M2_M3, covariance_from_C

include("types.jl")
include("backends.jl")
include("storage.jl")
include("cache.jl")

# Canonical blockwise kernels — loaded early so all other files can call them
include("kernels/blockwise_kernels.jl")
include("kernels/merge_kernels.jl")

include("windows.jl")
include("statistics/stats.jl")
include("statistics/core.jl")
include("statistics/parallel_merge.jl")
include("statistics/product_reductions.jl")
include("plan.jl")
include("tower.jl")
include("tower_groups.jl")
include("execution/buffer_pool.jl")
include("hybrid_mode.jl")
include("public_api.jl")

# Additional SIMD kernel variants
include("kernels/simd_kernels.jl")

# Distributed scheduling
include("execution/distributed_scheduler.jl")

include("execution.jl")

end
