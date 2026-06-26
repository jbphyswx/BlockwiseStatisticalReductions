module BlockwiseStatisticalReductions

# ─────────────────────────────────────────────────────────────────────────────
# BlockwiseStatisticalReductions.jl
#
# A purpose-agnostic engine for computing statistics over N-dimensional data at many
# coarser scales efficiently, by treating every statistic as a mergeable monoid and
# reusing intermediate results across scales so the data is touched as few times as
# possible. See the docs for the theory (accumulator algebra, divisor-lattice DAG,
# sliding-window engine).
# ─────────────────────────────────────────────────────────────────────────────

# Execution-backend taxonomy
include("backends.jl")
export AbstractExecutionBackend, SerialBackend, ThreadedBackend, GPUBackend, AutoBackend
export DistributedBackend, MPIBackend, local_backend, is_distributed, resolve_backend

# Accumulator algebra
include("accumulators/interface.jl")
include("accumulators/builtin.jl")
include("accumulators/composite.jl")

# Generic block kernels (base reduction + cross-scale merge) + sliding-window engine
include("kernels/fold.jl")
include("kernels/block.jl")
include("kernels/sliding.jl")
export blockreduce!, blockreduce, coarsen!, allocate_accumulators, reduced_shape, sliding_reduce

# Divisor lattice + optimal-DAG planner
include("lattice.jl")
include("planner.jl")
export factor_shape, divides, reachable_factors
export ReductionStep, ReductionPlan, tower_plan, solver_plan
export plan_work, naive_work, n_base_passes, total_work

# Preallocated buffers + serial executor
include("buffers.jl")
include("execute.jl")
export TowerBuffers, allocate_tower, step_result, run!, execute, materialize

# Public API
include("api.jl")
export reduce_stats, MultiResResult, Tower, Sliding, factors, shapes, stat_name

# Display methods (show / summary)
include("show.jl")

export AbstractAccumulator, AbstractStatistic
export empty_acc, lift, inverse_merge, arity, is_invertible
export accumulation_eltype, accumulator_type, result_value, default_output_eltype, subsumes
export check_monoid
# Accumulator types
export CountAcc, SumAcc, MeanAcc, VarAcc, CovAcc, MinAcc, MaxAcc, RawMomentsAcc
export CompositeAccumulator, members
# Statistic tags
export Count, Sum, Mean, Var, Std, Cov, Min, Max, Moments

end # module
