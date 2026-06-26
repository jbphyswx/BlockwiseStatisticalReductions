# API Reference

```@meta
CurrentModule = BlockwiseStatisticalReductions
```

## High-level

```@docs
reduce_stats
MultiResResult
Tower
Sliding
factors
shapes
```

## Statistics

```@docs
Count
Sum
Mean
Var
Std
Cov
Min
Max
Moments
stat_name
```

## Accumulator algebra

```@docs
AbstractAccumulator
AbstractStatistic
empty_acc
lift
inverse_merge
arity
is_invertible
accumulation_eltype
accumulator_type
result_value
subsumes
check_monoid
CompositeAccumulator
members
```

## Planning

```@docs
ReductionStep
ReductionPlan
tower_plan
solver_plan
reachable_factors
factor_shape
divides
plan_work
naive_work
n_base_passes
total_work
```

## Execution

```@docs
TowerBuffers
allocate_tower
run!
execute
step_result
materialize
blockreduce
blockreduce!
coarsen!
sliding_reduce
allocate_accumulators
reduced_shape
```

## Backends

```@docs
AbstractExecutionBackend
SerialBackend
ThreadedBackend
GPUBackend
DistributedBackend
MPIBackend
AutoBackend
local_backend
is_distributed
resolve_backend
```
