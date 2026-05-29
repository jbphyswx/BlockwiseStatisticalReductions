# API Reference: Plan Building

Functions and types for constructing reduction DAGs.

## Tower Construction (Canonical Builder)

```@docs
build_tower_plan
build_tower_plan_from_outputs
build_optimal_multires_plan
```

## Factor Schedules

```@docs
seed_factor_ladder
build_factor_schedule
```

## Low-Level Plan Building

```@docs
build_plan
next_id!
add_node!
rolling_window
stats
fork
merge_branches!
finalize_plan
```

## Types

```@docs
ReductionPlan
ReductionPlanBuilder
WindowConfig
AbstractPlanNode
WindowNode
StatsNode
TreeNode
UserNode
MergeNode
ExecutionStep
```
