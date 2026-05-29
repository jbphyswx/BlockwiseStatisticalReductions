# API Reference: Execution

Functions for executing reduction plans and managing buffers.

## Plan Execution

```@docs
execute
execute!
allocate_buffers
```

## Types

```@docs
ReductionResult
ExecutionBuffers
AbstractExecutionBackend
CPUBackend
GPUBackend
DistributedBackend
```

## Storage and Caching

```@docs
AbstractStorage
MemoryStorage
DiskStorage
PlanCache
```

## Buffer Pool

```@docs
BufferPool
LevelBufferPool
acquire!
release!
with_buffer!
register_level!
acquire_level!
release_level!
create_buffer_pool_for_factors
```

## Internal Execution

```@docs
execute_node
topological_sort
find_node
get_inputs
```
