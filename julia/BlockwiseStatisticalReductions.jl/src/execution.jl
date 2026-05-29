"""
    execute(plan::ReductionPlanBuilder, array::AbstractArray; kwargs...)

Execute a plan from a builder (auto-finalizes).
"""
function execute(builder::ReductionPlanBuilder, array::AbstractArray; kwargs...)
    plan = finalize_plan(builder)
    return execute(plan, array; kwargs...)
end

"""
    execute(plan::ReductionPlan, array::AbstractArray;
            backend::AbstractExecutionBackend=CPUBackend(),
            cache::PlanCache=PlanCache(),
            storage::Union{AbstractStorage,Nothing}=nothing,
            disk_spill::Bool=false,
            disk_dir::Union{String,Nothing}=nothing)

Execute a reduction plan on an array.

Uses the pre-compiled execution sequence for zero-overhead DAG traversal.
No topological sort, no Dict lookups, no linear scans at runtime.

# Returns
Vector of `ReductionResult` objects for each output node.
"""
function execute(plan::ReductionPlan, array::AbstractArray;
                 backend::AbstractExecutionBackend=CPUBackend(),
                 cache::PlanCache=PlanCache(),
                 storage::Union{AbstractStorage,Nothing}=nothing,
                 disk_spill::Bool=false,
                 disk_dir::Union{String,Nothing}=nothing)
    
    # Use pre-compiled fast path if available
    if !isempty(plan.execution_sequence)
        return _execute_compiled(plan, array, backend, cache)
    end
    
    # Fallback: compile on first use (for plans not built via finalize_plan)
    _compile_execution_sequence!(plan)
    return _execute_compiled(plan, array, backend, cache)
end

"""
    _execute_compiled(plan, array, backend, cache)

Fast execution using pre-compiled sequence. Results stored in a flat Vector
indexed by step order — no Dict, no hash, no allocation beyond output arrays.
"""
function _execute_compiled(plan::ReductionPlan, array::AbstractArray,
                           backend::AbstractExecutionBackend, cache::PlanCache)
    seq = plan.execution_sequence
    n_steps = length(seq)
    
    # Flat results vector — one slot per execution step
    results = Vector{ReductionResult}(undef, n_steps)
    
    @inbounds for step in seq
        input_idxs = step.input_indices
        
        if isempty(input_idxs)
            # Root node: operate on original array
            results[step.result_index] = execute_node(step.node, array, backend, cache)
        elseif length(input_idxs) == 1
            # Single parent: pass ReductionResult
            results[step.result_index] = execute_node(step.node, results[input_idxs[1]], backend, cache)
        else
            # Multiple parents
            inputs = ReductionResult[results[idx] for idx in input_idxs]
            results[step.result_index] = execute_node(step.node, inputs, backend, cache)
        end
    end
    
    # Collect outputs by pre-compiled indices
    return ReductionResult[results[idx] for idx in plan.output_indices]
end

"""
    ExecutionBuffers

Pre-allocated buffer set for zero-allocation repeated execution.
Created once via `allocate_buffers(plan, data)`, then reused across calls.
"""
struct ExecutionBuffers{T,N}
    buffers::Vector{Array{T,N}}
end

"""
    allocate_buffers(plan::ReductionPlan, data::AbstractArray{T,N}) where {T,N}

Pre-allocate all output buffers for a plan given an input array shape.
Returns an `ExecutionBuffers` that can be passed to `execute!` for zero-allocation execution.
"""
function allocate_buffers(plan::ReductionPlan, data::AbstractArray{T,N}) where {T,N}
    isempty(plan.execution_sequence) && _compile_execution_sequence!(plan)

    bufs = Array{T,N}[]
    shapes = Vector{NTuple{N,Int}}(undef, length(plan.execution_sequence))
    for (i, step) in enumerate(plan.execution_sequence)
        node = step.node
        if node isa ReductionNode
            shapes[i] = node.output_shape::NTuple{N,Int}
        elseif node isa SufficientStatsNode
            # Multi-output node: allocate a dummy buffer (execute! falls back
            # to allocating path for plans containing SufficientStatsNode)
            shapes[i] = node.output_shape::NTuple{N,Int}
        elseif isempty(step.input_indices)
            config = node.config
            shapes[i] = ntuple(d -> div(size(data, d), config.sizes[d]), N)
        else
            parent_shape = shapes[step.input_indices[1]]
            config = node.config
            shapes[i] = ntuple(d -> div(parent_shape[d], config.sizes[d]), N)
        end
        push!(bufs, Array{T,N}(undef, shapes[i]))
    end
    return ExecutionBuffers{T,N}(bufs)
end

"""
    execute!(plan::ReductionPlan, buffers::ExecutionBuffers{T,N}, data::AbstractArray{T,N}) where {T,N}

Zero-allocation execution using pre-allocated buffers.
After the first call to `allocate_buffers`, repeated `execute!` calls allocate nothing.

Returns a view into the buffers (no copy). Do not mutate the input `data` or buffers
between calls if you need results from a previous call.
"""
function execute!(plan::ReductionPlan, buffers::ExecutionBuffers{T,N},
                  data::AbstractArray{T,N}) where {T,N}
    seq = plan.execution_sequence
    bufs = buffers.buffers

    @inbounds for (i, step) in enumerate(seq)
        node = step.node
        input_idxs = step.input_indices
        if node isa SufficientStatsNode
            # SufficientStatsNode not supported in zero-alloc path yet;
            # fall back to allocating execute()
            return map(r -> r.data, execute(plan, data))
        end
        # Resolve the kernel: ReductionNode carries it; WindowNode defaults to mean
        kernel = node isa ReductionNode ? node.kernel : blockwise_mean!
        if isempty(input_idxs)
            kernel(bufs[i], data, node.config.sizes)
        else
            kernel(bufs[i], bufs[input_idxs[1]], node.config.sizes)
        end
    end

    return ntuple(i -> bufs[plan.output_indices[i]], length(plan.output_indices))
end

"""
    find_node(plan::ReductionPlan, id::UInt64)

Find a node by ID.
"""
function find_node(plan::ReductionPlan, id::UInt64)
    for node in plan.nodes
        if node.id == id
            return node
        end
    end
    error("Node $id not found")
end

"""
    get_inputs(plan::ReductionPlan, node_id::UInt64)

Get the input node IDs for a given node.
"""
function get_inputs(plan::ReductionPlan, node_id::UInt64)
    inputs = UInt64[]
    for (src, dsts) in plan.edges
        if node_id in dsts
            push!(inputs, src)
        end
    end
    return inputs
end

"""
    topological_sort(plan::ReductionPlan)

Simple topological sort of plan nodes.
"""
function topological_sort(plan::ReductionPlan)
    # BFS from inputs
    visited = Set{UInt64}()
    order = UInt64[]
    queue = copy(plan.inputs)
    
    while !isempty(queue)
        node_id = popfirst!(queue)
        node_id in visited && continue
        push!(visited, node_id)
        push!(order, node_id)
        
        # Add children
        if haskey(plan.edges, node_id)
            for child in plan.edges[node_id]
                push!(queue, child)
            end
        end
    end
    
    return order
end

#
# ─── ReductionNode execution (new: dispatches through node.kernel) ───────────
#

"""
    execute_node(node::ReductionNode, input::AbstractArray, backend, cache)

Execute a ReductionNode on raw input data. Allocates output and calls the
kernel function stored in the node.
"""
function execute_node(node::ReductionNode, input::AbstractArray, backend, cache)
    config = node.config
    N = ndims(input)
    out_dims = ntuple(i -> div(size(input, i), config.sizes[i]), N)
    out = similar(input, eltype(input), out_dims)
    node.kernel(out, input, config.sizes)
    return ReductionResult(out, size(input))
end

function execute_node(node::ReductionNode, input::ReductionResult, backend, cache)
    arr = input.data
    arr isa Number && return input

    config = node.config
    N = ndims(arr)
    out_dims = ntuple(i -> div(size(arr, i), config.sizes[i]), N)
    any(d -> d == 0, out_dims) && return input

    out = similar(arr, eltype(arr), out_dims)
    node.kernel(out, arr, config.sizes)
    return ReductionResult(out, size(arr))
end

#
# ─── SufficientStatsNode execution (multi-output sufficient statistics) ───────
#

"""
    execute_node(node::SufficientStatsNode, input::AbstractArray, backend, cache)

Execute a SufficientStatsNode at the base level: compute sufficient statistics
from raw data. Returns ReductionResult with NamedTuple data containing
multiple arrays (e.g., mean + M2).
"""
function execute_node(node::SufficientStatsNode, input::AbstractArray, backend, cache)
    @assert node.is_base "Base-level SufficientStatsNode expected for raw array input"
    config = node.config
    T = eltype(input)
    N = ndims(input)
    out_dims = ntuple(i -> div(size(input, i), config.sizes[i]), N)

    # Allocate output arrays for sufficient statistics
    out_mean = similar(input, T, out_dims)
    out_M2 = similar(input, T, out_dims)
    node.compute_kernel(out_mean, out_M2, input, config.sizes)

    data = (mean = out_mean, M2 = out_M2)
    return ReductionResult(data, size(input))
end

"""
    execute_node(node::SufficientStatsNode, input::ReductionResult, backend, cache)

Execute a SufficientStatsNode at a merge level: merge sufficient statistics
from a finer level.
"""
function execute_node(node::SufficientStatsNode, input::ReductionResult, backend, cache)
    @assert !node.is_base "Merge-level SufficientStatsNode expected for ReductionResult input"
    ss = input.data  # NamedTuple{(:mean, :M2), ...}
    config = node.config
    T = eltype(ss.mean)
    N = ndims(ss.mean)
    out_dims = ntuple(i -> div(size(ss.mean, i), config.sizes[i]), N)

    out_mean = similar(ss.mean, T, out_dims)
    out_M2 = similar(ss.M2, T, out_dims)
    node.merge_kernel(out_mean, out_M2, ss.mean, ss.M2, node.count_per_block, config.sizes)

    data = (mean = out_mean, M2 = out_M2)
    return ReductionResult(data, size(ss.mean))
end

#
# ─── WindowNode execution (legacy — hardcodes mean for backward compat) ──────
#

"""
    execute_node(node::WindowNode, input, backend, cache)

Execute a legacy WindowNode (always computes mean).
"""
function execute_node(node::WindowNode, input::AbstractArray, backend, cache)
    config = node.config
    output = blockwise_mean_kernel(input, config.sizes)
    return ReductionResult(output, size(input))
end

function execute_node(node::WindowNode, input::ReductionResult, backend, cache)
    arr = input.data
    arr isa Number && return input

    config = node.config
    out_dims = ntuple(i -> div(size(arr, i), config.sizes[i]), ndims(arr))
    any(d -> d == 0, out_dims) && return input

    output = blockwise_mean_kernel(arr, config.sizes)
    return ReductionResult(output, size(arr))
end

function execute_node(node::WindowNode, input::Vector{ReductionResult}, backend, cache)
    return ReductionResult[execute_node(node, item, backend, cache) for item in input]
end


"""
    _execute_multi_input(node, input_ids, results, backend, cache)

Function barrier for multiple inputs to maintain type stability.
"""
function _execute_multi_input(node, input_ids, results, backend, cache)
    # Collect inputs - this is type-unstable but isolated in function barrier
    inputs = Any[results[id] for id in input_ids]
    return execute_node(node, inputs, backend, cache)
end

"""
    execute_node(node::TreeNode, input, backend, cache)

Execute a tree reduction node.
"""
function execute_node(node::TreeNode, input::Vector, backend, cache)
    if length(input) == 0
        return nothing
    elseif length(input) == 1
        return input[1]
    end
    
    # Extract data from ReductionResult if needed
    items = [i isa ReductionResult ? i.data : i for i in input]
    
    # Determine merge operation based on item type
    if items[1] isa OnlineStat
        op = (a, b) -> merge!(deepcopy(a), b)
    elseif items[1] isa Number
        op = node.arity == 2 ? (+) : (args...) -> sum(args)
    else
        # Default: concatenate
        op = (a, b) -> [a; b]
    end
    
    result = tree_reduce_impl(items, op, backend)
    
    return ReductionResult(result, ())
end

"""
    execute_node(node::UserNode, input, backend, cache)

Execute a user-defined function node.
"""
function execute_node(node::UserNode, input, backend, cache)
    data = input isa ReductionResult ? input.data : input
    result = node.f(data)
    return ReductionResult(result, ())
end

function execute_node(node::UserNode, input::Vector, backend, cache)
    # Apply to each input
    results = ReductionResult[]
    for item in input
        data = item isa ReductionResult ? item.data : item
        result = node.f(data)
        push!(results, ReductionResult(result, ()))
    end
    return results
end
