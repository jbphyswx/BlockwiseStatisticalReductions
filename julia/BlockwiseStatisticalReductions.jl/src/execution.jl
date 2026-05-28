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
    # Walk the sequence to determine output shapes
    shapes = Vector{NTuple{N,Int}}(undef, length(plan.execution_sequence))
    for (i, step) in enumerate(plan.execution_sequence)
        if isempty(step.input_indices)
            # Root node: output shape from applying window to input
            config = step.node.config
            shapes[i] = ntuple(d -> div(size(data, d), config.sizes[d]), N)
        else
            # Chain node: output shape from applying window to parent's output
            parent_shape = shapes[step.input_indices[1]]
            config = step.node.config
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
        input_idxs = step.input_indices
        if isempty(input_idxs)
            # Root: reduce from input data into buffer
            blockwise_mean!(bufs[i], data, step.node.config.sizes)
        else
            # Chain: reduce from parent buffer into this buffer
            blockwise_mean!(bufs[i], bufs[input_idxs[1]], step.node.config.sizes)
        end
    end
    
    # Return views of output buffers (zero-copy)
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

"""
    execute_node(node::WindowNode, input, backend, cache)

Execute a windowing node.
"""
function execute_node(node::WindowNode, input::AbstractArray, backend, cache)
    config = node.config
    output = blockwise_mean_kernel(input, config.sizes)
    return ReductionResult(output, size(input))
end

function execute_node(node::WindowNode, input::ReductionResult, backend, cache)
    arr = input.data
    
    # Handle scalar input (can't reduce further)
    if arr isa Number
        return input
    end
    
    config = node.config
    out_dims = ntuple(i -> div(size(arr, i), config.sizes[i]), ndims(arr))
    
    # Handle case where reduction produces empty output
    if any(d -> d == 0, out_dims)
        return input
    end
    
    output = blockwise_mean_kernel(arr, config.sizes)
    return ReductionResult(output, size(arr))
end

function execute_node(node::WindowNode, input::Vector{ReductionResult}, backend, cache)
    # Handle nested windows (apply to each previous result)
    all_results = ReductionResult[]
    for item in input
        # Apply window operation to this result
        result = execute_node(node, item, backend, cache)
        push!(all_results, result)
    end
    return all_results
end

"""
    execute_node(node::StatsNode, input, backend, cache)

Execute a statistics node.
"""
function execute_node(node::StatsNode{S}, input::AbstractArray, backend, cache) where S
    # Single array - compute stat directly
    # S is a Symbol from Val{S} in the node type
    stat = create_stat(S, eltype(input))
    fit_window!(stat, input)
    
    return ReductionResult(OnlineStats.value(stat), size(input))
end

function execute_node(node::StatsNode{S}, input::ReductionResult, backend, cache) where S
    # Extract array from ReductionResult
    arr = input.data
    
    # Handle scalar input (can't compute stat over single value meaningfully)
    if arr isa Number
        return input  # Return as-is
    end
    
    stat = create_stat(S, eltype(arr))
    fit_window!(stat, arr)
    
    return ReductionResult(OnlineStats.value(stat), size(arr))
end

function execute_node(node::StatsNode{S}, input::Vector{ReductionResult}, backend, cache) where S
    # Multiple window results - compute stat for each
    results = ReductionResult[]
    for item in input
        T = eltype(item.data)
        stat = create_stat(S, T)
        fit_window!(stat, item.data)
        push!(results, ReductionResult(OnlineStats.value(stat), item.shape))
    end
    return results
end

function execute_node(node::StatsNode{S}, input::Vector{Any}, backend, cache) where S
    # Handle Vector{Any} from execution engine
    # Check if all elements are ReductionResult
    if all(x -> x isa ReductionResult, input)
        return execute_node(node, ReductionResult[input...], backend, cache)
    else
        error("StatsNode received Vector{Any} with non-ReductionResult elements")
    end
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
