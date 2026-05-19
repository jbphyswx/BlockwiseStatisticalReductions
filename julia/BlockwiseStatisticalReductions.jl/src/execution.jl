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

# Arguments
- `plan::ReductionPlan`: The reduction plan to execute
- `array::AbstractArray`: Input data
- `backend::AbstractExecutionBackend`: Execution backend (default: CPUBackend)
- `cache::PlanCache`: Cache for intermediate results
- `storage::AbstractStorage`: Storage backend (overrides cache storage if provided)
- `disk_spill::Bool`: Whether to spill intermediates to disk
- `disk_dir::String`: Directory for disk spill (default: tempdir)

# Returns
Vector of `ReductionResult` objects for each output node.
"""
function execute(plan::ReductionPlan, array::AbstractArray;
                 backend::AbstractExecutionBackend=CPUBackend(),
                 cache::PlanCache=PlanCache(),
                 storage::Union{AbstractStorage,Nothing}=nothing,
                 disk_spill::Bool=false,
                 disk_dir::Union{String,Nothing}=nothing)
    
    # Validate plan
    validate(plan)
    
    # Setup storage
    if storage !== nothing
        cache = PlanCache(storage)
    elseif disk_spill
        dir = disk_dir !== nothing ? disk_dir : mktempdir()
        cache = PlanCache(DiskStorage(dir))
    end
    
    # Topological sort of nodes (simple dependency tracking)
    node_order = topological_sort(plan)
    
    # Execute nodes in order
    # Use Any to handle both ReductionResult and Vector{ReductionResult} from WindowNodes
    results = Dict{UInt64,Any}()
    
    for node_id in node_order
        node = find_node(plan, node_id)
        
        # Get input(s) for this node
        input_ids = get_inputs(plan, node_id)
        
        if isempty(input_ids)
            # Input node - use the original array
            result = execute_node(node, array, backend, cache)
        elseif length(input_ids) == 1
            # Single input - pass directly
            parent_result = results[only(input_ids)]
            result = execute_node(node, parent_result, backend, cache)
        else
            # Multiple inputs - use function barrier
            result = _execute_multi_input(node, input_ids, results, backend, cache)
        end
        
        results[node_id] = result
        
        # Only cache ReductionResult (not Vector{ReductionResult} from WindowNodes)
        if result isa ReductionResult
            key = cache_key(node, size(result.data))
            store!(cache.storage, key, result)
        end
    end
    
    # Return results for output nodes (outputs should always be StatsNodes)
    output_results = ReductionResult[]
    for id in plan.outputs
        res = results[id]
        if res isa ReductionResult
            push!(output_results, res)
        else
            error("Output node $id did not return ReductionResult")
        end
    end
    return output_results
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
    # Generate window views with metadata
    iter = rolling_views(input, node.config)
    
    # Collect all window results with metadata
    results = ReductionResult[]
    for (view, meta) in iter
        push!(results, ReductionResult(view, meta))
    end
    
    return results
end

function execute_node(node::WindowNode, input::Vector{ReductionResult}, backend, cache)
    # Handle nested windows (apply to each previous result)
    all_results = ReductionResult[]
    for item in input
        if item isa ReductionResult
            arr = item.data
        else
            arr = item
        end
        
        if arr isa AbstractArray
            iter = rolling_views(arr, node.config)
            for (view, meta) in iter
                combined_meta = merge(item.metadata, meta)
                push!(all_results, ReductionResult(view, combined_meta))
            end
        end
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
    
    metadata = Dict{Symbol,Any}(
        :input_shape => size(input),
        :stat_type => typeof(stat)
    )
    
    return ReductionResult(OnlineStats.value(stat), metadata)
end

function execute_node(node::StatsNode{S}, input::Vector{ReductionResult}, backend, cache) where S
    # Multiple window results - compute stat for each and assemble into reduced array
    # Get output dimensions from first result's metadata
    out_shape = input[1].metadata[:output_shape]
    T = eltype(input[1].data)
    
    # Pre-allocate output array
    output = similar(input[1].data, T, out_shape)
    
    # Compute statistic for each window and place in output array
    for (idx, item) in enumerate(input)
        # Get output position from metadata
        pos = item.metadata[:position]
        
        # Compute statistic
        stat = create_stat(S, T)
        fit_window!(stat, item.data)
        
        # Place in output array
        output[pos...] = OnlineStats.value(stat)
    end
    
    metadata = Dict{Symbol,Any}(
        :input_shape => size(input[1].data),
        :output_shape => out_shape,
        :stat_type => S
    )
    
    return ReductionResult(output, metadata)
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
    
    metadata = Dict{Symbol,Any}(
        :arity => node.arity,
        :n_inputs => length(input)
    )
    
    return ReductionResult(result, metadata)
end

"""
    execute_node(node::UserNode, input, backend, cache)

Execute a user-defined function node.
"""
function execute_node(node::UserNode, input, backend, cache)
    data = input isa ReductionResult ? input.data : input
    result = node.f(data)
    
    metadata = Dict{Symbol,Any}(
        :user_function => nameof(node.f),
        :output_type => typeof(result)
    )
    
    return ReductionResult(result, metadata)
end

function execute_node(node::UserNode, input::Vector, backend, cache)
    # Apply to each input
    results = []
    for item in input
        data = item isa ReductionResult ? item.data : item
        result = node.f(data)
        push!(results, ReductionResult(result, Dict(:user_function => nameof(node.f))))
    end
    return results
end
