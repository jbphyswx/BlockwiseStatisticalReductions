"""
    ReductionPlanBuilder

Fluent API builder for constructing reduction plans.
"""
mutable struct ReductionPlanBuilder
    plan::ReductionPlan
    current_node_id::Union{UInt64,Nothing}
    counter::UInt64
    input_shape::Union{NTuple,Nothing}
end

ReductionPlanBuilder() = ReductionPlanBuilder(ReductionPlan(), nothing, UInt64(0), nothing)

"""
    build_plan(input_shape::NTuple)

Start building a reduction plan for an array of given shape.
"""
function build_plan(input_shape::NTuple)
    builder = ReductionPlanBuilder()
    builder.input_shape = input_shape
    return builder
end

"""
    next_id!(builder::ReductionPlanBuilder)

Generate the next unique node ID.
"""
function next_id!(builder::ReductionPlanBuilder)
    builder.counter += 1
    return builder.counter
end

"""
    add_node!(builder::ReductionPlanBuilder, node::AbstractPlanNode)

Add a node to the plan and connect it to the previous node.
"""
function add_node!(builder::ReductionPlanBuilder, node::AbstractPlanNode)
    push!(builder.plan.nodes, node)
    
    # Connect from previous node if exists
    if builder.current_node_id !== nothing
        edges = get!(builder.plan.edges, builder.current_node_id, UInt64[])
        push!(edges, node.id)
    else
        # This is an input node
        push!(builder.plan.inputs, node.id)
    end
    
    builder.current_node_id = node.id
    return builder
end

"""
    rolling_window(builder::ReductionPlanBuilder, sizes::NTuple{D,Int}; 
                   strides=sizes, padding=:valid) where D

Add a rolling/tiled window node to the plan.
"""
function rolling_window(builder::ReductionPlanBuilder, sizes::NTuple{D,Int}; 
                        strides=sizes, padding=:valid) where D
    config = WindowConfig(sizes, strides, padding)
    node = WindowNode(config, next_id!(builder))
    return add_node!(builder, node)
end

rolling_window(builder::ReductionPlanBuilder, sizes::Int...; kwargs...) = 
    rolling_window(builder, sizes; kwargs...)

"""
    stats(builder::ReductionPlanBuilder, stat_type::Symbol)
    stats(builder::ReductionPlanBuilder, stat_types)

Add a statistics computation node using Val for type stability.
"""
function stats(builder::ReductionPlanBuilder, stat_type::Symbol)
    # Map stat symbol to kernel function
    kernel = _stat_to_kernel(stat_type)
    # Use a 1x1x... window (identity shape) since the windowing was
    # already handled by the preceding rolling_window node
    N = length(builder.input_shape)
    unit_sizes = ntuple(_ -> 1, N)
    config = WindowConfig(unit_sizes, unit_sizes, :valid)
    # Output shape is same as current shape (stat applied element-wise to windows)
    output_shape = builder.input_shape
    node = ReductionNode(kernel, config, output_shape, next_id!(builder))
    return add_node!(builder, node)
end

# Vector-of-symbols overload: use the first stat for the pipe node
function stats(builder::ReductionPlanBuilder, stat_types::AbstractVector{Symbol})
    return stats(builder, stat_types[1])
end

"""
    _stat_to_kernel(stat_type::Symbol) -> Function

Map a stat symbol to its canonical kernel function.
"""
function _stat_to_kernel(stat_type::Symbol)
    stat_type == :mean && return blockwise_mean!
    stat_type == :sum && return blockwise_sum!
    stat_type == :min && return blockwise_min!
    stat_type == :max && return blockwise_max!
    (stat_type == :var || stat_type == :variance) && return blockwise_variance!
    (stat_type == :std) && return blockwise_variance!  # caller wraps with sqrt
    error("Unknown statistic type: $stat_type. Use a kernel function directly.")
end

"""
    tree_reduce(builder::ReductionPlanBuilder, arity::Int=2)

Add a tree reduction node.
"""
function tree_reduce(builder::ReductionPlanBuilder, arity::Int=2)
    node = TreeNode(arity, next_id!(builder))
    return add_node!(builder, node)
end

"""
    fork(builder::ReductionPlanBuilder, num_branches::Int)

Create a fork point with multiple output branches from current node.
Returns vector of branch builder contexts for parallel construction.
"""
function fork(builder::ReductionPlanBuilder, num_branches::Int)
    # Mark current node as fork point
    fork_node_id = builder.current_node_id
    
    # Create branch contexts that all start from this fork node
    branches = Vector{ReductionPlanBuilder}(undef, num_branches)
    for i in 1:num_branches
        branch = ReductionPlanBuilder()
        branch.plan = builder.plan  # Share plan
        branch.counter = builder.counter  # Share counter
        branch.current_node_id = fork_node_id  # Start from fork point
        branch.input_shape = builder.input_shape
        branches[i] = branch
    end
    
    return branches
end

"""
    merge!(builder::ReductionPlanBuilder, branches)

Merge multiple branches back into main builder, finalizing their edges.
"""
function merge!(builder::ReductionPlanBuilder, branches)
    # Update counter from branches (they share the counter reference)
    builder.counter = maximum(b.counter for b in branches)
    
    # Update current node to be last node's id from any branch
    # (caller should continue from here)
    builder.current_node_id = nothing  # Force new input node on next add
    
    return builder
end

"""
    user_reduce(builder::ReductionPlanBuilder, f::Function, output_type::Type=Any)

Add a user-defined reduction function node.
"""
function user_reduce(builder::ReductionPlanBuilder, f::Function, output_type::Type=Any)
    node = UserNode{typeof(f)}(f, output_type, next_id!(builder))
    return add_node!(builder, node)
end

"""
    finalize_plan(builder::ReductionPlanBuilder)

Finalize the plan: compile the DAG into a flat execution sequence
that eliminates all runtime graph traversal overhead.
"""
function finalize_plan(builder::ReductionPlanBuilder)
    plan = builder.plan
    _compile_execution_sequence!(plan)
    return plan
end

"""
    _compile_execution_sequence!(plan::ReductionPlan)

Pre-compile the DAG into a flat Vector of ExecutionSteps.
After this, `execute` just iterates the vector — no Dict lookups,
no topological sort, no linear node scans at runtime.
"""
function _compile_execution_sequence!(plan::ReductionPlan)
    # Topological sort (done once at compile time, not at every execute call)
    visited = Set{UInt64}()
    order = UInt64[]
    queue = copy(plan.inputs)
    
    while !isempty(queue)
        node_id = popfirst!(queue)
        node_id in visited && continue
        push!(visited, node_id)
        push!(order, node_id)
        if haskey(plan.edges, node_id)
            for child in plan.edges[node_id]
                push!(queue, child)
            end
        end
    end
    
    # Build node_id → result_index mapping
    id_to_idx = Dict{UInt64, Int}()
    for (i, nid) in enumerate(order)
        id_to_idx[nid] = i
    end
    
    # Build reverse edge map: child → parents
    reverse_edges = Dict{UInt64, Vector{UInt64}}()
    for (src, dsts) in plan.edges
        for dst in dsts
            rev = get!(reverse_edges, dst, UInt64[])
            push!(rev, src)
        end
    end
    
    # Compile steps
    steps = ExecutionStep[]
    for (i, node_id) in enumerate(order)
        # Find the node
        node = nothing
        for n in plan.nodes
            if n.id == node_id
                node = n
                break
            end
        end
        node === nothing && error("Node $node_id not found during compilation")
        
        # Find input indices (parents in the DAG)
        parent_ids = get(reverse_edges, node_id, UInt64[])
        input_indices = Int[id_to_idx[pid] for pid in parent_ids]
        
        push!(steps, ExecutionStep(node, input_indices, i))
    end
    
    plan.execution_sequence = steps
    plan.output_indices = Int[id_to_idx[oid] for oid in plan.outputs]
    
    return nothing
end

Base.convert(::Type{ReductionPlan}, builder::ReductionPlanBuilder) = finalize_plan(builder)

"""
    merge_branches!(builder::ReductionPlanBuilder, branches::Vector{ReductionPlanBuilder}, merge_fn::Function)

Merge multiple branches back into a single path.

Creates a merge node that takes outputs from all branches and combines them.
The merge_fn should accept a vector of inputs and return a single combined result.
"""
function merge_branches!(builder::ReductionPlanBuilder, branches::Vector{ReductionPlanBuilder}, merge_fn::Function)
    # Collect all branch terminal nodes
    branch_outputs = UInt64[]
    for branch in branches
        if branch.current_node_id !== nothing
            push!(branch_outputs, branch.current_node_id)
        end
        # Sync counter to keep IDs unique
        builder.counter = max(builder.counter, branch.counter)
    end
    
    if isempty(branch_outputs)
        error("Cannot merge: no outputs from branches")
    end
    
    # Create merge node
    merge_node = MergeNode(branch_outputs, merge_fn, next_id!(builder))
    push!(builder.plan.nodes, merge_node)
    
    # Connect all branch outputs to merge node
    for output_id in branch_outputs
        edges = get!(builder.plan.edges, output_id, UInt64[])
        push!(edges, merge_node.id)
    end
    
    builder.current_node_id = merge_node.id
    return builder
end

"""
    parallel_reduce(builder::ReductionPlanBuilder, reduce_fn::Function, dims::Vector{Int})

Create parallel reduction branches for different dimensions.

Example: parallel_reduce(builder, mean, [1, 2]) creates horizontal and vertical
reduction branches that can be merged later.
"""
function parallel_reduce(builder::ReductionPlanBuilder, reduce_fn::Function, dims::Vector{Int})
    # Fork for each dimension
    branches = fork(builder, length(dims))
    
    # Add reduction to each branch
    for (i, branch) in enumerate(branches)
        # Create dimension-specific reduction node using mean kernel
        N = length(branch.input_shape)
        unit_sizes = ntuple(_ -> 1, N)
        config = WindowConfig(unit_sizes, unit_sizes, :valid)
        node = ReductionNode(blockwise_mean!, config, branch.input_shape, next_id!(branch))
        add_node!(branch, node)
    end
    
    # Merge branches back
    return merge_branches!(builder, branches, reduce_fn)
end

"""
    validate(plan::ReductionPlan)

Validate that the plan has valid connectivity and compatible shapes.
"""
function validate(plan::ReductionPlan)
    # Check that all nodes are connected
    all_node_ids = Set(node.id for node in plan.nodes)
    
    # Check edges point to valid nodes
    for (src, dsts) in plan.edges
        src in all_node_ids || error("Edge source $src not in nodes")
        for dst in dsts
            dst in all_node_ids || error("Edge destination $dst not in nodes")
        end
    end
    
    # Check inputs are valid
    for input in plan.inputs
        input in all_node_ids || error("Input $input not in nodes")
    end
    
    # Check outputs are valid
    for output in plan.outputs
        output in all_node_ids || error("Output $output not in nodes")
    end
    
    return true
end

"""
    plan_hash(plan::ReductionPlan)

Generate a hash for the plan structure (for caching).
"""
function plan_hash(plan::ReductionPlan)
    h = hash(length(plan.nodes))
    for node in plan.nodes
        h = hash(node.id, h)
        h = hash(typeof(node), h)
    end
    for (src, dsts) in plan.edges
        h = hash(src, h)
        for dst in dsts
            h = hash(dst, h)
        end
    end
    return h
end
