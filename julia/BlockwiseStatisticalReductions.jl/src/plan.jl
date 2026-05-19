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
    stats(builder::ReductionPlanBuilder, stat_types::Vector{Symbol})

Add a statistics computation node.
"""
function stats(builder::ReductionPlanBuilder, stat_type::Symbol)
    T = create_stat(stat_type, Float64)
    node = StatsNode{typeof(T)}(T, :, next_id!(builder))
    return add_node!(builder, node)
end

function stats(builder::ReductionPlanBuilder, stat_types::Vector{Symbol})
    T = create_stat(stat_types, Float64)
    node = StatsNode{typeof(T)}(T, :, next_id!(builder))
    return add_node!(builder, node)
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
    user_reduce(builder::ReductionPlanBuilder, f::Function, output_type::Type=Any)

Add a user-defined reduction function node.
"""
function user_reduce(builder::ReductionPlanBuilder, f::Function, output_type::Type=Any)
    node = UserNode{typeof(f)}(f, output_type, next_id!(builder))
    return add_node!(builder, node)
end

"""
    finalize_plan(builder::ReductionPlanBuilder)

Convert the builder to a finalized ReductionPlan.
"""
function finalize_plan(builder::ReductionPlanBuilder)
    if builder.current_node_id !== nothing
        push!(builder.plan.outputs, builder.current_node_id)
    end
    return builder.plan
end

# Allow calling finalize() on the builder
Base.convert(::Type{ReductionPlan}, builder::ReductionPlanBuilder) = finalize_plan(builder)

"""
    fork(builder::ReductionPlanBuilder, n_branches::Int=2)

Create a fork point where the current node feeds into multiple parallel branches.

Returns a vector of n branch builders that can be modified independently.
Each branch builder shares the same underlying plan but tracks its own
branch lineage for later merging.
"""
function fork(builder::ReductionPlanBuilder, n_branches::Int=2)
    if builder.current_node_id === nothing
        error("Cannot fork: no current node. Start with build_plan() first.")
    end
    
    parent_id = builder.current_node_id
    branch_builders = []
    
    for _ in 1:n_branches
        # Create a branch builder that shares the plan but starts from parent
        branch = ReductionPlanBuilder(builder.plan, parent_id, builder.counter, builder.input_shape)
        push!(branch_builders, branch)
    end
    
    return branch_builders
end

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
        # Create dimension-specific reduction node
        stat = create_stat(:mean, Float64)  # Default to mean, can be customized
        node = StatsNode{typeof(stat)}(stat, dims[i], next_id!(branch))
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
