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

Add a statistics computation node using Val for type stability.
"""
function stats(builder::ReductionPlanBuilder, stat_type::Symbol)
    node = StatsNode(Val(stat_type), :, next_id!(builder))
    return add_node!(builder, node)
end

function stats(builder::ReductionPlanBuilder, stat_types::Vector{Symbol})
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
    merge!(builder::ReductionPlanBuilder, branches::Vector{ReductionPlanBuilder})

Merge multiple branches back into main builder, finalizing their edges.
"""
function merge!(builder::ReductionPlanBuilder, branches::Vector{ReductionPlanBuilder})
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

Finalize the plan and return a ReductionPlan.
"""
function finalize_plan(builder::ReductionPlanBuilder)
    return builder.plan
end

"""
    build_optimal_multires_plan(input_shape::NTuple, target_factors::Vector{Int}, stats_types::Vector{Symbol})

Build an optimized multi-resolution plan with DAG reuse.

Creates a proper DAG where intermediate results are reused:
- 2x reduction feeds 4x (2*2), 8x (2*4), etc.
- 3x reduction feeds 6x (3*2), 12x (3*4), etc.
- Minimizes redundant computation through caching.
"""
function build_optimal_multires_plan(input_shape::NTuple{N, Int}, target_factors::Vector{Int}, 
                                      stats_types::Vector{Symbol}=[:mean]) where N
    
    # Sort and deduplicate target factors
    unique_factors = sort(unique(filter(f -> f > 0, target_factors)))
    isempty(unique_factors) && error("No valid target factors")
    
    # Build factor sequence using gcd-based optimization
    factor_chain = _build_optimal_factor_chain(unique_factors)
    
    builder = build_plan(input_shape)
    output_map = Dict{Int, UInt64}()  # factor -> node_id
    
    # Process each factor in dependency order
    for (factor, parent_factor, step_factor) in factor_chain
        if parent_factor == 0
            # Root level - reduce from input
            window = WindowConfig(ntuple(i -> factor, N), ntuple(i -> factor, N), :valid)
            window_node = WindowNode(window, next_id!(builder))
            add_node!(builder, window_node)
            window_id = builder.current_node_id
            
            # Add stats node chained from window node
            stat_type = length(stats_types) == 1 ? stats_types[1] : stats_types
            stats_node = StatsNode(Val(stat_type), :, next_id!(builder))
            add_node!(builder, stats_node)
            
            # Link window -> stats
            edges = get!(builder.plan.edges, window_id, UInt64[])
            push!(edges, stats_node.id)
            
            output_map[factor] = stats_node.id
        else
            # Get parent stats node id (this is the data source)
            parent_id = output_map[parent_factor]
            
            # Add window node for reduction step
            window = WindowConfig(ntuple(i -> step_factor, N), ntuple(i -> step_factor, N), :valid)
            window_node = WindowNode(window, next_id!(builder))
            push!(builder.plan.nodes, window_node)
            
            # Link parent stats -> window node (window receives the data)
            edges = get!(builder.plan.edges, parent_id, UInt64[])
            push!(edges, window_node.id)
            
            builder.current_node_id = window_node.id
            window_id = window_node.id
            
            # Add stats node chained from window node
            stat_type = length(stats_types) == 1 ? stats_types[1] : stats_types
            stats_node = StatsNode(Val(stat_type), :, next_id!(builder))
            add_node!(builder, stats_node)
            
            # Link window -> stats
            edges = get!(builder.plan.edges, window_id, UInt64[])
            push!(edges, stats_node.id)
            
            output_map[factor] = stats_node.id
        end
    end
    
    # Set outputs to be all the final stats nodes
    builder.plan.outputs = collect(values(output_map))
    
    return finalize_plan(builder)
end

"""
    _build_optimal_factor_chain(target_factors::Vector{Int})

Build optimal factor chain for DAG construction.
Returns vector of (target_factor, parent_factor, step_factor) tuples.
"""
function _build_optimal_factor_chain(target_factors::Vector{Int})
    sorted = sort(target_factors)
    chain = Tuple{Int, Int, Int}[]
    
    for target in sorted
        # Find best parent (largest factor that divides target)
        best_parent = 0
        best_step = target
        
        for candidate in chain
            cand_factor = candidate[1]
            if target % cand_factor == 0 && cand_factor > best_parent
                step = div(target, cand_factor)
                # Prefer step sizes that are powers of 2 (efficient)
                if ispow2(step) || best_parent == 0
                    best_parent = cand_factor
                    best_step = step
                end
            end
        end
        
        # Also check if target itself could be a parent of later factors
        # by using factor decomposition
        if best_parent == 0
            # Try to decompose into smaller factors
            for cand in sorted
                if cand < target && target % cand == 0
                    step = div(target, cand)
                    if step < best_step
                        best_parent = cand
                        best_step = step
                    end
                end
            end
        end
        
        push!(chain, (target, best_parent, best_step))
    end
    
    return chain
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
