#
# ─── Multi-group dimension linking API ───────────────────────────────────────
#

"""
    build_multires_plan_groups(input_shape::NTuple{N,Int},
                               dim_groups::Vector{Pair{Tuple{Vararg{Int}}, Vector{Int}}};
                               stats_types=[:mean]) where N

Build a multi-resolution plan with arbitrary dimension linking.

Each dimension group is a set of dimensions that receive the same reduction factors.
This enables complex patterns like:
- Link dims 1,3 with factor 2, dims 2,6 with factor 3
- Fixed 100x100 horizontal with variable vertical (1,2,3)

# Arguments
- `input_shape`: Input array dimensions
- `dim_groups`: Vector of `dims => factors` pairs where `dims` is a tuple of dimension indices
  and `factors` is a vector of reduction factors for those dimensions
- `stats_types`: Statistics to compute (default: [:mean])

# Examples
```julia
# Link dims 1,3 with same factor, dims 2,6 with same factor
plan = build_multires_plan_groups(
    (nx, ny, nz, nw, nv, nu),
    [(1, 3) => [2, 4],      # dims 1,3 get factors 2,4
     (2, 6) => [3, 6]]      # dims 2,6 get factors 3,6
)

# 100x100 fixed horizontal with 1,2,3 vertical levels
plan = build_multires_plan_groups(
    (1000, 1000, 64),
    [(1, 2) => [10],        # horizontal: 1000/10 = 100
     (3,) => [1, 2, 4]]     # vertical: 64, 32, 16
)
```
"""
function build_multires_plan_groups(input_shape::NTuple{N,Int},
                                     dim_groups::Vector{Pair{TG, Vector{Int}}};
                                     stats_types=[:mean]) where {N, TG<:Tuple{Vararg{Int}}}
    isempty(dim_groups) && error("At least one dimension group required")
    
    # Validate dimension indices
    all_dims = Set{Int}()
    for (dims, _) in dim_groups
        for d in dims
            d < 1 || d > N && error("Invalid dimension index $d for input shape $input_shape")
            d in all_dims && error("Dimension $d appears in multiple groups")
            push!(all_dims, d)
        end
    end
    
    # Build all combinations of factor selections across groups
    target_shapes = Set{NTuple{N,Int}}()
    
    # Handle single group vs multiple groups
    group_factor_lists = [factors for (_, factors) in dim_groups]
    
    if length(group_factor_lists) == 1
        # Single group - just iterate over its factors
        (dims, factors) = dim_groups[1]
        for f in factors
            shape = ntuple(d -> d in dims ? div(input_shape[d], f) : input_shape[d], N)
            all(s -> s >= 1, shape) && push!(target_shapes, shape)
        end
    else
        # Multiple groups - use product of all factor lists
        for factor_selection in Iterators.product(group_factor_lists...)
            shape = input_shape
            for (i, (dims, _)) in enumerate(dim_groups)
                f = factor_selection[i]
                shape = ntuple(d -> d in dims ? div(shape[d], f) : shape[d], N)
            end
            all(s -> s >= 1, shape) && push!(target_shapes, shape)
        end
    end
    
    isempty(target_shapes) && error("No valid target shapes generated")
    
    # Build per-dimension factor lists from the targets
    # For each dimension, find all unique factors that were applied
    all_reduced_dims = Tuple(reduce(union, [Set(dims) for (dims, _) in dim_groups]))
    
    # Compute base block size from the largest target
    target_list = collect(target_shapes)
    sort!(target_list, by=prod, rev=true)  # Largest first
    base_target = first(target_list)
    base_block = ntuple(d -> div(input_shape[d], base_target[d]), N)
    
    # For tower_factors, we need per-dimension factors
    # But since we have arbitrary dimension linking, we need to be clever
    # We'll use the base_block to get to the first target, then use tower_factors
    # to reach the remaining targets
    
    # Actually simpler: just build the plan from targets directly using build_tower_plan
    # with a custom base_block derived from targets
    
    # Compute the min output size across all targets
    min_output = ntuple(d -> minimum(t[d] for t in target_list), N)
    
    # Use build_tower_plan with computed base_block
    # and tower_factors that will generate all other targets
    # For dims in the same group, we need factors that match the group pattern
    
    # Build per-dimension tower factors
    # For each dimension, compute what factors would reach all target sizes
    dim_factors = ntuple(d -> Int[], N)
    for d in all_reduced_dims
        target_sizes = unique!([t[d] for t in target_list])
        sort!(target_sizes, rev=true)
        factors_for_d = Int[]
        for i in 2:length(target_sizes)
            f = div(target_sizes[i-1], target_sizes[i])
            f > 1 && push!(factors_for_d, f)
        end
        isempty(factors_for_d) && push!(factors_for_d, 1)
        dim_factors = Base.setindex(dim_factors, factors_for_d, d)
    end
    
    return BlockwiseStatisticalReductions.build_tower_plan(
        input_shape;
        base_block=base_block,
        tower_factors=dim_factors,
        stats=stats_types,
        dims=all_reduced_dims,
        min_output_size=min_output,
        include_base=true,
        include_full=false
    )
end
