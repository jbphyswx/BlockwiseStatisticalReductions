"""
    Tower construction — the canonical multi-resolution DAG builder.

Provides:
- `seed_factor_ladder` / `build_factor_schedule`: generate factor towers from seeds
- `build_tower_plan`: BFS lattice + DAG construction (the core engine)
- `build_optimal_multires_plan`: factor-based solver (derives base_block/tower_factors,
  delegates to `build_tower_plan`)
- `build_tower_plan_from_outputs`: convenience wrapper from target output shapes
- `multiresolution_stats`: high-level execute-and-return convenience API
"""

#
# ─── Seed-based factor ladders ────────────────────────────────────────────────
#

"""
    seed_factor_ladder(n::Int, seed::Int; min_factor::Int=1) -> Tuple

Generate a tower of factors by repeatedly doubling the seed: `seed, seed*2, seed*4, ...`
up to the limit `n`.

Each factor in the ladder divides the next by 2, enabling optimal DAG reuse.

# Arguments
- `n`: Maximum factor (typically the dimension size)
- `seed`: Starting factor (e.g., 1, 3, 5, 7...)
- `min_factor`: Minimum factor to include

# Example
```julia
seed_factor_ladder(128, 1)           # (1, 2, 4, 8, 16, 32, 64, 128)
seed_factor_ladder(128, 3)           # (3, 6, 12, 24, 48, 96)
seed_factor_ladder(128, 1; min_factor=4)  # (4, 8, 16, 32, 64, 128)
```
"""
function seed_factor_ladder(n::Int, seed::Int; min_factor::Int=1)
    n >= 1 || throw(ArgumentError("n must be >= 1"))
    seed >= 1 || throw(ArgumentError("seed must be >= 1"))
    min_factor >= 1 || throw(ArgumentError("min_factor must be >= 1"))
    seed > n && return ()

    max_k = floor(Int, log2(div(n, seed)))
    factors = ntuple(k -> seed * (1 << (k - 1)), max_k + 1)
    return filter(f -> f >= min_factor && f <= n, factors)
end

"""
    build_factor_schedule(n::Int; seeds::NTuple{N,Int}=(1,), min_factor::Int=1, include_full::Bool=true) where N -> Vector{Int}

Build a sorted schedule of reduction factors from multiple seed ladders.

Each seed generates a tower `seed * 2^k` up to `n`. All ladders are merged,
deduplicated, and sorted. Optionally includes `n` itself (full reduction).

# Arguments
- `n`: Dimension size (maximum factor)
- `seeds`: Tuple of seed values (e.g., `(1,)`, `(1, 3)`, `(1, 3, 5)`)
- `min_factor`: Minimum factor to include
- `include_full`: Whether to include `n` as the final level

# Example
```julia
build_factor_schedule(128; seeds=(1,))       # [1,2,4,8,16,32,64,128]
build_factor_schedule(128; seeds=(1,3))      # [1,2,3,4,6,8,12,16,24,32,48,64,96,128]
build_factor_schedule(60; seeds=(1,), min_factor=2, include_full=true)  # [2,4,8,16,32,60]
```
"""
function build_factor_schedule(n::Int; seeds::NTuple{N,Int}=(1,),
                               min_factor::Int=1,
                               include_full::Bool=true) where N
    n >= 1 || throw(ArgumentError("n must be >= 1"))

    factors = Int[]
    for seed in seeds
        for f in seed_factor_ladder(n, seed; min_factor=min_factor)
            push!(factors, f)
        end
    end

    if include_full && !(n in factors)
        push!(factors, n)
    end

    sort!(unique!(factors))
    return factors
end

#
# ─── Tower plan from base block size ─────────────────────────────────────────
#

"""
    build_tower_plan(input_shape::NTuple{N,Int};
                     base_block::NTuple{N,Int},
                     tower_factors=[2],
                     stats=[:mean],
                     dims=ntuple(i->i, N-1),
                     min_output_size::NTuple{N,Int}=ntuple(_->1, N),
                     include_base::Bool=true,
                     include_full::Bool=true,
                     target_output_sizes=nothing) where N

Build a multi-resolution plan starting from a base block size and constructing
a tower of progressively coarser resolutions.

The BFS explores the **N-dimensional output shape lattice** independently per
dimension, applying per-dimension factor sets and respecting per-dimension
minimum output size constraints.

# How it works
1. Base reduction: input reduced by `base_block` → base output shape
2. Tower: BFS expands from the base output shape by dividing each participating
   dimension by its allowed factors, subject to `min_output_size` floors
3. DAG is built with maximum intermediate reuse (largest parent chosen)

# Arguments
- `input_shape`: Shape of input array
- `base_block`: Base block size per dim (finest coarsening). Can be a single
  `NTuple{N,Int}` or a `Vector{NTuple{N,Int}}` for multiple starting blocks.
  When multiple blocks are provided, BFS explores from all of them jointly
  in one DAG — shared intermediate shapes are computed only once.
- `tower_factors`: Factors for building coarser levels. Accepts:
  - `Vector{Int}` — same factors applied to all `dims` (e.g., `[2, 3]`)
  - `NTuple{N, Vector{Int}}` — per-dimension factor sets (non-participating
    dims should use `Int[]` or `[1]`)
- `stats`: Statistics to compute
- `dims`: Which dimensions to reduce (default: all except last)
- `min_output_size`: Per-dimension minimum output size (floor constraint).
  BFS will not produce levels with any dim smaller than this. Default: `(1,1,...)`
- `include_base`: Include the base level in outputs
- `include_full`: Include full-domain reduction if reachable

# Examples
```julia
# Uniform factors across x,y; z excluded via dims
plan = build_tower_plan((6000, 6000, 64);
    base_block=(100, 100, 4),
    tower_factors=[2, 3],
    dims=(1, 2))
# base output: 60×60×16, tower: 30×30×16, 20×20×16, 10×10×16, ...

# Per-dimension factors with z capped
plan = build_tower_plan((6000, 6000, 64);
    base_block=(100, 100, 4),
    tower_factors=([2, 3], [2, 3], [2]),  # z only halves
    min_output_size=(1, 1, 4),            # z never goes below 4
    dims=(1, 2, 3))
# Produces shapes like 60×60×16, 30×30×8, 20×20×8, 10×10×4, ...
# z stops reducing once it hits 4
```
"""
function build_tower_plan(input_shape::NTuple{N,Int};
                          base_block::Union{NTuple{N,Int}, Vector{NTuple{N,Int}}},
                          tower_factors=[2],
                          stats=[:mean],
                          dims=ntuple(i->i, N-1),
                          min_output_size::NTuple{N,Int}=ntuple(_->1, N),
                          include_base::Bool=true,
                          include_full::Bool=true) where N
    # Normalize base_block to a vector of blocks
    base_blocks = base_block isa Vector ? base_block : NTuple{N,Int}[base_block]

    # Validate all base blocks
    for bb in base_blocks
        for i in 1:N
            if i in dims && bb[i] > input_shape[i]
                throw(ArgumentError("base_block[$i]=$(bb[i]) > input_shape[$i]=$(input_shape[i])"))
            end
        end
    end

    # Normalize tower_factors to per-dimension Vector{Int}
    dim_factors = _normalize_tower_factors(Val(N), tower_factors, dims)

    # Compute base output shapes for all base blocks
    base_outputs = NTuple{N,Int}[
        ntuple(i -> i in dims ? div(input_shape[i], bb[i]) : input_shape[i], N)
        for bb in base_blocks
    ]

    # BFS over N-dimensional output shapes — seeded from ALL base outputs jointly
    reachable = Set{NTuple{N,Int}}(base_outputs)
    queue = NTuple{N,Int}[bo for bo in base_outputs]
    # For each reachable shape, store (parent_shape, window_sizes) for DAG construction
    parent_map = Dict{NTuple{N,Int}, Tuple{NTuple{N,Int}, NTuple{N,Int}}}()

    while !isempty(queue)
        current = popfirst!(queue)
        # Try all combinations: for each dim independently, try each factor
        # Generate "single-step" children by applying one factor to one dim at a time
        for d in 1:N
            d in dims || continue
            for tf in dim_factors[d]
                tf <= 1 && continue
                current[d] % tf == 0 || continue
                next_d = div(current[d], tf)
                next_d >= min_output_size[d] || continue
                child = ntuple(i -> i == d ? next_d : current[i], N)
                if !(child in reachable)
                    push!(reachable, child)
                    push!(queue, child)
                    # Record parent: prefer largest parent (set on first discovery via BFS)
                    window = ntuple(i -> i == d ? tf : 1, N)
                    parent_map[child] = (current, window)
                end
            end
        end
        # Also try reducing multiple dims simultaneously (common: same factor in x and y)
        # Generate combined steps for dims that share factors
        _bfs_combined_steps!(queue, reachable, parent_map, current, dims, dim_factors, min_output_size, N)
    end

    # Sort reachable shapes: finest first (largest total cells), coarsest last
    sorted_shapes = sort(collect(reachable), by=s -> -prod(s))

    # Build DAG
    builder = build_plan(input_shape)
    output_node_ids = UInt64[]
    shape_to_node = Dict{NTuple{N,Int}, UInt64}()


    # Determine whether stats require sufficient-statistics merging
    needs_sufstats = _needs_sufficient_stats(stats)

    if needs_sufstats
        # ─── Sufficient-statistics DAG (variance, std, etc.) ─────────────
        # Track cumulative count for each shape (needed by merge kernel)
        shape_count = Dict{NTuple{N,Int}, Int}()

        # Create one root node per base block — each is an independent input
        for (bb, base_output) in zip(base_blocks, base_outputs)
            base_window_sizes = ntuple(i -> i in dims ? bb[i] : 1, N)
            base_count = prod(base_window_sizes)
            base_window = WindowConfig(base_window_sizes, base_window_sizes, :valid)
            base_node = SufficientStatsNode(
                blockwise_mean_M2!, blockwise_merge_mean_M2!,
                base_window, base_output, base_count, 2, true, next_id!(builder))
            push!(builder.plan.nodes, base_node)
            push!(builder.plan.inputs, base_node.id)
            shape_to_node[base_output] = base_node.id
            shape_count[base_output] = base_count
            if include_base
                push!(output_node_ids, base_node.id)
            end
        end

        for shape in sorted_shapes
            shape in base_outputs && continue
            haskey(parent_map, shape) || continue
            parent_shape, window_sizes = parent_map[shape]
            haskey(shape_to_node, parent_shape) || continue

            parent_node_id = shape_to_node[parent_shape]
            parent_count = shape_count[parent_shape]

            level_window = WindowConfig(window_sizes, window_sizes, :valid)
            level_node = SufficientStatsNode(
                blockwise_mean_M2!, blockwise_merge_mean_M2!,
                level_window, shape, parent_count, 2, false, next_id!(builder))
            push!(builder.plan.nodes, level_node)

            edges = get!(builder.plan.edges, parent_node_id, UInt64[])
            push!(edges, level_node.id)

            shape_to_node[shape] = level_node.id
            shape_count[shape] = parent_count * prod(window_sizes)
            push!(output_node_ids, level_node.id)
        end

        # Full-domain reduction
        if include_full
            full_shape = ntuple(i -> i in dims ? 1 : input_shape[i], N)
            if !(full_shape in reachable) && !haskey(shape_to_node, full_shape)
                smallest_shape = sorted_shapes[end]
                if haskey(shape_to_node, smallest_shape) && haskey(shape_count, smallest_shape) && any(i -> smallest_shape[i] > min_output_size[i], 1:N)
                    parent_count = shape_count[smallest_shape]
                    full_window_sizes = ntuple(i -> i in dims ? smallest_shape[i] : 1, N)
                    full_window = WindowConfig(full_window_sizes, full_window_sizes, :valid)
                    full_node = SufficientStatsNode(
                        blockwise_mean_M2!, blockwise_merge_mean_M2!,
                        full_window, full_shape, parent_count, 2, false, next_id!(builder))
                    push!(builder.plan.nodes, full_node)
                    edges = get!(builder.plan.edges, shape_to_node[smallest_shape], UInt64[])
                    push!(edges, full_node.id)
                    shape_to_node[full_shape] = full_node.id
                    push!(output_node_ids, full_node.id)
                end
            elseif haskey(shape_to_node, full_shape)
                push!(output_node_ids, shape_to_node[full_shape])
            end
        end
    else
        # ─── Simple composable DAG (mean, sum, min, max) ────────────────
        kernel = _select_kernel(stats)

        # Create one root node per base block — each is an independent input
        for (bb, base_output) in zip(base_blocks, base_outputs)
            base_window_sizes = ntuple(i -> i in dims ? bb[i] : 1, N)
            base_window = WindowConfig(base_window_sizes, base_window_sizes, :valid)
            base_node = ReductionNode(kernel, base_window, base_output, next_id!(builder))
            push!(builder.plan.nodes, base_node)
            push!(builder.plan.inputs, base_node.id)
            shape_to_node[base_output] = base_node.id
            if include_base
                push!(output_node_ids, base_node.id)
            end
        end

        for shape in sorted_shapes
            shape in base_outputs && continue
            haskey(parent_map, shape) || continue

            parent_shape, window_sizes = parent_map[shape]
            haskey(shape_to_node, parent_shape) || continue

            parent_node_id = shape_to_node[parent_shape]

            level_window = WindowConfig(window_sizes, window_sizes, :valid)
            level_node = ReductionNode(kernel, level_window, shape, next_id!(builder))
            push!(builder.plan.nodes, level_node)

            edges = get!(builder.plan.edges, parent_node_id, UInt64[])
            push!(edges, level_node.id)

            shape_to_node[shape] = level_node.id
            push!(output_node_ids, level_node.id)
        end

        # Full-domain reduction
        if include_full
            full_shape = ntuple(i -> i in dims ? 1 : input_shape[i], N)
            if !(full_shape in reachable) && !haskey(shape_to_node, full_shape)
                smallest_shape = sorted_shapes[end]
                if haskey(shape_to_node, smallest_shape) && any(i -> smallest_shape[i] > min_output_size[i], 1:N)
                    full_window_sizes = ntuple(i -> i in dims ? smallest_shape[i] : 1, N)
                    full_window = WindowConfig(full_window_sizes, full_window_sizes, :valid)
                    full_node = ReductionNode(kernel, full_window, full_shape, next_id!(builder))
                    push!(builder.plan.nodes, full_node)
                    edges = get!(builder.plan.edges, shape_to_node[smallest_shape], UInt64[])
                    push!(edges, full_node.id)
                    shape_to_node[full_shape] = full_node.id
                    push!(output_node_ids, full_node.id)
                end
            elseif haskey(shape_to_node, full_shape)
                push!(output_node_ids, shape_to_node[full_shape])
            end
        end
    end

    builder.plan.outputs = output_node_ids
    return finalize_plan(builder)
end

"""
    _normalize_tower_factors(::Val{N}, tf, dims) -> NTuple{N, Vector{Int}}

Normalize tower_factors to per-dimension Vector{Int}.
"""
function _normalize_tower_factors(::Val{N}, tf::NTuple{N}, dims) where N
    # Per-dimension factor collections (tuple of iterables)
    return ntuple(i -> collect(Int, tf[i]), N)
end

function _normalize_tower_factors(::Val{N}, tf, dims) where N
    # Single collection of factors applied to all active dims
    factors = collect(Int, tf)
    return ntuple(i -> i in dims ? factors : Int[], N)
end

"""
    _bfs_combined_steps!(queue, reachable, parent_map, current, dims, dim_factors, min_output_size, N)

Expand BFS by applying factors to multiple dims simultaneously.
This covers the common case of square reductions (same factor in x and y at once).
"""
function _bfs_combined_steps!(queue, reachable, parent_map, current::NTuple{N,Int},
                              dims, dim_factors, min_output_size::NTuple{N,Int}, ::Int) where N
    # Find factors shared across multiple active dims
    active_dims = [d for d in 1:N if d in dims]
    length(active_dims) < 2 && return

    # Collect factors common to at least 2 active dims
    all_factors = Set{Int}()
    for d in active_dims
        for f in dim_factors[d]
            f > 1 && push!(all_factors, f)
        end
    end

    for tf in all_factors
        # Which active dims can be divided by this factor?
        applicable_dims = [d for d in active_dims
                           if tf in dim_factors[d] &&
                              current[d] % tf == 0 &&
                              div(current[d], tf) >= min_output_size[d]]
        length(applicable_dims) < 2 && continue

        # Apply factor to all applicable dims simultaneously
        child = ntuple(i -> i in applicable_dims ? div(current[i], tf) : current[i], N)
        if !(child in reachable)
            push!(reachable, child)
            push!(queue, child)
            window = ntuple(i -> i in applicable_dims ? tf : 1, N)
            parent_map[child] = (current, window)
        end
    end
end

"""
    _needs_sufficient_stats(stats) -> Bool

Check whether any requested statistic requires sufficient-statistics merging
(i.e., cannot be computed by simple chaining of the stat through the DAG).

Variance and std require tracking mean + M2 (sum of squared deviations) and
using Chan's parallel merge formula at each coarser level.
"""
function _needs_sufficient_stats(stats)
    svec = stats isa Symbol ? [stats] : collect(Symbol, stats)
    return any(s -> s in (:variance, :var, :std), svec)
end

"""
    _select_kernel(stats) -> Function

Map requested statistics to the appropriate kernel function for DAG nodes.

Returns the kernel that each `ReductionNode` in the tower should call.
For composable stats (mean, sum, min, max) the kernel chains correctly
through the DAG. For stats requiring sufficient-statistics merging (variance,
std), the merge kernels are used at coarser levels.
"""
function _select_kernel(stats)
    # Normalize to vector
    svec = stats isa Symbol ? [stats] : collect(Symbol, stats)

    if length(svec) == 1
        s = svec[1]
        s == :mean && return blockwise_mean!
        s == :sum && return blockwise_sum!
        s == :min && return blockwise_min!
        s == :max && return blockwise_max!
    end

    # Default: mean kernel (safe for DAG chaining; multi-stat handled at
    # execution time by multiresolution_stats)
    return blockwise_mean!
end

#
# ─── Factor-based convenience: build_optimal_multires_plan ───────────────────
#

"""
    build_optimal_multires_plan(input_shape::NTuple{N,Int}, target_factors,
                                stats_types=[:mean];
                                dims=ntuple(i->i, N-1)) where N

Build a minimal DAG that produces exactly the requested output sizes.

This is a **solver** — distinct from `build_tower_plan`.  A tower explores all
reachable shapes and outputs everything.  This function works backward from the
desired targets to find the shortest reduction paths, sharing intermediates
where multiple targets have a common ancestor.

# Algorithm
1. Convert each target factor to a target output shape.
2. For each target, find the shortest factorization chain from raw data →
   target shape (greedy: always take the largest valid divisor first).
3. Build one DAG containing only the nodes on those chains, merging where
   paths share intermediate shapes.

# Arguments
- `input_shape`: Shape of input array
- `target_factors`: Desired reduction factors (e.g., `[2, 4, 8, 12]`)
- `stats_types`: Statistics to compute (default: `[:mean]`)
- `dims`: Which dimensions to reduce (default: all except last)

# Example
```julia
plan = build_optimal_multires_plan((120, 120, 8), [2, 4, 6, 12], [:mean]; dims=(1, 2))
```
"""
function build_optimal_multires_plan(input_shape::NTuple{N, Int}, target_factors,
                                      stats_types=[:mean];
                                      dims=ntuple(i->i, N-1)) where N
    unique_factors = sort(unique(filter(f -> f > 0, target_factors)))
    isempty(unique_factors) && error("No valid target factors")

    # Target output shapes
    target_shapes = Set{NTuple{N,Int}}()
    for f in unique_factors
        shape = ntuple(i -> i in dims ? div(input_shape[i], f) : input_shape[i], N)
        push!(target_shapes, shape)
    end

    # Build a minimal DAG by finding parent-child relationships between shapes.
    # Strategy: process targets from finest (largest shape) to coarsest (smallest).
    # For each target, check if any already-registered shape can serve as its parent
    # (i.e., the parent can be reduced to the target by dividing each dim by an integer).
    # If so, prefer the closest (smallest) such parent. Otherwise, reduce directly
    # from input_shape.
    #
    # parent_of[child] = (parent_shape, window_sizes_from_parent_to_child)
    needed_shapes = Set{NTuple{N,Int}}()
    parent_of = Dict{NTuple{N,Int}, Tuple{NTuple{N,Int}, NTuple{N,Int}}}()

    # Sort targets: finest first (largest prod = closest to input_shape)
    sorted_targets = sort(collect(target_shapes), by=s -> -prod(s))

    for target in sorted_targets
        target == input_shape && continue
        haskey(parent_of, target) && continue

        # Can any already-registered needed_shape serve as parent?
        # A shape P is a valid parent of target if for each dim d in dims:
        #   P[d] % target[d] == 0 and P[d] > target[d]  (or P[d] == target[d] for non-reduced dims)
        # and for non-dims: P[d] == target[d]
        best_parent = nothing
        best_window = nothing

        for candidate in needed_shapes
            candidate == target && continue
            valid = true
            for i in 1:N
                if i in dims
                    # Compute window and verify it actually produces the target
                    w = div(candidate[i], target[i])
                    if w < 2 || div(candidate[i], w) != target[i]
                        valid = false
                        break
                    end
                else
                    if candidate[i] != target[i]
                        valid = false
                        break
                    end
                end
            end
            valid || continue
            # Prefer smallest valid parent (closest ancestor → smallest window)
            if best_parent === nothing || prod(candidate) < prod(best_parent)
                best_parent = candidate
                best_window = ntuple(i -> i in dims ? div(candidate[i], target[i]) : 1, N)
            end
        end

        # Also consider input_shape as parent (direct one-step reduction)
        input_valid = true
        for i in 1:N
            if i in dims
                w = div(input_shape[i], target[i])
                if w < 1 || div(input_shape[i], w) != target[i]
                    input_valid = false
                    break
                end
            end
        end
        if input_valid
            input_window = ntuple(i -> i in dims ? div(input_shape[i], target[i]) : 1, N)
            # Prefer an existing intermediate over input_shape (to share computation)
            if best_parent === nothing
                best_parent = input_shape
                best_window = input_window
            end
        end

        if best_parent === nothing
            error("Cannot reach target shape $target from input $input_shape")
        end

        parent_of[target] = (best_parent, best_window)
        push!(needed_shapes, target)
    end

    # Now build the DAG — shapes whose parent is input_shape are root nodes,
    # others are interior nodes.
    builder = build_plan(input_shape)
    output_node_ids = UInt64[]
    shape_to_node = Dict{NTuple{N,Int}, UInt64}()

    needs_sufstats = _needs_sufficient_stats(stats_types)

    # Process shapes from largest to smallest (parents before children)
    sorted_needed = sort(collect(needed_shapes), by=s -> -prod(s))

    if needs_sufstats
        shape_count = Dict{NTuple{N,Int}, Int}()

        for shape in sorted_needed
            haskey(shape_to_node, shape) && continue
            haskey(parent_of, shape) || continue
            parent_shape, window_sizes = parent_of[shape]

            if parent_shape == input_shape
                # Root node: reduces from raw data
                base_count = prod(window_sizes)
                window = WindowConfig(window_sizes, window_sizes, :valid)
                node = SufficientStatsNode(
                    blockwise_mean_M2!, blockwise_merge_mean_M2!,
                    window, shape, base_count, 2, true, next_id!(builder))
                push!(builder.plan.nodes, node)
                push!(builder.plan.inputs, node.id)
                shape_to_node[shape] = node.id
                shape_count[shape] = base_count
            else
                # Interior node: reduces from a parent shape
                haskey(shape_to_node, parent_shape) || continue
                parent_node_id = shape_to_node[parent_shape]
                parent_count = shape_count[parent_shape]
                window = WindowConfig(window_sizes, window_sizes, :valid)
                node = SufficientStatsNode(
                    blockwise_mean_M2!, blockwise_merge_mean_M2!,
                    window, shape, parent_count, 2, false, next_id!(builder))
                push!(builder.plan.nodes, node)
                edges = get!(builder.plan.edges, parent_node_id, UInt64[])
                push!(edges, node.id)
                shape_to_node[shape] = node.id
                shape_count[shape] = parent_count * prod(window_sizes)
            end

            if shape in target_shapes
                push!(output_node_ids, shape_to_node[shape])
            end
        end
    else
        kernel = _select_kernel(stats_types)

        for shape in sorted_needed
            haskey(shape_to_node, shape) && continue
            haskey(parent_of, shape) || continue
            parent_shape, window_sizes = parent_of[shape]

            if parent_shape == input_shape
                # Root node
                window = WindowConfig(window_sizes, window_sizes, :valid)
                node = ReductionNode(kernel, window, shape, next_id!(builder))
                push!(builder.plan.nodes, node)
                push!(builder.plan.inputs, node.id)
                shape_to_node[shape] = node.id
            else
                # Interior node
                haskey(shape_to_node, parent_shape) || continue
                parent_node_id = shape_to_node[parent_shape]
                window = WindowConfig(window_sizes, window_sizes, :valid)
                node = ReductionNode(kernel, window, shape, next_id!(builder))
                push!(builder.plan.nodes, node)
                edges = get!(builder.plan.edges, parent_node_id, UInt64[])
                push!(edges, node.id)
                shape_to_node[shape] = node.id
            end

            if shape in target_shapes
                push!(output_node_ids, shape_to_node[shape])
            end
        end
    end

    builder.plan.outputs = output_node_ids
    return finalize_plan(builder)
end

#
# ─── Tower plan from target output sizes ─────────────────────────────────────
#

"""
    build_tower_plan_from_outputs(input_shape::NTuple{N,Int},
                                  target_outputs;
                                  stats=[:mean],
                                  dims=ntuple(i->i, N-1)) where N

Build a multi-resolution plan from desired output shapes.

Converts target output shapes to factors and builds a minimal DAG via
`build_optimal_multires_plan`.

# Example
```julia
plan = build_tower_plan_from_outputs((6000, 6000, 8),
    [(60, 60, 8), (30, 30, 8), (20, 20, 8), (10, 10, 8)];
    dims=(1, 2))
# Equivalent to factors [100, 200, 300, 600]
```
"""
function build_tower_plan_from_outputs(input_shape::NTuple{N,Int},
                                       target_outputs;
                                       stats=[:mean],
                                       dims=ntuple(i->i, N-1)) where N
    ref_dim = first(dims)
    factors = Int[]

    for target in target_outputs
        push!(factors, div(input_shape[ref_dim], target[ref_dim]))
    end

    unique!(sort!(factors))
    return build_optimal_multires_plan(input_shape, factors, stats; dims=dims)
end

#
# ─── Convenience high-level API ──────────────────────────────────────────────
#

"""
    multiresolution_stats(data::AbstractArray{T,N}, target_factors;
                          stats=[:mean],
                          dims=ntuple(i->i, N-1),
                          corrected::Bool=true) where {T,N}

High-level API for multi-resolution statistics.

Builds and executes a tower plan DAG, then extracts the requested statistics
from the results. For composable stats (mean, sum, min, max), the DAG chains
the kernel directly. For stats requiring sufficient-statistics merging
(variance, std), the DAG uses `blockwise_mean_M2!` at the base level and
`blockwise_merge_mean_M2!` at coarser levels, then extracts the final
statistics from the sufficient statistics.

Returns `Dict{Int, Dict{Symbol, AbstractArray}}` — factor → stat → array.

# Example
```julia
results = multiresolution_stats(data, [2, 4, 8]; stats=[:mean, :variance])
results[4][:mean]       # mean at 4× coarsening
results[4][:variance]   # variance at 4× coarsening
```
"""
function multiresolution_stats(data::AbstractArray{T,N}, target_factors;
                               stats=[:mean],
                               dims=ntuple(i->i, N-1),
                               corrected::Bool=true) where {T,N}
    input_shape = size(data)
    stats_vec = collect(Symbol, stats)
    sorted_factors = sort(unique(filter(f -> f > 0, collect(Int, target_factors))))
    isempty(sorted_factors) && return Dict{Int, Dict{Symbol, AbstractArray}}()

    # Build a single plan (may contain multiple independent tower subgraphs
    # if factors are incompatible, e.g. [2, 5]).
    plan = build_optimal_multires_plan(input_shape, sorted_factors, stats_vec; dims=dims)
    needs_sufstats = _needs_sufficient_stats(stats_vec)

    results = execute(plan, data)
    out = Dict{Int, Dict{Symbol, AbstractArray}}()
    for r in results
        if needs_sufstats
            ss = r.data
            out_shape = size(ss.mean)
            factor = _shape_to_factor(input_shape, out_shape, dims)
            factor === nothing && continue
            haskey(out, factor) && continue
            stats_dict = Dict{Symbol, AbstractArray}()
            total_count = _total_count_for_shape(input_shape, out_shape, dims)
            :mean in stats_vec && (stats_dict[:mean] = ss.mean)
            (:variance in stats_vec || :var in stats_vec) &&
                (stats_dict[:variance] = _m2_to_variance(ss.M2, total_count; corrected=corrected))
            :std in stats_vec &&
                (stats_dict[:std] = sqrt.(_m2_to_variance(ss.M2, total_count; corrected=corrected)))
            out[factor] = stats_dict
        else
            arr = r.data
            out_shape = size(arr)
            factor = _shape_to_factor(input_shape, out_shape, dims)
            factor === nothing && continue
            haskey(out, factor) && continue
            stats_dict = Dict{Symbol, AbstractArray}()
            stat_name = length(stats_vec) == 1 ? stats_vec[1] : :mean
            stats_dict[stat_name] = arr
            out[factor] = stats_dict
        end
    end

    return out
end

"""
    _shape_to_factor(input_shape, output_shape, dims) -> Union{Int, Nothing}

Infer the coarsening factor from input and output shapes.
Uses the first active dimension as reference.
"""
function _shape_to_factor(input_shape::NTuple{N,Int}, output_shape::NTuple{N,Int}, dims) where N
    for d in 1:N
        d in dims || continue
        output_shape[d] > 0 || continue
        return div(input_shape[d], output_shape[d])
    end
    return nothing
end

"""
    _total_count_for_shape(input_shape, output_shape, dims) -> Int

Compute the total number of raw samples per output element.
"""
function _total_count_for_shape(input_shape::NTuple{N,Int}, output_shape::NTuple{N,Int}, dims) where N
    count = 1
    for d in 1:N
        d in dims || continue
        count *= div(input_shape[d], output_shape[d])
    end
    return count
end

"""
    _m2_to_variance(M2, count; corrected=true) -> Array

Convert M2 (sum of squared deviations) array to variance array.
"""
function _m2_to_variance(M2::AbstractArray{T}, count::Int; corrected::Bool=true) where T
    denom = corrected ? T(count - 1) : T(count)
    return M2 ./ denom
end

