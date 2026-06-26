# ─────────────────────────────────────────────────────────────────────────────
# The planner: build a minimum-work DAG over the divisor lattice
# ─────────────────────────────────────────────────────────────────────────────
#
# Cost model. Producing a node at factor `f` from a source (the input, or a finer materialized
# node at factor `p` with `p | f`) touches every element of the *source* once:
#
#     cost(from input)      = prod(X)                       # one full pass over the data
#     cost(from parent p)   = prod(shape(p)) = prod(X .÷ p) # one pass over the (smaller) parent
#
# so the cheapest parent of `f` is the materialized proper divisor with the LARGEST factor (=
# smallest array). Given the materialized set, picking that parent for every node is exactly
# optimal (edge cost depends only on the parent, monotone in the lattice). Along a single 1-D
# chain with per-level shrink ρ≥2 this telescopes to total work < ρ/(ρ-1)·prod(X) ≤ 2·prod(X) —
# the whole stack costs less than two passes. A full N-D tower also materializes every anisotropic
# intermediate, so its total is ≈ (ρ/(ρ-1))^N·prod(X) = O(2^N·prod(X)) — independent of the number
# of levels, and still far below the naive (#outputs)·prod(X) of independent reductions.
#
# For an arbitrary requested set (solver mode), materializing a shared finer intermediate can let
# several targets reuse one pass; the beneficial candidates are exactly the gcd-closure of the
# targets (see [`gcd_closure`]). Steiner augmentation greedily adds the best-improving candidate.

"""
    ReductionStep{N}

One node of a [`ReductionPlan`]: reduce to `shape` (= input `.÷ factor`). `source == 0` means a base
pass from the input array with block `window` (`== factor`); otherwise `source` indexes the parent
step and `window` (`= factor ./ parent.factor`) is the merge block applied to it. `is_output` marks
nodes the user requested (vs shared intermediates introduced by the planner).
"""
struct ReductionStep{N}
    factor::NTuple{N,Int}
    shape::NTuple{N,Int}
    source::Int
    window::NTuple{N,Int}
    is_output::Bool
end

"""
    ReductionPlan{N}

A topologically ordered DAG (parents before children) of [`ReductionStep`]s producing all requested
output shapes from one input of shape `input_shape`, reusing intermediate accumulators. Purely
geometric — independent of the statistic/eltype, which is applied at compile time.
"""
struct ReductionPlan{N}
    input_shape::NTuple{N,Int}
    steps::Vector{ReductionStep{N}}
    output_steps::Vector{Int}
end

# Largest-factor proper divisor of `f` within `factors` (the optimal parent); `nothing` if none.
function _best_parent(f::NTuple{N,Int}, factors) where {N}
    best = nothing
    bestprod = 0
    for p in factors
        if p != f && divides(p, f)
            pp = prod(p)
            if pp > bestprod
                bestprod = pp
                best = p
            end
        end
    end
    return best
end

# Compute cost of a node at `f` given a materialized factor set (input pass vs cheapest parent).
function _node_cost(X::NTuple{N,Int}, f::NTuple{N,Int}, factors) where {N}
    p = _best_parent(f, factors)
    return p === nothing ? prod(X) : prod(factor_shape(X, p))
end

"""
    total_work(X, factors) -> Int

Total element-touches to materialize every factor in `factors` from one input of shape `X`, under
optimal parent selection. Compare to `length(outputs) * prod(X)` for independent reductions.
"""
total_work(X::NTuple{N,Int}, factors) where {N} = sum(f -> _node_cost(X, f, factors), factors; init = 0)

# Greedily add the best-improving gcd-closure candidate until no candidate lowers total work.
function _augment_steiner(X::NTuple{N,Int}, targets::Vector{NTuple{N,Int}}; cap::Int) where {N}
    M = unique(targets)
    candidates = setdiff(gcd_closure(M; cap = cap), M)
    isempty(candidates) && return M
    while true
        current = total_work(X, M)
        best = nothing
        bestcost = current
        for m in candidates
            c = total_work(X, vcat(M, m))
            if c < bestcost
                bestcost = c
                best = m
            end
        end
        best === nothing && break
        push!(M, best)
        setdiff!(candidates, (best,))
    end
    return M
end

# Assemble a topologically-ordered plan from a materialized factor set and the requested outputs.
function _build_plan(X::NTuple{N,Int}, factors::Vector{NTuple{N,Int}}, outputs::Set{NTuple{N,Int}}) where {N}
    order = sort(unique(factors); by = prod)     # divisors have strictly smaller prod ⇒ appear first
    steps = Vector{ReductionStep{N}}(undef, length(order))
    index_of = Dict{NTuple{N,Int},Int}()
    for (i, f) in enumerate(order)
        p = _best_parent(f, order)
        if p === nothing
            steps[i] = ReductionStep(f, factor_shape(X, f), 0, f, f in outputs)
        else
            steps[i] = ReductionStep(f, factor_shape(X, f), index_of[p], factor_window(p, f), f in outputs)
        end
        index_of[f] = i
    end
    output_steps = [i for (i, s) in enumerate(steps) if s.is_output]
    return ReductionPlan(X, steps, output_steps)
end

# ── Public plan builders ──────────────────────────────────────────────────────

"""
    tower_plan(input_shape; base_factor, steps, maxfactor) -> ReductionPlan

Build a multi-scale tower: enumerate every factor reachable from `base_factor` by the per-dimension
`steps` multipliers up to `maxfactor`, and wire each to its optimal parent. Every reachable shape is
an output.
"""
function tower_plan(input_shape::NTuple{N,Int};
                    base_factor::NTuple{N,Int},
                    steps::NTuple{N,Vector{Int}},
                    maxfactor::NTuple{N,Int}) where {N}
    _validate_factor(input_shape, base_factor)
    factors = reachable_factors(base_factor, steps, maxfactor)
    return _build_plan(input_shape, factors, Set(factors))
end

"""
    solver_plan(input_shape, target_factors; allow_steiner=true, steiner_cap=4096) -> ReductionPlan

Build a minimum-work DAG that produces exactly `target_factors` (each a per-dimension block size).
With `allow_steiner`, the planner may materialize shared finer intermediates (from the targets'
gcd-closure) when doing so lowers total work.
"""
function solver_plan(input_shape::NTuple{N,Int}, target_factors::AbstractVector{NTuple{N,Int}};
                     allow_steiner::Bool = true, steiner_cap::Int = 4096) where {N}
    targets = unique(collect(target_factors))
    for f in targets
        _validate_factor(input_shape, f)
    end
    factors = allow_steiner ? _augment_steiner(input_shape, targets; cap = steiner_cap) : targets
    return _build_plan(input_shape, factors, Set(targets))
end

function _validate_factor(X::NTuple{N,Int}, f::NTuple{N,Int}) where {N}
    for i in 1:N
        1 <= f[i] <= X[i] ||
            throw(ArgumentError("factor $f out of range in dim $i (input shape $X)"))
    end
    return nothing
end

# ── Diagnostics ───────────────────────────────────────────────────────────────

"Number of base passes over the full input (`source == 0` steps) in a plan."
n_base_passes(plan::ReductionPlan) = count(s -> s.source == 0, plan.steps)

"Total element-touches the plan performs (base passes + per-node merges)."
plan_work(plan::ReductionPlan) =
    sum(s -> s.source == 0 ? prod(plan.input_shape) : prod(plan.steps[s.source].shape), plan.steps; init = 0)

"Element-touches a naive independent reduction of each output would cost."
naive_work(plan::ReductionPlan) = length(plan.output_steps) * prod(plan.input_shape)
