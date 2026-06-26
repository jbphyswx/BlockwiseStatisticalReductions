```@meta
CurrentModule = BlockwiseStatisticalReductions
```

# The Tower & Lattice

A reduction is described by a per-dimension **factor** `f` (block size): it reduces input of shape
`X` to `X .÷ f` (floor / low-index truncation). Factors form a lattice under per-dimension
divisibility, and that structure is what lets a whole family of scales share work.

## Reuse condition

A node at factor `c` can reuse a node at factor `p` iff `f_p` divides `f_c` in every dimension; the
merge window is `f_c ./ f_p`. Crucially, floor division composes —

```
(X ÷ f_p) ÷ (f_c ÷ f_p) == X ÷ f_c
```

— so coarsening a finer node by `f_c ./ f_p` produces exactly the same accumulators (same covered
region) as a fresh reduction at `f_c`, even when `X` is not divisible by the factors. Reuse is
*exact*, not approximate.

## Cost model and optimal parents

Producing a node from a source touches every element of the *source* once:

```
cost(from input)    = prod(X)              # one full pass over the data
cost(from parent p) = prod(X .÷ p)         # one pass over the (smaller) parent
```

So the cheapest way to build `c` is from the materialized proper divisor with the **largest** factor
(the smallest array). Choosing that parent for every node is optimal. Along a 1-D chain the total
telescopes below `2·prod(X)`; a full N-D tower is `O(2^N·prod(X))` — independent of the number of
levels, and far below the naive `(#scales)·prod(X)` of independent reductions.

## Building plans

```julia
# enumerate every factor reachable from base_factor by the per-level multipliers, up to maxfactor
plan = tower_plan((512, 512); base_factor = (2, 2), steps = ([2, 3], [2, 3]), maxfactor = (128, 128))

# or request specific targets; the planner shares finer intermediates (gcd-closure) when it lowers cost
plan = solver_plan((360, 360), [(4, 4), (6, 6)])     # materializes (2,2) once, both reuse it
```

A [`ReductionPlan`](@ref) is pure geometry — a topologically ordered list of [`ReductionStep`](@ref)s
(each a base pass or a coarsening of a parent), independent of the statistic and eltype. Inspect its
efficiency with [`plan_work`](@ref) vs [`naive_work`](@ref):

```julia
plan_work(plan) / naive_work(plan)     # ≈ 1/(#scales): the data is touched about once
```

The statistic is applied at execution: [`allocate_tower`](@ref) sizes one accumulator buffer per
step, [`run!`](@ref) fills them (base ⇒ `blockreduce!`, otherwise `coarsen!`), and `M2`/`C`
numerators are converted to variance/covariance only at the requested output nodes.
