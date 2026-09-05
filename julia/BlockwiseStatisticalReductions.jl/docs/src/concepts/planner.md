# The planner

The planner is the reason this package exists. Given a set of target windows, it finds a cheap way to
produce all of them together.

![The plan for five tile sizes, against one pass each](../assets/plan.png)

## Nodes and derivations

A node is a window. An edge says how that node is computed:

| Derivation | Meaning | Cost |
|---|---|---|
| `Base_` | reduce raw input cells inside every window | reads the input |
| `Coarsen` | merge `k` consecutive windows of a tiled parent, per axis | reads the parent |
| `Compose` | merge parent `a` at each origin with parent `b` at origin + `a.size` | reads two parents |
| `Restride` | select a subset of a parent's windows | free — it is a view |
| `Scan` | dense sliding windows along an axis, two-stack | reads the parent once |

Every edge is legal exactly when the matching predicate from [windows](windows.md) holds on every axis, so
a plan never contains a merge that has not been proven aligned.

## Why it spans every size

Two facts do all the work:

- **Doubling.** A dense chain `D₀ = lift(x)`, `Dⱼ₊₁[i] = Dⱼ[i] ⊕ Dⱼ[i + 2ʲ]` costs one merge per cell per
  level and gives every power-of-two window size.
- **Composition.** A window of size `s₁ + s₂` is the size-`s₁` window merged with the size-`s₂` window
  shifted by `s₁`, whenever `s₁` is a multiple of the stride so the positions stay on the grid.

Together: window size 12 is `D₃ ⊕ D₂` shifted by 8 — one merge per output. Size 13 is two. Any size costs
`popcount(size) - 1` merges per output from the chain, and a *tile* of that size is a free `Restride` of
the dense node.

Divisor coarsening remains the cheapest special case and the planner prefers it where it applies, because
a coarsened parent has far fewer cells than a dense one.

## Choosing

Cost is roofline: `max(bytes / bandwidth, merges / merge_rate, serial_merges / serial_merge_rate)`, with
the rates coming from `kernel_limits(backend, N)`. The search is greedy Steiner — repeatedly add the
candidate intermediate that lowers total cost the most — over candidates built from the gcd-closure of the
tiled targets, per-axis doubling chains (raw and on each coarser grid), and splitters that keep base boxes
inside the backend's tile limit. Splitters are mandatory, not optional: a target whose window exceeds the
limit never gets a `Base_` derivation.

Then a topological order, and a liveness pass that assigns physical buffers so peak memory is the maximum
live set rather than the sum of all nodes.

Because `kernel_limits` differs per backend, so does the plan. One plan run on two backends is bitwise
identical; the same *request* on two backends agrees to rounding.

## Reading a plan

```julia
p = prepare(x, [2, 4, 8, 16, 32, 64]; stats = (Mean(), Var()))
explain(p)
```

reports the derivation of every node, the number of passes over the input, bytes read and written, merges,
and peak workspace bytes.

## What it will not do

Summed-area tables and other prefix-difference tricks are rejected on purpose. They need `unmerge`, and
subtracting accumulators is exactly the cancellation-prone form the [numerics](numerics.md) policy rules
out for means, variances and covariances. `unmerge` stays in the algebra for callers who want it; it never
appears in a plan.
