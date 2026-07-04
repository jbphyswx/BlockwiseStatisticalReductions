# ─────────────────────────────────────────────────────────────────────────────
# Sequential fold over an N-D box (the generic leaf reduction)
# ─────────────────────────────────────────────────────────────────────────────
#
# `blockfold` merges the accumulators of every index in a box `[lo, hi]` into one running accumulator,
# iterating in COLUMN-MAJOR order (first dimension fastest) so the innermost work walks contiguous
# memory. It serves the generic base pass (leaf = lift a data element) and the generic cross-scale
# merge (leaf = an already-computed accumulator), for ANY user monoid.
#
# Why a flat sequential fold rather than a recursive divide-and-conquer tree:
#   * Speed. A recursive fold that splits the box on its widest dimension pays, at every node,
#     `CartesianIndex` splitting + index-corner reconstruction + a non-inlined recursive call, and —
#     because the widest dim is generally NOT the contiguous (first) dim on column-major arrays — it
#     reads memory with a non-unit stride that defeats SIMD auto-vectorization and cache prefetch.
#     Measured ~8× slower than this flat loop for a small block. (Julia's own `Base.mapreduce_impl`
#     only recurses down to a ~1024-element base case and then runs a flat `@simd` loop; recursing to
#     the leaves is the documented anti-pattern.)
#   * Accuracy needs no tree here. The mergeable monoids used for statistics (Welford mean, Chan
#     variance, Pébay covariance, additive power sums, min/max) are numerically stable under sequential
#     accumulation — that is what Welford/Chan are for. A pairwise tree is only worthwhile for
#     PARALLELISM (across threads / GPU lanes — see the threaded/GPU backends and §5/§8 of
#     docs/DESIGN_OPTIMIZATION.md), which is where BSR's real reduction tree lives.
#
# Order note: the base pass and the cross-scale merge use the SAME column-major fold order, and `merge`
# is associative + commutative up to floating point, so hierarchical reuse (coarsen a fine node ==
# reduce the raw data at the coarse scale) is exact for integer/exact fields and numerically equal
# otherwise; the covered-region identity itself is exact (floor-division composition).
#
# Fast paths for the built-in reducible statistics (SIMD sum/count/min/max, sequential Welford/Chan)
# are layered on top by dispatching the driver on the accumulator type; this generic fold is the
# correct-for-any-monoid fallback.

"""
    blockfold(leaf, lo::CartesianIndex{N}, hi::CartesianIndex{N})

Merge `leaf(J)` over every `J` in the box `lo:hi` into one accumulator, folding sequentially in
column-major (first-dimension-fastest) order. `leaf` returns an accumulator for a single index.
Allocation-free and type-stable when `leaf` returns a concrete isbits accumulator.
"""
@inline function blockfold(leaf::F, lo::CartesianIndex{N}, hi::CartesianIndex{N}) where {F,N}
    acc = leaf(lo)
    @inbounds for J in lo:hi
        J === lo && continue          # `lo` already seeded `acc`; branch is predicted-taken once
        acc = merge(acc, leaf(J))
    end
    return acc
end
