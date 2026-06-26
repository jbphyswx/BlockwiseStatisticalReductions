# ─────────────────────────────────────────────────────────────────────────────
# Pairwise (divide-and-conquer) fold over an N-D box
# ─────────────────────────────────────────────────────────────────────────────
#
# `treefold` merges the accumulators of every index in a box `[lo, hi]` using a balanced binary
# tree (split the widest dimension in half, recurse, merge). This is the "tree reduction" that
# keeps floating-point error at O(log n) instead of O(n), exposes independent sub-reductions to
# the compiler/SIMD, and — because `merge` is associative + commutative — yields exactly the same
# value as any other grouping. The same routine serves the base pass (leaf = lift a data element)
# and the cross-scale merge (leaf = an already-computed accumulator).

# Index of the dimension with the largest extent in box [lo, hi].
@inline function _widest_dim(lo::CartesianIndex{N}, hi::CartesianIndex{N}) where {N}
    d = 1
    best = hi[1] - lo[1]
    for i in 2:N
        s = hi[i] - lo[i]
        if s > best
            best = s
            d = i
        end
    end
    return d
end

# `I` with coordinate `d` replaced by `v`.
@inline _replace(I::CartesianIndex{N}, v::Int, d::Int) where {N} =
    CartesianIndex(ntuple(i -> ifelse(i == d, v, I[i]), Val(N)))

"""
    treefold(leaf, lo::CartesianIndex{N}, hi::CartesianIndex{N})

Merge `leaf(J)` over every `J` in the box `lo:hi` via a balanced binary tree. `leaf` returns an
accumulator for a single index; the result is their `merge`. Allocation-free and type-stable when
`leaf` returns a concrete isbits accumulator.
"""
function treefold(leaf::F, lo::CartesianIndex{N}, hi::CartesianIndex{N}) where {F,N}
    lo == hi && return leaf(lo)
    d = _widest_dim(lo, hi)
    mid = (lo[d] + hi[d]) >> 1
    left = treefold(leaf, lo, _replace(hi, mid, d))
    right = treefold(leaf, _replace(lo, mid + 1, d), hi)
    return merge(left, right)
end
