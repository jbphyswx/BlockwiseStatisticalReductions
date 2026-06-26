# ─────────────────────────────────────────────────────────────────────────────
# The divisor lattice of reduction factors
# ─────────────────────────────────────────────────────────────────────────────
#
# A reduction is described by a per-dimension *factor* `f` (block size): a node reduces the input
# of shape `X` to `shape(f) = X .÷ f` (floor / low-index truncation). The set of factors forms a
# lattice under per-dimension divisibility: `c` is reachable from `p` (and can reuse `p`'s
# accumulators) iff `f_p` divides `f_c` in every dimension, with merge window `f_c ./ f_p`.
#
# Floor division composes — `(X ÷ f_p) ÷ (f_c ÷ f_p) == X ÷ f_c` — so coarsening a materialized
# finer node by `f_c ./ f_p` yields exactly the same accumulators (same covered region) as a fresh
# reduction at factor `f_c`. That identity is what makes hierarchical reuse exact even when `X` is
# not divisible by the factors.

"Output shape produced by reduction factor `f` on input of shape `X` (floor / truncating)."
@inline factor_shape(X::NTuple{N,Int}, f::NTuple{N,Int}) where {N} = ntuple(i -> X[i] ÷ f[i], Val(N))

"`true` if factor `a` divides factor `b` in every dimension (so `b` can reuse `a`)."
@inline divides(a::NTuple{N,Int}, b::NTuple{N,Int}) where {N} = all(ntuple(i -> b[i] % a[i] == 0, Val(N)))

"Merge window taking a node at factor `a` to its coarser child at factor `b` (requires `divides(a,b)`)."
@inline factor_window(a::NTuple{N,Int}, b::NTuple{N,Int}) where {N} = ntuple(i -> b[i] ÷ a[i], Val(N))

"""
    reachable_factors(base, steps, maxfactor) -> Vector{NTuple{N,Int}}

Enumerate every factor reachable from `base` by repeatedly multiplying a single dimension by one of
its allowed `steps[d]` multipliers, staying within `f[d] ≤ maxfactor[d]`. Combined multi-dimension
coarsenings appear naturally (multiply one dim, then another). This is the tower lattice; the
planner then wires optimal parents. Returns factors in BFS-discovery order (always includes `base`).
"""
function reachable_factors(base::NTuple{N,Int}, steps::NTuple{N,Vector{Int}}, maxfactor::NTuple{N,Int}) where {N}
    seen = Set{NTuple{N,Int}}((base,))
    queue = NTuple{N,Int}[base]
    out = NTuple{N,Int}[base]
    head = 1
    while head <= length(queue)
        f = queue[head]; head += 1
        for d in 1:N
            for s in steps[d]
                s <= 1 && continue
                fd = f[d] * s
                fd <= maxfactor[d] || continue
                g = ntuple(i -> i == d ? fd : f[i], Val(N))
                if g ∉ seen
                    push!(seen, g); push!(queue, g); push!(out, g)
                end
            end
        end
    end
    return out
end

"""
    gcd_closure(targets; cap=4096) -> Vector{NTuple{N,Int}}

Per-dimension gcd-closure of `targets`: the smallest superset closed under elementwise `gcd`. These
are the only factors worth considering as shared (Steiner) intermediates, since a useful shared
parent must divide every target it serves. Excludes the all-ones factor (the input itself). Returns
at most `cap` factors (closure is tiny in practice for product-of-chains lattices).
"""
function gcd_closure(targets::AbstractVector{NTuple{N,Int}}; cap::Int = 4096) where {N}
    closure = Set{NTuple{N,Int}}(targets)
    changed = true
    while changed && length(closure) < cap
        changed = false
        cur = collect(closure)
        for a in cur, b in cur
            g = ntuple(i -> gcd(a[i], b[i]), Val(N))
            if g ∉ closure
                push!(closure, g)
                changed = true
                length(closure) >= cap && break
            end
        end
    end
    return filter(f -> !all(isone, f), collect(closure))
end
