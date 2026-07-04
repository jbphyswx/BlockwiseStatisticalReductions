# ─────────────────────────────────────────────────────────────────────────────
# Gap-filling multi-scale schedules
# ─────────────────────────────────────────────────────────────────────────────
#
# `Tower` reaches factors only by multiplying `base_factor` by its `steps`, so a dyadic tower
# (steps=[2]) produces factors 2,4,8,16,… with GAPS at 3,5,6,7,9,…. To densely span a range of
# scales you want the *multiplicative closure* of a set of multipliers: start from the seeds and
# repeatedly multiply by any allowed step. With steps `[2,3]` this is the set of 2,3-smooth numbers
# 1,2,3,4,6,8,9,12,16,18,24,27,32,… — the "full reduction tree" that fills every gap reachable by
# 2× and 3× coarsenings (9 = 3², 18 = 2·3², 27 = 3³, …). With a single step it degenerates to a
# geometric ladder (dyadic when steps=[2]); several seeds under one step reproduce the classic
# merged-dyadic schedule (seeds=[1,3], steps=[2] → 1,2,3,4,6,8,12,16,24,32,48,64,96,128).
#
# This is the 1-D analogue of `reachable_factors` (the lattice BFS the planner already uses). The
# resulting factor list is purely integer/geometric (no domain concepts); `Ladder` wraps it into a
# scale-spec that resolves to a target-factor set handed to `solver_plan`, which builds the optimal
# shared DAG over that set.

"""
    scale_ladder(n; seeds=[1], steps=[2], minfactor=1, maxfactor=n, include_full=false) -> Vector{Int}

A gap-filling 1-D ladder of reduction factors: the multiplicative closure of `seeds` under the
multipliers `steps`, i.e. every value reachable by starting from a seed and repeatedly multiplying
by a step, kept within `[minfactor, min(maxfactor, n)]`, deduplicated and sorted ascending. With
`include_full`, `min(maxfactor, n)` is appended so the coarsest scale is always present.

`steps` controls how densely the range is filled:
  * `steps=[2]` — dyadic geometric ladder (`1,2,4,8,…`); several seeds merge dyadic ladders.
  * `steps=[2,3]` — the **full reduction tree** of 2,3-smooth factors (fills 9, 18, 27, …).

```julia
scale_ladder(128)                          # [1, 2, 4, 8, 16, 32, 64, 128]                (dyadic)
scale_ladder(128; seeds=[1,3])             # [1,2,3,4,6,8,12,16,24,32,48,64,96,128]       (merged dyadic)
scale_ladder(128; steps=[2,3])             # [1,2,3,4,6,8,9,12,16,18,24,27,32,36,48,54,64,72,81,96,108,128]  (full tree)
scale_ladder(60; steps=[2,3], minfactor=2, include_full=true)
scale_ladder(81; steps=[3])                # [1, 3, 9, 27, 81]
```

This is the 1-D companion of [`reachable_factors`](@ref); see [`Ladder`](@ref) to use it as a scale
specification with [`reduce_stats`](@ref).
"""
function scale_ladder(n::Integer; seeds = [1], steps = [2],
                      minfactor::Integer = 1, maxfactor::Integer = n,
                      include_full::Bool = false)
    n >= 1 || throw(ArgumentError("n must be ≥ 1"))
    minfactor >= 1 || throw(ArgumentError("minfactor must be ≥ 1"))
    all(s -> s >= 1, seeds) || throw(ArgumentError("all seeds must be ≥ 1"))
    all(s -> s >= 2, steps) || throw(ArgumentError("all steps must be ≥ 2"))
    hi = min(Int(maxfactor), Int(n))
    seen = Set{Int}()
    queue = Int[]
    for s in seeds
        si = Int(s)
        if si <= hi && si ∉ seen
            push!(seen, si)
            push!(queue, si)
        end
    end
    head = 1
    while head <= length(queue)
        f = queue[head]
        head += 1
        for st in steps
            g = f * Int(st)
            if g <= hi && g ∉ seen
                push!(seen, g)
                push!(queue, g)
            end
        end
    end
    factors = filter(f -> f >= minfactor, collect(seen))
    include_full && hi >= minfactor && push!(factors, hi)
    sort!(unique!(factors))
    return factors
end

"""
    Ladder(; seeds=[1], steps=[2], minfactor=1, maxfactor=nothing, include_full=false, combine=:isotropic)

A multi-scale schedule spec (like [`Tower`](@ref)/[`Sliding`](@ref)) that resolves to a set of
target factors via [`scale_ladder`](@ref) and hands them to [`solver_plan`](@ref), which builds a
shared minimum-work DAG. Use `steps=[2,3]` for the full 2,3-smooth reduction tree, `steps=[2]` for a
dyadic ladder. `maxfactor=nothing` means each dimension's extent; exclude a dimension by setting its
`maxfactor` to `1`. Each of `seeds`/`steps`/`minfactor`/`maxfactor` may be a scalar (every dimension)
or an `NTuple`/tuple-of-vectors (per dimension).

`combine`:
  * `:isotropic` (default) — one shared 1-D ladder (dim-1 params), each factor `v` broadcast to every
    dimension clamped to that dim's `maxfactor`. Reproduces `[(v,v,…) for v in scale_ladder(...)]`.
  * `:product` — full per-dimension ladders, Cartesian-producted (anisotropic; can be large).

Relationship to `Tower`: `Ladder(...; combine=:isotropic)` gives only the isotropic factors `(v,v,…)`
of the 1-D closure; `Tower(base_factor=1, steps=[2,3])` gives the full *anisotropic* N-D reachable
lattice (all mixed combinations). Use `Ladder` when you want a clean list of isotropic scales, `Tower`
when you want every per-dimension combination.

Note (like `Tower.steps`): `seeds`/`steps` as a `Vector` apply to *every* dimension; a `Tuple` of
vectors is *per-dimension*.
"""
struct Ladder{SE,ST,MN,MX}
    seeds::SE
    steps::ST
    minfactor::MN
    maxfactor::MX
    include_full::Bool
    combine::Symbol
end
Ladder(; seeds = [1], steps = [2], minfactor = 1, maxfactor = nothing,
       include_full = false, combine = :isotropic) =
    Ladder(seeds, steps, minfactor, maxfactor, include_full, combine)

# Normalize seeds/steps to NTuple{N,Vector{Int}} (scalar ⇒ single value broadcast to every dim).
_ladder_mults(s::Integer, ::Val{N}) where {N} = ntuple(_ -> [Int(s)], Val(N))
_ladder_mults(s, v::Val{N}) where {N} = _perdim_steps(s, v)

function _resolve_ladder(l::Ladder, X::NTuple{N,Int}) where {N}
    seeds = _ladder_mults(l.seeds, Val(N))                       # NTuple{N,Vector{Int}}
    steps = _ladder_mults(l.steps, Val(N))                       # NTuple{N,Vector{Int}}
    minf = _perdim(l.minfactor, Val(N))
    maxf = l.maxfactor === nothing ? X : _perdim(l.maxfactor, Val(N))
    return seeds, steps, minf, maxf
end

# Resolve a Ladder to the set of target factors (dropping the all-ones identity factor).
function _ladder_targets(l::Ladder, X::NTuple{N,Int}) where {N}
    seeds, steps, minf, maxf = _resolve_ladder(l, X)
    if l.combine === :product
        perdim = ntuple(d -> scale_ladder(X[d]; seeds = seeds[d], steps = steps[d],
                                          minfactor = minf[d], maxfactor = maxf[d],
                                          include_full = l.include_full), Val(N))
        targets = NTuple{N,Int}[NTuple{N,Int}(t) for t in Iterators.product(perdim...)]
    elseif l.combine === :isotropic
        hi = maximum(maxf)
        S = scale_ladder(hi; seeds = seeds[1], steps = steps[1], minfactor = minf[1],
                         maxfactor = hi, include_full = l.include_full)
        targets = NTuple{N,Int}[ntuple(d -> clamp(v, 1, maxf[d]), Val(N)) for v in S]
    else
        throw(ArgumentError("Ladder combine must be :isotropic or :product, got :$(l.combine)"))
    end
    return unique(filter(f -> !all(isone, f), targets))
end

_plan_for(X::NTuple{N,Int}, l::Ladder) where {N} = solver_plan(X, _ladder_targets(l, X))
