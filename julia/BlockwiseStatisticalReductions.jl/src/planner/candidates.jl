# Candidate intermediate windows the solver may materialize to share work between targets.

_is_origin_tiled(aw::AxisWindow) = is_tiled(aw) && aw.pos.offset == 0
_policy(partial::Bool) = partial ? Partial() : Truncate()
_identity(extent::Int) = AxisWindow(extent, 1, Progression(0, 1, extent), false)
_is_identity(aw::AxisWindow) = aw.size == 1 && aw.pos == Progression(0, 1, aw.extent)
_is_dense(aw::AxisWindow) = aw.pos isa Progression && aw.pos.stride == 1 && aw.pos.offset == 0

"Tiled windows at the elementwise gcd closure of the tiled targets' sizes (per edge policy)."
function tile_candidates(targets::Vector{Window{N}}, shape::NTuple{N,Int}) where {N}
    out = Window{N}[]
    for partial in (false, true)
        sizes = Set{NTuple{N,Int}}()
        for w in targets
            all(a -> _is_origin_tiled(a) && a.partial == partial, w) && push!(sizes, map(a -> a.size, w))
        end
        length(sizes) < 2 && continue
        closure = copy(sizes)
        changed = true
        while changed && length(closure) < 512
            changed = false
            for a in collect(closure), b in collect(closure)
                g = map(gcd, a, b)
                if g ∉ closure
                    push!(closure, g)
                    changed = true
                end
            end
        end
        for s in closure
            s in sizes && continue
            push!(out, tiled(shape, s, _policy(partial)))
        end
    end
    return out
end

"Dense doubling chains along each axis toward every target, one axis completed at a time."
function chain_candidates(targets::Vector{Window{N}}, shape::NTuple{N,Int}) where {N}
    out = Set{Window{N}}()
    for w in targets
        for d in 1:N
            s = w[d].size
            s > 1 || continue
            prefix = ntuple(e -> w[e], d - 1)
            suffix = ntuple(e -> _identity(shape[d+e]), N - d)
            j = 1
            while j < s
                push!(out, (prefix..., strided(shape[d], j, 1, Truncate()), suffix...))
                j *= 2
            end
            push!(out, (prefix..., strided(shape[d], s, 1, Truncate()), suffix...))
            push!(out, (prefix..., w[d], suffix...))
        end
    end
    return collect(out)
end

"Tiled intermediates that keep base boxes at or under `max_tile_elements` cells."
function splitter_candidates(targets::Vector{Window{N}}, shape::NTuple{N,Int}, max_tile_elements::Int) where {N}
    out = Window{N}[]
    for w in targets
        all(_is_origin_tiled, w) || continue
        s = collect(map(a -> a.size, w))
        prod(s) <= max_tile_elements && continue
        partial = w[1].partial
        while prod(s) > max_tile_elements
            d = argmax(s)
            f = _smallest_factor(s[d])
            f == 1 && break
            s[d] ÷= f
            push!(out, tiled(shape, Tuple(s), _policy(partial)))
        end
    end
    return out
end
function _smallest_factor(n::Int)
    n <= 1 && return 1
    for f in 2:isqrt(n)
        n % f == 0 && return f
    end
    return n
end

"All candidate windows for `targets`, excluding the targets themselves."
function candidates(targets::Vector{Window{N}}, shape::NTuple{N,Int}, limits::KernelLimits; chains::Bool = true) where {N}
    cands = Window{N}[]
    append!(cands, tile_candidates(targets, shape))
    chains && append!(cands, chain_candidates(targets, shape))
    append!(cands, splitter_candidates(targets, shape, limits.max_tile_elements))
    unique!(cands)
    filter!(w -> !(w in targets) && all(aw -> nwindows(aw) > 0, w), cands)
    return cands
end
