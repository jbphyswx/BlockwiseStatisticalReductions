"""
    resolve(spec, shape::NTuple{N,Int}; spacing = nothing, dimnames = nothing) -> Vector{Window{N}}

Windows requested by `spec` on an array of `shape`, deduplicated and sorted by volume. `spec` is a
[`ScaleSet`](@ref), a vector of isotropic sizes, a vector of per-axis size tuples, a single size or size
tuple, or windows. `spacing` (one `AxisSpacing` or `nothing` per axis) resolves `Length` bounds;
`dimnames` matches a `NamedTuple` of generators to axes.
"""
function resolve(spec::ScaleSet, shape::NTuple{N,Int}; edge::EdgePolicy = spec.edge, spacing = nothing, dimnames = nothing) where {N}
    gens = _axis_generators(spec.axes, Val(N), dimnames)
    sp = _axis_spacings(spacing, Val(N))
    sets = ntuple(d -> generate(gens[d], shape[d], sp[d]), Val(N))
    reduced = ntuple(d -> !(gens[d] isa Fixed), Val(N))
    tuples = _combine(spec.combine, sets, reduced, dimnames)
    if spec.include_full && all(d -> reduced[d] || !isempty(sets[d]), 1:N)
        push!(tuples, ntuple(d -> reduced[d] ? shape[d] : sets[d][1], Val(N)))
    end
    placements = _axis_placements(spec.placement, Val(N))
    windows = Window{N}[]
    for sizes in unique(tuples)
        all(d -> 1 <= sizes[d] <= shape[d], 1:N) || continue
        w = ntuple(d -> place(placements[d], shape[d], sizes[d], edge), Val(N))
        all(aw -> nwindows(aw) > 0, w) || continue
        spec.min_elements <= volume(w) <= spec.max_elements || continue
        spec.filter(w) || continue
        push!(windows, w)
    end
    unique!(windows)
    isempty(windows) && throw(ArgumentError("no windows satisfy the scale specification on shape $shape"))
    sort!(windows; by = w -> (volume(w), map(aw -> aw.size, w)))
    return windows
end

resolve(sizes::AbstractVector{<:Integer}, shape::NTuple{N,Int}; edge::EdgePolicy = Truncate(), kw...) where {N} =
    resolve(ScaleSet(Sizes(sizes); combine = Isotropic(), edge = edge), shape; kw...)
resolve(size::Integer, shape::NTuple{N,Int}; kw...) where {N} = resolve([size], shape; kw...)
resolve(sizes::Tuple{Integer,Vararg{Integer}}, shape::NTuple{N,Int}; kw...) where {N} = resolve([sizes], shape; kw...)
function resolve(tuples::AbstractVector{<:Tuple{Integer,Vararg{Integer}}}, shape::NTuple{N,Int};
                 edge::EdgePolicy = Truncate(), placement = Tiled(), kw...) where {N}
    placements = _axis_placements(placement, Val(N))
    windows = Window{N}[]
    for sizes in unique(tuples)
        length(sizes) == N || throw(DimensionMismatch("window sizes $sizes for $N axes"))
        all(d -> 1 <= sizes[d] <= shape[d], 1:N) ||
            throw(ArgumentError("window sizes $sizes do not fit shape $shape"))
        push!(windows, ntuple(d -> place(placements[d], shape[d], Int(sizes[d]), edge), Val(N)))
    end
    unique!(windows)
    sort!(windows; by = w -> (volume(w), map(aw -> aw.size, w)))
    return windows
end
function resolve(w::Tuple{AxisWindow,Vararg{AxisWindow}}, shape::NTuple{N,Int}; kw...) where {N}
    length(w) == N || throw(DimensionMismatch("window with $(length(w)) axes for $N axes"))
    map(aw -> aw.extent, w) == shape || throw(DimensionMismatch("window extents $(map(aw -> aw.extent, w)) do not match shape $shape"))
    return Window{N}[w]
end
resolve(ws::AbstractVector{<:Tuple{AxisWindow,Vararg{AxisWindow}}}, shape::NTuple{N,Int}; kw...) where {N} =
    unique!(reduce(vcat, (resolve(w, shape) for w in ws); init = Window{N}[]))

"""
    Resolved(windows)

Target windows worked out by the caller. [`resolve`](@ref) returns them in the given order and keeps
every one of them, so a caller that has to line its windows up with something else keeps its own
ordering. The windows must be distinct, since a plan produces each of its targets once.
"""
struct Resolved{W<:AbstractVector}
    windows::W
end
function resolve(r::Resolved, shape::NTuple{N,Int}; kw...) where {N}
    for w in r.windows
        length(w) == N || throw(DimensionMismatch("window with $(length(w)) axes for $N axes"))
        map(aw -> aw.extent, w) == shape ||
            throw(DimensionMismatch("window extents $(map(aw -> aw.extent, w)) do not match shape $shape"))
    end
    allunique(map(canonicalize, w) for w in r.windows) || throw(ArgumentError("resolved targets must be distinct"))
    return Window{N}[w for w in r.windows]
end

# ── Per-axis generators ──────────────────────────────────────────────────────────

_generator(g::SizeGenerator) = g
_generator(s::Integer) = Fixed(Int(s))
_generator(v::AbstractVector{<:Integer}) = Sizes(v)

_axis_generators(axes::SizeGenerator, ::Val{N}, dimnames) where {N} = ntuple(_ -> axes, Val(N))
_axis_generators(axes::Integer, ::Val{N}, dimnames) where {N} = ntuple(_ -> Fixed(Int(axes)), Val(N))
function _axis_generators(axes::Tuple, ::Val{N}, dimnames) where {N}
    length(axes) == N || throw(ArgumentError("$(length(axes)) axis generators for $N axes"))
    return map(_generator, axes)
end
function _axis_generators(axes::NamedTuple, ::Val{N}, dimnames) where {N}
    dimnames === nothing && throw(ArgumentError("a NamedTuple of axis generators needs dimnames"))
    length(dimnames) == N || throw(ArgumentError("$(length(dimnames)) dimnames for $N axes"))
    for k in keys(axes)
        k in dimnames || throw(ArgumentError("unknown axis $k; axes are $dimnames"))
    end
    return ntuple(d -> haskey(axes, dimnames[d]) ? _generator(axes[dimnames[d]]) : Fixed(1), Val(N))
end

_axis_spacings(::Nothing, ::Val{N}) where {N} = ntuple(_ -> nothing, Val(N))
_axis_spacings(sp::AxisSpacing, ::Val{N}) where {N} = ntuple(_ -> sp, Val(N))
function _axis_spacings(sp::Tuple, ::Val{N}) where {N}
    length(sp) == N || throw(ArgumentError("$(length(sp)) spacings for $N axes"))
    return sp
end

_lower(::Nothing, extent, sp) = 1
_lower(x::Integer, extent, sp) = Int(x)
_lower(L::Length, extent, sp) = cells_at_least(_need_spacing(sp), extent, L)
_upper(::Nothing, extent, sp) = extent
_upper(x::Integer, extent, sp) = Int(x)
_upper(L::Length, extent, sp) = cells_at_most(_need_spacing(sp), extent, L)
_need_spacing(::Nothing) = throw(ArgumentError("a Length bound needs the axis spacing"))
_need_spacing(sp::AxisSpacing) = sp
_bounds(g, extent, sp) = (max(1, _lower(g.min, extent, sp)), min(extent, _upper(g.max, extent, sp)))

"Sizes (cells) a generator yields on an axis of `extent` cells with spacing `sp`; sorted, within the axis."
generate(g::Sizes, extent::Int, sp) = filter(s -> 1 <= s <= extent, sort!(unique(Int.(g.sizes))))
function generate(g::Dyadic, extent::Int, sp)
    lo, hi = _bounds(g, extent, sp)
    return [1 << k for k in 0:63 if lo <= 1 << k <= hi]
end
function generate(g::Smooth, extent::Int, sp)
    lo, hi = _bounds(g, extent, sp)
    all(p -> p >= 2, g.primes) || throw(ArgumentError("Smooth factors must be ≥ 2"))
    found = Set{Int}(1)
    queue = [1]
    while !isempty(queue)
        s = pop!(queue)
        for p in g.primes
            t = s * p
            if t <= hi && t ∉ found
                push!(found, t)
                push!(queue, t)
            end
        end
    end
    return sort!(filter(s -> s >= lo, collect(found)))
end
function generate(g::Every, extent::Int, sp)
    lo, hi = _bounds(g, extent, sp)
    return collect(lo:hi)
end
function generate(g::Divisors, extent::Int, sp)
    lo, hi = _bounds(g, extent, sp)
    return [s for s in lo:hi if extent % s == 0]
end
generate(g::Fixed, extent::Int, sp) = 1 <= g.size <= extent ? [g.size] : Int[]
function generate(g::Subsample, extent::Int, sp)
    v = generate(g.gen, extent, sp)
    g.budget >= 1 || throw(ArgumentError("Subsample budget must be ≥ 1"))
    n = length(v)
    n <= g.budget && return v
    g.budget == 1 && return [v[cld(n, 2)]]
    return unique!([v[1 + round(Int, (i - 1) * (n - 1) / (g.budget - 1))] for i in 1:g.budget])
end

# ── Combination ──────────────────────────────────────────────────────────────────

_combine(::Product, sets::NTuple{N,Vector{Int}}, reduced, dimnames) where {N} =
    NTuple{N,Int}[t for t in Iterators.product(sets...)]
function _combine(::Zip, sets::NTuple{N,Vector{Int}}, reduced, dimnames) where {N}
    lengths = unique([length(sets[d]) for d in 1:N if reduced[d]])
    length(lengths) <= 1 || throw(ArgumentError("Zip needs equally many sizes on every reduced axis, got $lengths"))
    any(d -> !reduced[d] && isempty(sets[d]), 1:N) && return NTuple{N,Int}[]
    m = isempty(lengths) ? 1 : lengths[1]
    return NTuple{N,Int}[ntuple(d -> reduced[d] ? sets[d][i] : sets[d][1], Val(N)) for i in 1:m]
end
function _combine(iso::Isotropic, sets::NTuple{N,Vector{Int}}, reduced, dimnames) where {N}
    coupled = _coupled_axes(iso.axes, reduced, dimnames, Val(N))
    any(coupled) || return _combine(Product(), sets, reduced, dimnames)
    common = sort!(collect(reduce(intersect, (Set(sets[d]) for d in 1:N if coupled[d]))))
    others = ntuple(d -> coupled[d] ? [0] : sets[d], Val(N))
    return NTuple{N,Int}[ntuple(d -> coupled[d] ? s : t[d], Val(N)) for s in common for t in Iterators.product(others...)]
end

_coupled_axes(::Nothing, reduced, dimnames, ::Val{N}) where {N} = reduced
function _coupled_axes(axes::Tuple, reduced, dimnames, ::Val{N}) where {N}
    positions = map(a -> _axis_position(a, dimnames, N), axes)
    return ntuple(d -> d in positions, Val(N))
end
_axis_position(a::Integer, dimnames, N) = (1 <= a <= N || throw(ArgumentError("axis $a out of range for $N axes")); Int(a))
function _axis_position(a::Symbol, dimnames, N)
    dimnames === nothing && throw(ArgumentError("coupling axis $a by name needs dimnames"))
    i = findfirst(==(a), dimnames)
    i === nothing && throw(ArgumentError("unknown axis $a; axes are $dimnames"))
    return i
end

# ── Placement ────────────────────────────────────────────────────────────────────

_axis_placements(p::Placement, ::Val{N}) where {N} = ntuple(_ -> p, Val(N))
function _axis_placements(p::Tuple, ::Val{N}) where {N}
    length(p) == N || throw(ArgumentError("$(length(p)) placements for $N axes"))
    return p
end

"The `AxisWindow` a placement produces for windows of `size` on an axis of `extent` cells."
place(::Tiled, extent::Int, size::Int, edge::EdgePolicy) = tiled(extent, size, edge)
place(p::Stride, extent::Int, size::Int, edge::EdgePolicy) = strided(extent, size, p.stride, edge)
place(p::Overlap, extent::Int, size::Int, edge::EdgePolicy) =
    strided(extent, size, max(1, round(Int, size * (1 - p.fraction))), edge)
place(::Dense, extent::Int, size::Int, edge::EdgePolicy) = strided(extent, size, 1, edge)
place(p::Anchors, extent::Int, size::Int, edge::EdgePolicy) = anchored(extent, size, p.origins; partial = edge isa Partial)
function place(p::Spread, extent::Int, size::Int, edge::EdgePolicy)
    p.count >= 1 || throw(ArgumentError("Spread count must be ≥ 1"))
    last = extent - size
    last < 0 && return anchored(extent, size, Int[])
    k = min(p.count, last + 1)
    origins = k == 1 ? [0] : unique!([(i * last) ÷ (k - 1) for i in 0:k-1])
    return anchored(extent, size, origins)
end
