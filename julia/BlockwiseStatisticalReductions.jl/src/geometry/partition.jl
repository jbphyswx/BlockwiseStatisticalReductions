# A tensor partitioned along one axis: every rank holds a contiguous slab and the global tensor is the
# concatenation of the slabs in rank order. A window belongs to the slab its first cell falls in, so a
# window only ever reaches forward, into the slabs after its owner's — one halo, on one side.

"""
    Slab(axis, offset, extent, global_extent)

One share of a tensor partitioned along `axis`: `extent` cells starting `offset` cells into an axis of
`global_extent` cells.
"""
struct Slab
    axis::Int
    offset::Int
    extent::Int
    global_extent::Int
    function Slab(axis::Integer, offset::Integer, extent::Integer, global_extent::Integer)
        axis >= 1 || throw(ArgumentError("axis must be ≥ 1, got $axis"))
        offset >= 0 || throw(ArgumentError("offset must be ≥ 0, got $offset"))
        extent >= 0 || throw(ArgumentError("extent must be ≥ 0, got $extent"))
        offset + extent <= global_extent ||
            throw(ArgumentError("slab [$offset, $(offset + extent)) does not fit an axis of $global_extent cells"))
        return new(Int(axis), Int(offset), Int(extent), Int(global_extent))
    end
end

"One past the last cell (0-based) the window of `aw` at origin `o` covers."
@inline window_stop(aw::AxisWindow, o::Integer) = min(Int(o) + aw.size, aw.extent)

"""
    split_origins(aw::AxisWindow, s::Slab) -> (interior, boundary)

Global origins of `aw` whose first cell lies in `s`, as those whose window also ends inside `s` and those
that reach past it. `window_stop` grows with the origin, so the interior ones come first.
"""
function split_origins(aw::AxisWindow, s::Slab)
    stop = s.offset + s.extent
    owned = filter(o -> s.offset <= o < stop, collect(origins(aw)))
    k = count(o -> window_stop(aw, o) <= stop, owned)
    return owned[1:k], owned[k+1:end]
end

_replace_axis(w::Window{N}, d::Integer, aw::AxisWindow) where {N} = ntuple(i -> i == d ? aw : w[i], Val(N))

# The window of `aw` at the given global origins, re-expressed on an axis of `extent` cells whose first
# cell is global cell `base`. A window is marked partial exactly when it runs off that axis, which for
# both the slab and the staging region happens only where the global window was itself clipped.
function _axis_window(aw::AxisWindow, extent::Int, os::Vector{Int}, base::Int)
    local_os = os .- base
    return AxisWindow(extent, aw.size, _as_positions(local_os, aw.pos), any(o -> o + aw.size > extent, local_os))
end

"""
    partition(targets, s::Slab) -> NamedTuple

How a slab divides a resolved request. Returns the `owned` windows in global coordinates, the `interior`
windows in slab-local coordinates, the `boundary` windows in staging-local coordinates, the 0-based global
cell `staging` range the boundary windows read, and the number of interior windows `split` in each target
— the point along the partitioned axis where each output array changes hands.
"""
function partition(targets::AbstractVector{<:Window{N}}, s::Slab) where {N}
    for w in targets
        w[s.axis].extent == s.global_extent ||
            throw(DimensionMismatch("target extent $(w[s.axis].extent) on axis $(s.axis) is not the global extent $(s.global_extent)"))
    end
    parts = [split_origins(w[s.axis], s) for w in targets]
    base = s.offset + s.extent
    stop = base
    for (w, (_, b)) in zip(targets, parts)
        isempty(b) && continue
        base = min(base, first(b))
        stop = max(stop, maximum(o -> window_stop(w[s.axis], o), b))
    end
    staging = base:(stop - 1)
    owned = [_replace_axis(w, s.axis, _axis_window(w[s.axis], s.global_extent, vcat(p[1], p[2]), 0))
             for (w, p) in zip(targets, parts)]
    interior = [_replace_axis(w, s.axis, _axis_window(w[s.axis], s.extent, p[1], s.offset))
                for (w, p) in zip(targets, parts)]
    boundary = [_replace_axis(w, s.axis, _axis_window(w[s.axis], length(staging), p[2], base))
                for (w, p) in zip(targets, parts)]
    return (; owned, interior, boundary, staging, split = [length(p[1]) for p in parts])
end

"The cells of a slab, as an index range into the global axis (0-based, exclusive at the top)."
slab_range(s::Slab) = s.offset:(s.offset + s.extent - 1)

"A view of `a` holding only the cells `r` (1-based) along `axis`."
axis_slice(a::AbstractArray{T,N}, axis::Integer, r) where {T,N} =
    view(a, ntuple(d -> d == axis ? r : Base.Colon(), Val(N))...)

"An array like `a` with `axis` shortened to `len` cells."
axis_similar(a::AbstractArray{T,N}, axis::Integer, len::Integer) where {T,N} =
    similar(a, ntuple(d -> d == axis ? Int(len) : size(a, d), Val(N)))
