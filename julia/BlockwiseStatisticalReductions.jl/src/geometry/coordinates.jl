abstract type AxisSpacing end

"Cells of equal physical `step`; the low edge of the first cell is at `first`."
struct Regular{T<:Real} <: AxisSpacing
    step::T
    first::T
end
Regular(step::Real; first::Real = zero(step)) = (step > 0 || throw(ArgumentError("step must be > 0")); Regular(promote(step, first)...))

"Cell edges (`extent + 1` increasing values)."
struct Edges{V<:AbstractVector{<:Real}} <: AxisSpacing
    edges::V
    function Edges(edges::V) where {V<:AbstractVector{<:Real}}
        length(edges) >= 2 || throw(ArgumentError("need at least two edges"))
        issorted(edges; lt = <=) || throw(ArgumentError("edges must be strictly increasing"))
        return new{V}(edges)
    end
end

"A size given in physical units of an axis' coordinates."
struct Length{T<:Real}
    value::T
end

"Mean physical size of one cell."
mean_step(r::Regular, extent::Integer) = r.step
mean_step(e::Edges, extent::Integer) =
    (length(e.edges) == extent + 1 || throw(DimensionMismatch("$(length(e.edges)) edges for extent $extent")); (e.edges[end] - e.edges[1]) / extent)

"Smallest window (in cells) whose mean physical span is at least `L`."
cells_at_least(sp::AxisSpacing, extent::Integer, L::Length) = max(1, ceil(Int, L.value / mean_step(sp, extent)))
"Largest window (in cells) whose mean physical span is at most `L`."
cells_at_most(sp::AxisSpacing, extent::Integer, L::Length) = floor(Int, L.value / mean_step(sp, extent))

"Physical `(low, high)` bounds of every window."
function cell_bounds(sp::Regular, aw::AxisWindow)
    return [(sp.first + o * sp.step, sp.first + (o + window_length(aw, i)) * sp.step) for (i, o) in enumerate(origins(aw))]
end
function cell_bounds(sp::Edges, aw::AxisWindow)
    length(sp.edges) == aw.extent + 1 || throw(DimensionMismatch("$(length(sp.edges)) edges for extent $(aw.extent)"))
    return [(sp.edges[o+1], sp.edges[o+window_length(aw, i)+1]) for (i, o) in enumerate(origins(aw))]
end
"Physical centers of every window."
cell_centers(sp::AxisSpacing, aw::AxisWindow) = [(lo + hi) / 2 for (lo, hi) in cell_bounds(sp, aw)]
"Physical spans of every window."
cell_spans(sp::AxisSpacing, aw::AxisWindow) = [hi - lo for (lo, hi) in cell_bounds(sp, aw)]

"""
    spacing_from_points(centers::AbstractVector) -> AxisSpacing

The spacing of an axis whose coordinates are the cell centres `centers`. Evenly spaced centres give a
[`Regular`](@ref) spacing; otherwise the edges are the midpoints between neighbours, with the outermost
two reflected so the first and last cells keep their own width. One centre gives a unit-wide cell.
"""
function spacing_from_points(centers::AbstractVector)
    n = length(centers)
    n == 0 && throw(ArgumentError("an axis needs at least one coordinate"))
    v = float.(collect(centers))
    n == 1 && return Edges([v[1] - 0.5, v[1] + 0.5])
    steps = diff(v)
    all(>(0), steps) || throw(ArgumentError("axis coordinates must be strictly increasing"))
    if all(s -> isapprox(s, steps[1]; rtol = 1e-9), steps)
        return Regular(steps[1]; first = v[1] - steps[1] / 2)
    end
    mid = [(v[i] + v[i+1]) / 2 for i in 1:(n - 1)]
    return Edges(vcat(v[1] - (mid[1] - v[1]), mid, v[end] + (v[end] - mid[end])))
end
