module BlockwiseStatisticalReductionsDimensionalDataExt

# Labelled input and output. A dimensional array already carries what the engine otherwise has to be
# told: what the axes are called and where their cells sit. This reads both off the dims, so a scale
# specification can address an axis by name and a size can be given as a physical `Length`, and puts the
# geometry back on the results — every output axis becomes an interval lookup whose bounds are the cells
# each window covered, which is what makes a coarsened axis honest about the span it represents.

using DimensionalData: DimensionalData as DD
using BlockwiseStatisticalReductions: BlockwiseStatisticalReductions as BSR

_names(A) = map(DD.dim2key, DD.dims(A))

# The spacing of an axis, as cell edges. A lookup that already describes intervals states its own
# bounds; points are treated as cell centres, with edges at the midpoints between them.
function _spacing(dim)
    lk = DD.lookup(dim)
    lk isa DD.NoLookup && return nothing
    return _spacing(lk, DD.span(lk), DD.sampling(lk))
end
_spacing(lk, span::DD.Regular, ::DD.Points) = BSR.Regular(abs(Base.step(span)); first = first(DD.val(lk)) - abs(Base.step(span)) / 2)
_spacing(lk, span::DD.Regular, ::DD.Intervals) = BSR.Regular(abs(Base.step(span)); first = DD.bounds(lk)[1])
function _spacing(lk, span::DD.Explicit, ::DD.Intervals)
    m = DD.val(span)
    return BSR.Edges(vcat(view(m, 1, :), m[2, end]))
end
_spacing(lk, span, sampling) = BSR.spacing_from_points(DD.val(lk))

_spacings(A) = map(_spacing, DD.dims(A))

# One output axis: the windows that were reduced, as an interval lookup carrying their true bounds.
function _output_dim(dim, aw::BSR.AxisWindow, sp)
    sp === nothing && return DD.rebuild(dim, DD.NoLookup(Base.OneTo(BSR.nwindows(aw))))
    bounds = BSR.cell_bounds(sp, aw)
    isempty(bounds) && return DD.rebuild(dim, DD.NoLookup(Base.OneTo(0)))
    centers = [(lo + hi) / 2 for (lo, hi) in bounds]
    m = permutedims(hcat(first.(bounds), last.(bounds)))
    lk = DD.Sampled(centers; order = DD.ForwardOrdered(), span = DD.Explicit(m), sampling = DD.Intervals(DD.Center()))
    return DD.rebuild(dim, lk)
end

_output_dims(dims, w::BSR.Window, sps) = ntuple(d -> _output_dim(dims[d], w[d], sps[d]), length(w))

"Results of one window as a `DimStack`, one layer per statistic, on the coarsened axes."
function _as_stack(r::BSR.ScaleResults, w, dims, sps)
    od = _output_dims(dims, w, sps)
    nt = r[w]
    return DD.DimStack(NamedTuple{keys(nt)}(map(v -> DD.DimArray(v, od), values(nt))))
end

"""
    blockstats(A::AbstractDimArray, scales; stats, kwargs...) -> ScaleResults
    blockstats(A::AbstractDimStack, scales; stats, kwargs...)

Statistics of a dimensional array (or every layer of a stack, as named fields) over the windows
`scales` asks for. Axis names and coordinates come from the dims, so a scale specification may name its
axes and bound their sizes with a [`Length`](@ref). The result is the usual [`ScaleResults`](@ref) — so
it is indexed and inspected the same way — except each window holds a `DimStack` whose axes are interval
lookups over the cells that window covered.
"""
function BSR.blockstats(A::DD.AbstractDimArray, scales; kwargs...)
    p = BSR.prepare(A, scales; kwargs...)
    return BSR.blockstats!(p, A)
end
function BSR.blockstats(A::DD.AbstractDimStack, scales; kwargs...)
    p = BSR.prepare(A, scales; kwargs...)
    return BSR.blockstats!(p, A)
end

# A dimensional request is a plain one over the parent arrays, plus the names and spacings the dims
# carry; the handle keeps the dims so every result can be rebuilt on the coarsened axes.
struct DimPrepared{P,D}
    inner::P
    dims::D
end
BSR.explain(io::IO, p::DimPrepared; kw...) = BSR.explain(io, p.inner; kw...)

function BSR.prepare(A::DD.AbstractDimArray, scales; dimnames = nothing, spacing = nothing, kwargs...)
    dims = DD.dims(A)
    inner = BSR.prepare(parent(A), scales;
                        dimnames = dimnames === nothing ? _names(A) : dimnames,
                        spacing = spacing === nothing ? _spacings(A) : spacing, kwargs...)
    return DimPrepared(inner, dims)
end
function BSR.prepare(A::DD.AbstractDimStack, scales; dimnames = nothing, spacing = nothing, kwargs...)
    dims = DD.dims(A)
    fields = NamedTuple{keys(A)}(map(parent, values(A)))
    inner = BSR.prepare(fields, scales;
                        dimnames = dimnames === nothing ? map(DD.dim2key, dims) : dimnames,
                        spacing = spacing === nothing ? map(_spacing, dims) : spacing, kwargs...)
    return DimPrepared(inner, dims)
end

_parents(A::DD.AbstractDimArray) = parent(A)
_parents(A::DD.AbstractDimStack) = NamedTuple{keys(A)}(map(parent, values(A)))

"""
    blockstats!(p::DimPrepared, A) -> ScaleResults

Run a prepared dimensional request on new data of the same shape. The `DimStack`s are rebuilt each call;
their arrays alias the handle's, exactly as [`blockstats!`](@ref) on a plain request.
"""
function BSR.blockstats!(p::DimPrepared, A)
    r = BSR.blockstats!(p.inner, _parents(A))
    sps = BSR._axis_spacings(BSR.spacing(r), Val(length(p.dims)))
    return BSR.rebuild(r, [_as_stack(r, w, p.dims, sps) for w in BSR.windows(r)])
end

end
