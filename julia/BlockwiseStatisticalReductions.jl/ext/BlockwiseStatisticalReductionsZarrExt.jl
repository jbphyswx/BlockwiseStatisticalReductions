module BlockwiseStatisticalReductionsZarrExt

# Zarr input. A Zarr array names its axes in the `_ARRAY_DIMENSIONS` attribute that xarray writes, and a
# group conventionally holds an axis' coordinates in a like-named array, so a request over a Zarr array
# can address axes by name and bound their sizes in physical units. Data is read into memory before it
# is reduced: the kernels index their input directly, and a Zarr array is a store, possibly remote.

using Zarr: Zarr
using BlockwiseStatisticalReductions: BlockwiseStatisticalReductions as BSR

"The attribute xarray and the Zarr spec use to name an array's axes."
const DIMENSIONS_ATTRIBUTE = "_ARRAY_DIMENSIONS"

function _names(a::Zarr.ZArray)
    haskey(a.attrs, DIMENSIONS_ATTRIBUTE) ||
        throw(ArgumentError("the array has no `$DIMENSIONS_ATTRIBUTE` attribute naming its axes; pass `dimnames` instead"))
    names = a.attrs[DIMENSIONS_ATTRIBUTE]
    length(names) == ndims(a) ||
        throw(DimensionMismatch("`$DIMENSIONS_ATTRIBUTE` names $(length(names)) axes for a $(ndims(a))-dimensional array"))
    return Tuple(Symbol(n) for n in names)
end

# An axis' spacing, from the like-named array in the same group if there is one.
function _spacing(g::Zarr.ZGroup, name::Symbol)
    key = String(name)
    haskey(g.arrays, key) || return nothing
    coord = g.arrays[key]
    (ndims(coord) == 1 && eltype(coord) <: Real) || return nothing
    values = coord[:]
    isempty(values) && return nothing
    return BSR.spacing_from_points(values)
end
_spacing(::Nothing, ::Symbol) = nothing
_spacings(g, names::Tuple) = map(n -> _spacing(g, n), names)

"Read a Zarr array into memory."
_materialize(a::Zarr.ZArray) = a[axes(a)...]

_check_same_shape(as::Tuple) =
    all(a -> size(a) == size(as[1]), as) ||
        throw(DimensionMismatch("arrays have different shapes: $(map(size, as))"))

"""
    blockstats(a::Zarr.ZArray, scales; stats, group = nothing, kwargs...) -> ScaleResults
    blockstats(g::Zarr.ZGroup, names, scales; stats, kwargs...)

Statistics of a Zarr array, or of several arrays of one group as named fields, over the windows `scales`
asks for. Axis names come from the `_ARRAY_DIMENSIONS` attribute; coordinates come from like-named
arrays in `group` (the group itself, in the group form), so a scale specification may address an axis by
name and bound its sizes with a [`Length`](@ref). The arrays are read into memory first.
"""
BSR.blockstats(a::Zarr.ZArray, scales; kwargs...) = BSR.blockstats!(BSR.prepare(a, scales; kwargs...), a)
BSR.blockstats(g::Zarr.ZGroup, names, scales; kwargs...) =
    BSR.blockstats!(BSR.prepare(g, names, scales; kwargs...), g)

function BSR.prepare(a::Zarr.ZArray, scales; dimnames = nothing, spacing = nothing, group = nothing, kwargs...)
    names = dimnames === nothing ? _names(a) : dimnames
    return BSR.prepare(_materialize(a), scales;
                       dimnames = names,
                       spacing = spacing === nothing ? _spacings(group, names) : spacing, kwargs...)
end

function BSR.prepare(g::Zarr.ZGroup, names, scales; dimnames = nothing, spacing = nothing, kwargs...)
    keys = Tuple(Symbol(n) for n in names)
    as = Tuple(g.arrays[String(n)] for n in keys)
    _check_same_shape(as)
    axisnames = dimnames === nothing ? _names(as[1]) : dimnames
    fields = NamedTuple{keys}(map(_materialize, as))
    return BSR.prepare(fields, scales;
                       dimnames = axisnames,
                       spacing = spacing === nothing ? _spacings(g, axisnames) : spacing, kwargs...)
end

BSR.blockstats!(p::BSR.Prepared, a::Zarr.ZArray) = BSR.blockstats!(p, _materialize(a))
BSR.blockstats!(p::BSR.Prepared, g::Zarr.ZGroup) =
    BSR.blockstats!(p, NamedTuple{p.fieldnames}(map(n -> _materialize(g.arrays[String(n)]), p.fieldnames)))

end
