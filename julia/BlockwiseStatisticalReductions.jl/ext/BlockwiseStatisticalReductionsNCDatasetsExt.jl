module BlockwiseStatisticalReductionsNCDatasetsExt

# NetCDF input. A variable already names its axes, and a dataset conventionally stores an axis'
# coordinates in a variable of the same name, so a request over a NetCDF variable can address axes by
# name and bound their sizes in physical units without any other labelled-array package. Data is read
# into memory before it is reduced: the kernels index their input directly, which is not something to do
# through a file handle.

using NCDatasets: NCDatasets as NC, CommonDataModel as CDM
using BlockwiseStatisticalReductions: BlockwiseStatisticalReductions as BSR

_names(v) = map(Symbol, CDM.dimnames(v))

# An axis' spacing, from the coordinate variable of the same name if the dataset has one.
function _spacing(ds, name::AbstractString)
    haskey(ds, name) || return nothing
    coord = ds[name]
    ndims(coord) == 1 || return nothing
    values = coord[:]
    (isempty(values) || !(eltype(values) <: Real)) && return nothing
    any(ismissing, values) && return nothing
    return BSR.spacing_from_points(values)
end
_spacings(ds, v) = map(n -> _spacing(ds, n), CDM.dimnames(v))

_dataset(v::CDM.AbstractVariable) = CDM.dataset(v)

"""
Read a variable into memory, applying the CF conventions and dropping the mask.

`Array` is what the NCDatasets documentation gives for loading a variable; `v[:]` would flatten it. A
variable whose fill value actually occurs comes back holding `missing`, which no statistic is defined
over, so `nomissing` refuses it rather than letting it propagate.
"""
_materialize(v::CDM.AbstractVariable) = NC.nomissing(Array(v))

_check_same_dims(vs::Tuple) =
    all(v -> CDM.dimnames(v) == CDM.dimnames(vs[1]), vs) ||
        throw(DimensionMismatch("variables span different dimensions: $(map(CDM.dimnames, vs))"))

"""
    blockstats(v::NCDatasets.CFVariable, scales; stats, kwargs...) -> ScaleResults
    blockstats(ds::NCDatasets.NCDataset, names, scales; stats, kwargs...)

Statistics of a NetCDF variable, or of several variables of one dataset as named fields, over the
windows `scales` asks for. Axis names come from the variable's dimensions and coordinates from the
like-named variables, so a scale specification may address an axis by name and bound its sizes with a
[`Length`](@ref). The variables are read into memory first.
"""
BSR.blockstats(v::CDM.AbstractVariable, scales; kwargs...) = BSR.blockstats!(BSR.prepare(v, scales; kwargs...), v)
BSR.blockstats(ds::NC.NCDataset, names, scales; kwargs...) =
    BSR.blockstats!(BSR.prepare(ds, names, scales; kwargs...), ds)

function BSR.prepare(v::CDM.AbstractVariable, scales; dimnames = nothing, spacing = nothing, kwargs...)
    ds = _dataset(v)
    return BSR.prepare(_materialize(v), scales;
                       dimnames = dimnames === nothing ? _names(v) : dimnames,
                       spacing = spacing === nothing ? _spacings(ds, v) : spacing, kwargs...)
end

function BSR.prepare(ds::NC.NCDataset, names, scales; dimnames = nothing, spacing = nothing, kwargs...)
    keys = Tuple(Symbol(n) for n in names)
    vs = Tuple(ds[String(n)] for n in keys)
    _check_same_dims(vs)
    fields = NamedTuple{keys}(map(_materialize, vs))
    return BSR.prepare(fields, scales;
                       dimnames = dimnames === nothing ? _names(vs[1]) : dimnames,
                       spacing = spacing === nothing ? _spacings(ds, vs[1]) : spacing, kwargs...)
end

BSR.blockstats!(p::BSR.Prepared, v::CDM.AbstractVariable) = BSR.blockstats!(p, _materialize(v))
BSR.blockstats!(p::BSR.Prepared, ds::NC.NCDataset) =
    BSR.blockstats!(p, NamedTuple{p.fieldnames}(map(n -> _materialize(ds[String(n)]), p.fieldnames)))

end
