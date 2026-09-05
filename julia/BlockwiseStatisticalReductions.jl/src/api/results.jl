"""
    ScaleResults{N}

Results of one request: for every requested window, a `NamedTuple` of arrays keyed by statistic name.
Index by window, by size tuple when that is unambiguous, or by position; iterate [`windows`](@ref) /
[`scales`](@ref).

```julia
r[(8, 8)]              # NamedTuple of arrays at tile size 8×8
r[(8, 8)].mean         # one array
scales(r)              # the requested sizes, coarsest last
geometry(r, (8, 8))    # window, per-axis origins and covered ranges
```
"""
struct ScaleResults{N,E,W<:AbstractVector,R<:AbstractVector,DN,SP,K}
    input_shape::NTuple{N,Int}
    windows::W
    results::R
    plan::Plan{N}
    dimnames::DN
    spacing::SP
    statnames::NTuple{K,Symbol}
end
ScaleResults(input_shape::NTuple{N,Int}, windows, results, plan, statnames::NTuple{K,Symbol},
             dimnames = nothing, spacing = nothing) where {N,K} =
    ScaleResults{N,eltype(results),typeof(windows),typeof(results),typeof(dimnames),typeof(spacing),K}(
        input_shape, windows, results, plan, dimnames, spacing, statnames)

"Names of the statistics in each result."
statnames(r::ScaleResults) = r.statnames

"""
    rebuild(r::ScaleResults, results) -> ScaleResults

The same request with each window's statistics replaced by `results[i]`, keeping the windows, plan,
names and spacing. Extensions use this to present results in their own container.
"""
rebuild(r::ScaleResults, results::AbstractVector) =
    ScaleResults(r.input_shape, r.windows, results, r.plan, r.statnames, r.dimnames, r.spacing)

"Axis names of the input, or `nothing` when it had none."
dimnames(r::ScaleResults) = r.dimnames
"Per-axis coordinate spacing of the input, or `nothing` where an axis had none."
spacing(r::ScaleResults) = r.spacing

"Requested windows, in the order the request produced them."
windows(r::ScaleResults) = r.windows
"Requested window sizes, aligned with [`windows`](@ref)."
scales(r::ScaleResults) = map(w -> map(aw -> aw.size, w), r.windows)
"Shapes of the result arrays, aligned with [`windows`](@ref)."
shapes(r::ScaleResults) = map(shape, r.windows)
Base.keys(r::ScaleResults) = scales(r)
Base.length(r::ScaleResults) = length(r.windows)
Base.eltype(::Type{<:ScaleResults{N,E,W}}) where {N,E,W} = Pair{eltype(W),E}
Base.iterate(r::ScaleResults, i = 1) = i > length(r) ? nothing : (r.windows[i] => r.results[i], i + 1)

Base.getindex(r::ScaleResults, i::Integer) = r.results[i]
function Base.getindex(r::ScaleResults{N}, w::Tuple{AxisWindow,Vararg{AxisWindow}}) where {N}
    i = findfirst(==(w), r.windows)
    i === nothing && throw(KeyError(w))
    return r.results[i]
end
function Base.getindex(r::ScaleResults{N}, sizes::Tuple{Integer,Vararg{Integer}}) where {N}
    length(sizes) == N || throw(DimensionMismatch("$(length(sizes)) sizes for $N axes"))
    hits = findall(w -> map(aw -> aw.size, w) == map(Int, sizes), r.windows)
    isempty(hits) && throw(KeyError(sizes))
    length(hits) == 1 || throw(ArgumentError("size $sizes is ambiguous: $(length(hits)) requested windows share it; index by window instead"))
    return r.results[hits[1]]
end
Base.getindex(r::ScaleResults{1}, size::Integer) = r[(size,)]
Base.haskey(r::ScaleResults{N}, sizes::Tuple{Integer,Vararg{Integer}}) where {N} =
    length(sizes) == N && count(w -> map(aw -> aw.size, w) == map(Int, sizes), r.windows) == 1
Base.haskey(r::ScaleResults, w::Tuple{AxisWindow,Vararg{AxisWindow}}) = any(==(w), r.windows)

"""
    geometry(r::ScaleResults, key) -> NamedTuple

Where the windows of one result sit: the `window`, the axis `names`, the per-axis `origins` (0-based),
the input index `ranges` each output cell covers, and — when the axis had a coordinate spacing — the
physical `bounds` and `centers` of every output cell.
"""
function geometry(r::ScaleResults{N}, key) where {N}
    w = _window_for(r, key)
    sp = _axis_spacings(r.spacing, Val(N))
    return (window = w,
            names = r.dimnames,
            origins = map(origins, w),
            ranges = ntuple(d -> [window_range(w[d], i) for i in 1:nwindows(w[d])], Val(N)),
            bounds = ntuple(d -> sp[d] === nothing ? nothing : cell_bounds(sp[d], w[d]), Val(N)),
            centers = ntuple(d -> sp[d] === nothing ? nothing : cell_centers(sp[d], w[d]), Val(N)))
end
_window_for(r::ScaleResults, w::Tuple{AxisWindow,Vararg{AxisWindow}}) = (haskey(r, w) || throw(KeyError(w)); w)
_window_for(r::ScaleResults, i::Integer) = r.windows[i]
function _window_for(r::ScaleResults{N}, sizes::Tuple{Integer,Vararg{Integer}}) where {N}
    hits = findall(w -> map(aw -> aw.size, w) == map(Int, sizes), r.windows)
    isempty(hits) && throw(KeyError(sizes))
    length(hits) == 1 || throw(ArgumentError("size $sizes is ambiguous: $(length(hits)) requested windows share it; index by window instead"))
    return r.windows[hits[1]]
end
_window_for(r::ScaleResults{1}, size::Integer) = _window_for(r, (size,))
