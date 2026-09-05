# Labelled axes

## Names and coordinates, without a dependency

Two keywords carry everything the core needs:

```julia
blockstats(x, spec; stats = (Mean(),),
           dimnames = (:x, :y, :z),
           spacing = (Regular(0.5), Regular(0.5), Edges(z_edges)))
```

- `dimnames` lets a [scale specification](scales.md) address axes by name.
- `spacing` gives each axis its coordinates, so sizes can be given in physical units and results can
  report where each output cell sits.

An axis spacing is `Regular(step; first)` for evenly spaced cells or `Edges(v)` for `extent + 1` explicit
edges. `spacing_from_points(centers)` builds one from cell centres: evenly spaced centres give a
`Regular`, otherwise the edges are the midpoints between neighbours with the outermost two reflected.

## Sizes in physical units

```julia
ScaleSet((Dyadic(; max = Length(2000.0)), Dyadic(; max = Length(2000.0)), Fixed(1)))
```

`Length(x)` is resolved against the axis' spacing: a minimum becomes the smallest window whose mean
physical span is at least `x`, a maximum the largest whose span is at most `x`.

## Where an output cell is

```julia
g = geometry(r, (8, 8))
g.bounds[1]     # (low, high) coordinate of every output cell along axis 1
g.centers[1]
```

Under a non-uniform `Edges` spacing the physical span of an output cell is the sum of the native spans it
covers, which is what you want and what a naive `size × Δ` would get wrong.

## DimensionalData

With DimensionalData loaded, `blockstats` on a `DimArray` or `DimStack` takes the names and coordinates
from the array and gives them back:

```julia
using DimensionalData
r = blockstats(A, [8]; stats = (Mean(), Var()))
r[(8, 8)]        # a DimStack whose lookups carry the true bounds of each output cell
```

Output axes are rebuilt as `Sampled` lookups with an `Explicit` span and `Intervals(Center())` sampling,
so each output cell carries the real extent of the window it came from — not a point.

The result is still a `ScaleResults`, so `scales`, `windows`, `geometry` and `explain` keep working.

## NetCDF and Zarr

Both carry labelled axes of their own, so both can be read without DimensionalData.

```julia
using NCDatasets
ds = NCDataset("f.nc")

blockstats(ds["temperature"], [8]; stats = (Mean(),))            # one variable
blockstats(ds, (:u, :w), [8]; stats = (Cov(:u, :w),))            # several, as named fields
```

```julia
using Zarr
g = zopen("store.zarr")

blockstats(g["temperature"], [8]; stats = (Mean(),), group = g)  # `group` supplies the coordinates
blockstats(g, (:u, :w), [8]; stats = (Cov(:u, :w),))
```

Axis names come from the variable's own dimensions — the file's dimension names for NetCDF, the
`_ARRAY_DIMENSIONS` attribute for Zarr — and coordinates from a coordinate variable of the same name where
one exists. A Zarr array opened on its own has no group to look in, so pass `group` for coordinates; an
array with no dimension names at all needs an explicit `dimnames`.

The variable is materialized before reducing: these are chunked, possibly remote stores, and the kernels
want an array. `prepare` once and `blockstats!` per variable to reuse the plan across a directory of
files.
