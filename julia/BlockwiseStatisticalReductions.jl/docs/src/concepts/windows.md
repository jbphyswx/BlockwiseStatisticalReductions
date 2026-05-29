# Window Configurations

A `WindowConfig{D}` defines how the input array is partitioned into tiles for
reduction.  It is the fundamental building block of every reduction operation.

## Fields

```julia
struct WindowConfig{D}
    sizes::NTuple{D,Int}     # Block size in each dimension
    strides::NTuple{D,Int}   # Step between consecutive blocks
    padding::Symbol          # :valid, :same, or :full
end
```

## Block size and strides

- **sizes** — the extent of each tile.  `(4, 4, 1)` means 4 elements in dim 1,
  4 in dim 2, 1 in dim 3.
- **strides** — how far to advance between tiles.  When `strides == sizes`, the
  tiles are non-overlapping (blockwise).  When `strides < sizes`, tiles overlap
  (rolling/sliding window).

## Constructors

```julia
# Explicit: sizes, strides, padding
window = WindowConfig((4, 4, 1), (4, 4, 1), :valid)

# Convenience: strides default to sizes (non-overlapping)
window = WindowConfig((4, 4, 1))

# Rolling window: stride 1 in all dims
rolling = WindowConfig((5, 5, 5), (1, 1, 1), :same)

# Strided: blocks of 8 with stride 4 (50% overlap)
strided = WindowConfig((8, 8, 4), (4, 4, 2), :valid)
```

## Padding modes

- **`:valid`** — Only tiles that fit entirely within the input are computed.
  Output size: `div(input_size, stride)` (floor division).  Boundary elements
  that don't fill a complete tile are dropped.

- **`:same`** — Output has the same spatial size as input (zero-padded at
  boundaries).  Only meaningful for rolling windows.

- **`:full`** — All positions where the window overlaps at least one element are
  computed (padded).  Output is larger than input.

For blockwise (non-overlapping) reductions, `:valid` is almost always what you
want.

## Output shape calculation

For `:valid` mode with `strides == sizes` (standard blockwise):

```
output_size[d] = div(input_size[d], sizes[d])
```

For `:valid` mode with general strides:

```
output_size[d] = div(input_size[d] - sizes[d], strides[d]) + 1
```

## Validation

```julia
validate_window_config(input_shape, window; strict=true)
```

When `strict=true`, throws an error if the block size doesn't evenly divide
the input.  When `strict=false`, allows remainder elements to be dropped.

## Common patterns

| Use case | sizes | strides | padding |
|----------|-------|---------|---------|
| 4× coarsening in x,y | `(4,4,1)` | `(4,4,1)` | `:valid` |
| Full z reduction | `(1,1,nz)` | `(1,1,nz)` | `:valid` |
| 3×3 sliding average | `(3,3,1)` | `(1,1,1)` | `:same` |
| 50% overlapping | `(8,8,4)` | `(4,4,2)` | `:valid` |
