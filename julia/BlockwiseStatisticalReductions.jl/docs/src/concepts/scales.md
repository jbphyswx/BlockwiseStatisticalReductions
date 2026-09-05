# Scale specifications

A request names a set of windows. The convenient forms come first; everything resolves to the same list.

```julia
blockstats(x, 8)                    # one isotropic tile size
blockstats(x, [2, 4, 8])            # several isotropic sizes
blockstats(x, [(4, 4, 1)])          # one anisotropic size (a 3-D input, last axis unreduced)
blockstats(x, [(4, 4, 1), (8, 8, 2)])
```

For anything more, build a `ScaleSet` from a per-axis size generator, a placement, and a rule for
combining the axes.

## Size generators

| Generator | Sizes |
|---|---|
| `Sizes(v)` | exactly `v` |
| `Dyadic(; min, max)` | powers of two in range |
| `Smooth(primes = (2, 3); min, max)` | products of those primes in range |
| `Every(; min, max)` | every integer in range |
| `Divisors(; min, max)` | sizes that divide the extent |
| `Fixed(k)` | just `k` — use `Fixed(1)` to leave an axis unreduced |
| `Subsample(gen, budget)` | at most `budget` evenly spaced members of `gen`, endpoints kept |

`min` and `max` take an integer, or a [`Length`](labeled.md) in the axis' own physical units.

## Placement

| Placement | Origins |
|---|---|
| `Tiled()` | every `size` cells (default) |
| `Stride(k)` | every `k` cells |
| `Overlap(f)` | every `size · (1 - f)` cells |
| `Dense()` | every cell |
| `Anchors(v)` | exactly `v` |
| `Spread(k)` | `k` windows, the first and last flush with the ends |

## Combining axes

```julia
ScaleSet((Dyadic(max = 64), Dyadic(max = 64), Fixed(1)); combine = Product())
```

- `Isotropic()` — all axes take the same size. `Isotropic((:x, :y))` couples only those two, leaving the
  rest to vary independently.
- `Product()` — the Cartesian product of the per-axis sets.
- `Zip()` — the axes advance together through their sets.

A `ScaleSet` also takes `edge`, a `filter` predicate on the window tuple, `include_full` to add the
window covering the whole extent, and `min_elements` / `max_elements` to bound the cells per window.

## Named axes

Give the axes names and address them by name:

```julia
blockstats(x, ScaleSet((h = Dyadic(max = 64), v = Fixed(1))); dimnames = (:h, :v), stats = (Mean(),))
```

The [DimensionalData extension](labeled.md) takes the names from the array instead.
