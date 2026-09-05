# Weights and gaps

## Weights

`weights` weights every statistic of a request:

```julia
blockstats(x, [8]; stats = (Mean(), Var()), weights = w)                  # one weight per element
blockstats(x, [8]; stats = (Mean(), Var()), weights = (nothing, dz))      # per-axis factors
blockstats(x, [8]; stats = (Mean(),), weights = (z = dz,), dimnames = (:x, :z))
```

Per-axis factors are multiplied per element, so the N-dimensional weight array is never materialized —
a cell-thickness vector along one axis costs a vector, not a copy of the input.

Weighting is a property of the *request*, not of a tag, so nothing can end up half-weighted. A statistic
with no weighted form throws when the request is prepared:

| Weighted | Not weighted |
|---|---|
| `Mean`, `Sum`, `Var`, `Std`, `Cov`, `Corr`, `ProductMean` | `Min`, `Max`, `Extrema`, `Moments`, `CentralMoments`, `Skewness`, `Kurtosis` |

`Count()` passes through unchanged: how many observations a window holds does not depend on their weights.
For the effective sample size, ask for the total weight itself with `Component(Mean(), :W)`.

The arithmetic is West's weighted lift with the Chan/Pébay merge run on weight totals instead of counts.
Unit weights reproduce an unweighted request bit for bit.

### Corrections

`Var`, `Std` and `Cov` take `corrected`:

| Value | Denominator |
|---|---|
| `false` | `W` (population) |
| `true`, `:frequency` | `W - 1` — weights are counts of repeated observations |
| `:reliability` | `W - W₂/W` — weights are relative precisions |

Unit weights make the two corrected forms identical (`W - W₂/W = n - 1`), so the unweighted accumulators
accept the symbols too. The frequency denominator is clamped at zero, so a total weight below one gives
`NaN` rather than a negative variance.

## Gaps

`skipnan = true` drops non-finite observations:

```julia
blockstats(x, [8]; stats = (Count(), Mean(), Var()), skipnan = true)
```

The skip is **per statistic**, not per element. A composite lifts each member from its own bound fields,
so a gap in `u` removes that element from `Mean(:u)` and from `Cov(:u, :w)`, but leaves `Mean(:w)`
untouched. `Count()` then reports the number of finite observations, and a window with none reports `NaN`
for its mean and variance.

Counts vary per cell once observations can be dropped, so the workspace stops storing a window's count
once for the whole node and keeps one per cell. That costs memory; it is the price of the feature and is
only paid when you ask for it.

Weights and `skipnan` compose: an element is dropped if any of the statistic's own bound fields — the
weight included — is not finite.
