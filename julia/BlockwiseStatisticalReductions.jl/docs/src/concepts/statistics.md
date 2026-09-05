# Statistics

## Accumulators

Every statistic is computed from an **accumulator**: an immutable, `isbits` struct holding sufficient
statistics for a set of observations, together with a way to combine two of them.

```julia
neutral(A)          # the identity: merge(neutral(A), a) == a
lift(A, xs)         # the accumulator of one observation
merge(a, b)         # associative and commutative
```

A variance accumulator is `VarAcc(n, mean, M2)`. Merging two of them is the Chan–Golub–LeVeque update —
no observation is revisited. That single property is what makes everything else possible: a tile of 64
cells is the merge of two tiles of 32, and the planner never has to touch the input twice.

Some accumulators also have a group inverse (`unmerge`), reported by `is_invertible`. It exists for
callers who want it; the planner never uses it, because subtracting accumulators is the
cancellation-prone form the [numerics](numerics.md) policy rules out.

## The statistics that ship

| Tag | Reads | Reports |
|---|---|---|
| `Count()` | one field | number of observations |
| `Sum()` | one field | Σx |
| `Mean()` | one field | x̄ |
| `Var(; corrected)` / `Std(; corrected)` | one field | variance / standard deviation |
| `Min()` / `Max()` / `Extrema()` | one field | extremes |
| `Moments(K)` | one field | ⟨x⟩ … ⟨xᴷ⟩ |
| `CentralMoments(K)` (K ≤ 4) | one field | central moments 2…K |
| `Skewness()` / `Kurtosis(; excess)` | one field | shape |
| `Cov(i, j; corrected)` | two fields | covariance |
| `Corr(i, j)` | two fields | correlation |
| `ProductMean(i, j)` | two fields | ⟨xy⟩ |
| `Component(tag, field)` | as `tag` | one raw accumulator field |

`corrected` takes `true` (the default, dividing by `n - 1`), `false` (dividing by `n`), or — for weighted
requests — `:frequency` and `:reliability`. See [weights](weights.md).

`Component` is the escape hatch for the numerators themselves: `Component(Var(), :M2)` is `Σ(x - x̄)²` and
`Component(Cov(:u, :w), :C)` is `Σ(u - ū)(w - w̄)`, which is what you want when the results will be merged
again later by something outside this package.

## Binding fields

A statistic reads one or more input fields, given by position or by name:

```julia
r = blockstats((u = u, w = w), [8];
               stats = (var_u = Var(:u), var_w = Var(:w), flux = Cov(:u, :w)))
```

This is **one pass** over `u` and `w`, not three. `assemble` works out the distinct accumulators the
request needs, drops any that another subsumes (a `Mean(:u)` next to a `Var(:u)` costs nothing extra,
because `VarAcc` already carries the mean), and builds a single composite accumulator. The composite
implements the same interface member-wise, with generated code, so every member's field index is a
compile-time constant.

Name the results by passing a `NamedTuple`, as above; otherwise each tag names itself (`:mean`, `:var`,
`:cov_u_w`) and two tags that would collide are an error.

## Two passes at most, whatever you ask for

A second moment is most accurate when computed in two passes — the mean first, then the centred sum. The
accumulators expose that as a protocol (`phases`, `p1lift`, `mid`, `p2lift`, `finish`) rather than as a
special case, and a composite fuses the phases of all its members. So a request for mean, variance,
minimum, maximum and a covariance costs **at most two passes over each box**, never one pass per
statistic.

Boxes are small enough to stay in cache, so those two passes are two passes over registers, not over
memory.

## Writing your own

Implement `neutral`, `lift` and `Base.merge` for an `AbstractAccumulator`, plus `acc_eltype` and
`arity` if it reads more than one field; then a tag with `bindings`, `accumulator_type`, `name`,
`result_eltype` and `finalize`. `check_monoid(A; samples)` verifies the laws — identity, commutativity,
associativity, that grouping does not change the result, and that `combine` agrees with a fold.
