```@meta
CurrentModule = BlockwiseStatisticalReductions
```

# BlockwiseStatisticalReductions.jl

A **purpose-agnostic** engine for computing statistics over N-dimensional data at *many coarser
scales at once* — efficiently, by reusing intermediate results so the data is touched as few times
as possible.

Give it an array and a set of scales; get back means / variances / covariances / extrema / moments
(or your own statistic) at every scale, computed in roughly a single pass over the data. Typical
uses: multi-scale features for machine-learning from high-resolution fields, image/volume pyramids,
downsampling of simulation or observational data, and any scale-dependent statistics.

![Multi-scale statistics in one pass](assets/multiscale_showcase.png)

```julia
using BlockwiseStatisticalReductions

data = randn(2048, 2048)
r = reduce_stats(data, [2, 4, 8, 16, 32, 64]; stats = (Mean(), Var()))
r[(8, 8)].mean      # 256×256 block means at 8× coarsening
r[(8, 8)].var       # … and variances
```

## Why it is fast

Every statistic is a **mergeable monoid** over sufficient statistics: a coarse block is *exactly*
the `merge` of its finer child blocks. The scales therefore form a tree — a DAG over the divisor
lattice of block sizes — and each scale is built by coarsening the nearest finer scale already
computed, instead of re-reading the data once per scale. A full multi-scale stack costs less than
two passes over a 1-D chain and far less than the naive (#scales)×(one pass each).

Results are numerically exact (Welford + Chan/Pebay merges), type-stable, and allocation-free at
steady state — including for variance and covariance.

## Highlights

- **One call, many scales:** [`reduce_stats`](@ref) returns a result keyed by output factor.
- **Composable statistics:** `Count`, `Sum`, `Mean`, `Var`, `Std`, `Cov`, `Min`, `Max`, `Moments`,
  computed together in one pass; or define your own (an isbits accumulator + a few methods).
- **Blockwise and sliding:** non-overlapping towers and overlapping windows (`Sliding`) share the
  same algebra; `stride == window` reproduces blockwise.
- **Backends:** `SerialBackend`, `ThreadedBackend` (OhMyThreads), `DistributedBackend` (Distributed),
  `AutoBackend`; a GPU backend is planned. All give identical results.

See [Getting Started](@ref) for a tour and [Overview & Theory](@ref) for the model.
