# Examples

Run any example with the `examples` project active:

```bash
julia --project=examples examples/basic_usage.jl
julia -t4 --project=examples examples/backends.jl     # multithreaded
```

| File | Shows |
|------|-------|
| [`basic_usage.jl`](basic_usage.jl) | several statistics at several scales in one pass; anisotropic factors; inspecting the plan |
| [`sliding_windows.jl`](sliding_windows.jl) | overlapping windows (`Sliding`), `stride == window` == blockwise, sliding covariance |
| [`custom_statistic.jl`](custom_statistic.jl) | defining your own statistic (geometric mean) with no changes to the package |
| [`backends.jl`](backends.jl) | serial / threaded / distributed parity; zero-allocation reuse |
| [`multiscale_showcase.jl`](multiscale_showcase.jl) | generates the frontline figure: a power-law field, its block-mean pyramid, and variance vs scale |
