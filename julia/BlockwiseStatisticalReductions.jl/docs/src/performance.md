# Performance

## What it costs

Measured on one core of a Xeon Gold 5418Y, 4096×4096 `Float64`, as a ratio to a streaming `sum(x)` in the
same process:

| Request | Ratio to one `sum` |
|---|---|
| mean, 8×8 tiles | 0.82 |
| mean + variance, 8×8 tiles | 1.25 |
| covariance of two fields, 8×8 tiles | 1.18 |
| mean, 2×2 tiles | 1.17 |
| **six scales 2…64, mean + variance** | **2.03** |

That last row is the point: six resolutions for about twice the cost of reading the array once, because
each level is merged from the level below rather than read from the input. Computing them independently
would be about 7.4.

For requests too large to compare against a single pass, the plan is measured against the same request
with no sharing at all — every target from its own base pass:

| Request | Planned cost / all-base cost |
|---|---|
| six scales 2…64, mean + variance | 0.28 |
| every tile size 2…64 | 0.31 |
| prime tiles {3,5,7,11,13} × {1,2,4,8} | 0.27 |
| fifteen dense sizes 2…16 | 0.16 |
| 3-D 512×512×128, fifteen anisotropic tiles | 0.25 |

And the executor against its own roofline model — measured time over modelled time — lands between 0.38
and 1.21, so the kernels hit the model or beat it.

Threaded, at 8 threads, against a *threaded* streaming sum: base pass mean 1.0–1.2, mean + variance
1.9–2.1, covariance 1.4–1.6.

![Cost against the number of scales](assets/cost.png)

## Running the gates

```bash
julia --project=benchmark benchmark/gates.jl --only=kernels,kernels-f32,planner,executor
julia -t8 --project=benchmark benchmark/gates.jl --only=threaded
julia --project=benchmark benchmark/gates.jl --list
```

Every gate is a ratio against a reference measured in the same process, so the thresholds hold across
machines. Run them on an otherwise idle machine, and run each group in the process it describes: the
serial groups measure 20–70 % higher inside a multithreaded process, whose allocator and garbage
collector they are not written against.

The gates are **not** run in CI. They are throughput ratios, and a shared runner cannot reproduce them —
even on a quiet 96-core machine the threaded base-pass ratio moved by 2× between processes until its terms
were made long enough to measure. What CI does check is the behaviour: the test suite asserts that every
kernel and every prepared request allocates nothing, on every backend.

## Why the ratios are what they are

A base pass reads the input once and writes one accumulator per output cell. At 8×8 tiles the writes are
1/64 of the reads, so the ratio to a pure streaming read is near one; at 2×2 tiles they are 1/4 of the
reads and the ratio rises. A coarsen level reads its parent and writes 1/kᴺ as much, so a tower's levels
cost a geometric series on top of the base pass — which is why six scales cost 2.03 rather than 6.

The remaining opportunity is fusion: computing several tiled levels within one cache tile so the
intermediates never reach DRAM. Measured on the six-scale tower, that would cut traffic from 402 MB to
267 MB. It is not implemented.
