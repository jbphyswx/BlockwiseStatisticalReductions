# How-To: Zero-Allocation Execution

## The problem

In production pipelines processing thousands of time steps, per-frame GC
allocations accumulate and cause unpredictable latency spikes.

## Solution: pre-allocate once, execute many

```julia
using BlockwiseStatisticalReductions

# Build plan once
plan = build_optimal_multires_plan((128, 128, 8), [2, 4, 8], [:mean])

# Pre-allocate all output buffers for this plan + data shape
data = randn(Float32, 128, 128, 8)
bufs = allocate_buffers(plan, data)

# Warmup (first call compiles everything)
execute!(plan, bufs, data)

# Subsequent calls: zero allocation
for frame in frames
    outputs = execute!(plan, bufs, frame)
    # outputs is an NTuple of views into bufs — no copy, no alloc
    process(outputs[1], outputs[2], outputs[3])
end
```

## How it works

`allocate_buffers(plan, data)` walks the execution sequence and pre-allocates
one `Array{T,N}` per step, sized according to the window configuration chain:

```julia
struct ExecutionBuffers{T,N}
    buffers::Vector{Array{T,N}}
end
```

`execute!(plan, bufs, data)` writes results directly into these buffers using
in-place kernels (`blockwise_mean!`).  The return value is a tuple of views
into the buffer arrays at output positions — zero-copy.

## Caveats

- **Don't mutate** the returned views between calls — they're aliased to the
  buffer arrays which will be overwritten on the next `execute!`.
- **Same shape required** — each call must pass data with the same shape used
  when `allocate_buffers` was called.
- **Plan reuse** — the same plan + buffers can process any number of frames.

## Measuring allocations

```julia
using Test

alloc = @allocated execute!(plan, bufs, data)
@test alloc < 10_000  # Less than 10KB (small tuple/view overhead)
```

The residual allocations (typically < 10KB) come from the tuple construction
and output index gathering — not from any array allocation.

## When to use

| Scenario | Recommended API |
|----------|----------------|
| One-shot analysis | `execute(plan, data)` or `multiresolution_stats(...)` |
| Processing a time series | `allocate_buffers` + `execute!` loop |
| Real-time pipeline | `allocate_buffers` + `execute!` with pinned buffers |
| Interactive exploration | `blockwise_mean(data, ...)` convenience functions |
