# How-To: Basic Blockwise Reductions

## Compute a blockwise mean

```julia
using BlockwiseStatisticalReductions

data = randn(Float32, 100, 100, 50)
means = blockwise_mean(data, (10, 10, 5))
# Result: 10×10×10 array
```

## Compute blockwise variance

```julia
# Bessel-corrected (unbiased) variance — the default
variances = blockwise_variance(data, (10, 10, 5); corrected=true)

# Population variance (biased)
variances = blockwise_variance(data, (10, 10, 5); corrected=false)

# Standard deviation
stds = blockwise_std(data, (10, 10, 5))
```

## Multiple statistics in one pass

When you need both mean and variance, the fused kernel reads data only once:

```julia
results = blockwise_stats(data, (10, 10, 5); stats=[:mean, :variance, :min, :max])
results[:mean]       # 10×10×10
results[:variance]   # 10×10×10
results[:min]        # 10×10×10
results[:max]        # 10×10×10
```

## Covariance between two fields

```julia
temperature = randn(Float32, 100, 100, 50)
humidity    = randn(Float32, 100, 100, 50)

covs = blockwise_covariance(temperature, humidity, (10, 10, 5))
# Result: 10×10×10 array of per-block covariances
```

Both arrays must have the same shape.

## Raw moments

```julia
# First 4 raw moments per block
moments = blockwise_moments(data, (10, 10, 5), 4)
# Result: 10×10×10 array of NTuple{4, Float32}

# Access individual moments
moments[1, 1, 1][1]  # first moment (mean)
moments[1, 1, 1][2]  # second moment (E[x²])
moments[1, 1, 1][3]  # third moment
moments[1, 1, 1][4]  # fourth moment
```

## Reduce only specific dimensions

Use `dims` to control which dimensions participate.

```julia
# Only reduce x and y (keep z intact)
# For blockwise_stats and the plan-based API:
plan = build_optimal_multires_plan((100, 100, 50), [10], [:mean]; dims=(1, 2))
results = execute(plan, data)
# Result: 10×10×50 (z preserved)
```

For `blockwise_mean` and friends, set the block size to 1 in dimensions you
don't want to reduce:

```julia
means = blockwise_mean(data, (10, 10, 1))
# Result: 10×10×50 (z preserved)
```

## Use the in-place kernels for maximum control

If you manage your own output buffers:

```julia
out = similar(data, Float32, (10, 10, 10))
blockwise_mean!(out, data, (10, 10, 5))
# `out` is now filled; no allocation occurred
```

Available in-place kernels:
- `blockwise_mean!`
- `blockwise_variance!`
- `blockwise_mean_variance!` (fused)
- `blockwise_min!`
- `blockwise_max!`
- `blockwise_product_mean!`
- `blockwise_joint_moments!`
