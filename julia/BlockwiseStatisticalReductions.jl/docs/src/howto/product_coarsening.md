# How-To: Product Coarsening

## The problem

In climate modeling, you often need `⟨x·y⟩` (the mean of a product) to compute
covariances via the identity:

```
Cov(x, y) = ⟨x·y⟩ - ⟨x⟩⟨y⟩
```

The naïve approach allocates a temporary array for `x .* y`, then reduces it.
For large 3D arrays this doubles memory pressure.

## Solution: fused product-mean

```julia
using BlockwiseStatisticalReductions

x = randn(Float32, 100, 100, 50)
y = randn(Float32, 100, 100, 50)
window = WindowConfig((10, 10, 5), (10, 10, 5), :valid)

# Compute ⟨x·y⟩ per block — no intermediate array allocated
mean_xy = product_mean(x, y, window)
# Result: 10×10×10 array
```

## Joint moments in one pass

Compute mean_x, mean_y, var_x, var_y, and cov_xy simultaneously:

```julia
jm = product_moments(x, y, window)
jm.mean_x    # 10×10×10
jm.mean_y    # 10×10×10
jm.var_x     # 10×10×10
jm.var_y     # 10×10×10
jm.cov_xy    # 10×10×10
```

One fused pass over the data produces all five statistics.

## Derive covariance from moments

If you already have `⟨x⟩`, `⟨y⟩`, and `⟨x·y⟩`:

```julia
cov_xy = covariance_from_moments(mean_x, mean_y, mean_xy)
# Elementwise: cov[i] = mean_xy[i] - mean_x[i] * mean_y[i]
```

Similarly for variance:

```julia
var_x = variance_from_moments(mean_x, mean_x_sq)
# Elementwise: var[i] = mean_x_sq[i] - mean_x[i]^2
```

## In-place kernels

For pre-allocated output buffers:

```julia
out = similar(x, Float32, (10, 10, 10))
blockwise_product_mean!(out, x, y, (10, 10, 5))
```

Joint moments with pre-allocated outputs:

```julia
mean_x  = similar(x, Float32, (10, 10, 10))
mean_y  = similar(x, Float32, (10, 10, 10))
mean_xy = similar(x, Float32, (10, 10, 10))
blockwise_joint_moments!(mean_x, mean_y, mean_xy, x, y, (10, 10, 5))
```

## Combining with multi-resolution

Product coarsening integrates with the plan-based system.  Compute `⟨x·y⟩` at
multiple scales:

```julia
# First compute product means at finest level
window = WindowConfig((2, 2, 1), (2, 2, 1), :valid)
mean_xy_2x = product_mean(x, y, window)

# Then feed into multi-resolution tower
plan = build_optimal_multires_plan(size(mean_xy_2x), [2, 4], [:mean]; dims=(1, 2))
results = execute(plan, mean_xy_2x)
```

## When to use

| Need | Function |
|------|----------|
| Just `⟨x·y⟩` | `product_mean(x, y, window)` |
| Full joint statistics | `product_moments(x, y, window)` |
| Blockwise covariance directly | `blockwise_covariance(x, y, window_sizes)` |
| In-place, pre-allocated | `blockwise_product_mean!(out, x, y, sizes)` |
