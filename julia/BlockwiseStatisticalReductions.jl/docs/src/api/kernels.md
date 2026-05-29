# API Reference: Kernels

Low-level in-place kernels that form the computational core of the package.
All higher-level APIs ultimately call these.

## Blockwise Reduction Kernels

These compute statistics over non-overlapping tiles, writing results into
pre-allocated output arrays.

```@docs
blockwise_mean!
blockwise_variance!
blockwise_mean_variance!
blockwise_min!
blockwise_max!
blockwise_product_mean!
blockwise_joint_moments!
blockwise_sum!
```

### Output shape contract

For all blockwise kernels:

```
size(out, d) == div(size(data, d), window_sizes[d])  ∀ d
```

The caller is responsible for allocating `out` with the correct shape.

### SIMD routing

For 3D `Float32`/`Float64` arrays larger than 10,000 elements, kernels
automatically route to `LoopVectorization.@turbo`-annotated inner loops.
Smaller arrays or non-float types use scalar fallbacks.

## Merge Kernels

Hierarchical sufficient-statistics composition.  Given block-level statistics,
these compute the merged statistics as if computed on the union of all blocks.

```@docs
blockwise_mean_M2!
blockwise_merge_mean_M2!
blockwise_mean_M2_M3!
blockwise_merge_mean_M2_M3!
blockwise_mean_C!
blockwise_merge_covariance!
blockwise_merge_raw_moments!
```

### Extracting final statistics from sufficient statistics

```@docs
variance_from_M2
std_from_M2
skewness_from_M2_M3
covariance_from_C
```

### Example: hierarchical variance

```julia
# Level 1: compute block means and M2 (sum of squared deviations)
mean1 = similar(data, Float32, out_shape)
M2_1  = similar(data, Float32, out_shape)
blockwise_mean_M2!(mean1, M2_1, data, (4, 4, 1))

# Level 2: merge blocks of means and M2
mean2 = similar(data, Float32, div.(out_shape, (2, 2, 1)))
M2_2  = similar(data, Float32, div.(out_shape, (2, 2, 1)))
blockwise_merge_mean_M2!(mean2, M2_2, mean1, M2_1,
    fill(prod((4,4,1)), out_shape), (2, 2, 1))

# Extract variance at level 2
var2 = variance_from_M2(M2_2, prod((4,4,1)) * prod((2,2,1)))
```

## SIMD Kernels

Explicit SIMD variants (automatically called by the canonical kernels for
eligible arrays, but can also be called directly):

```@docs
simd_blockwise_mean!
```
