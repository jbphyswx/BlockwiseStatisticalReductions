# Numerical Stability

## The problem with naïve merging

Computing statistics hierarchically introduces numerical error if done
carelessly.  Consider merging the means of two blocks:

```
Block A: n=100, mean=1000.001
Block B: n=100, mean=1000.002
```

A naïve average gives the correct pooled mean.  But for **variance**, naïvely
averaging two block variances is **wrong** — the inter-block variability is
lost.

For higher moments and covariance, the problem compounds.

## Solution: sufficient statistics with stable merge

This package uses well-known numerically stable algorithms:

### Welford's online algorithm (single accumulator)

```
For each new sample x:
    n += 1
    δ = x - mean
    mean += δ / n
    δ₂ = x - mean
    M2 += δ * δ₂
```

`M2` is the sum of squared deviations.  `var = M2 / (n - 1)`.

### Chan's parallel algorithm (merge two accumulators)

```
n = n₁ + n₂
δ = μ₂ - μ₁
μ = (n₁μ₁ + n₂μ₂) / n
M2 = M2₁ + M2₂ + δ² × (n₁n₂ / n)
```

The key term `δ² × (n₁n₂ / n)` accounts for the inter-block variability.

### Pebay's extension (covariance)

```
C = C₁ + C₂ + (μx₂ - μx₁)(μy₂ - μy₁) × (n₁n₂ / n)
```

Same principle extended to the cross-term.

## Accumulators in this package

```julia
# Variance: stores (count, mean, M2)
acc = VarianceAccumulator{Float64}()
fit!(acc, data)
Statistics.mean(acc)  # μ
Statistics.var(acc)   # M2 / (n-1)

# Merge: exact as computing on concatenated data
merged = merge(acc1, acc2)

# Covariance: stores (count, mean_x, mean_y, C)
acc_xy = CovarianceAccumulator{Float64}()
fit!(acc_xy, x, y)
Statistics.cov(acc_xy)

# Raw moments: stores (count, moments[1..K])
acc_m = RawMomentsAccumulator{Float64, 4}()
fit!(acc_m, data)
```

## Merge kernels

For array-level operations (not single-value accumulators), the merge kernels
operate on pre-computed sufficient statistics arrays:

```julia
# Given arrays of means and M2 from sub-blocks, compute merged mean and M2
blockwise_merge_mean_M2!(mean_out, M2_out, means, M2s, counts, window_sizes)

# Merge covariance
blockwise_merge_covariance!(cov_out, mean_x, mean_y, C, counts, window_sizes)
```

These are used internally by the execution engine when a plan specifies
hierarchical reduction (e.g., multi-resolution towers).

## When does it matter?

Numerical stability matters most when:

- Data values are large relative to their variation (`mean >> std`)
- Block sizes are large (more samples per accumulator)
- Many merge levels (deep tower)
- Single precision (`Float32`) where roundoff accumulates faster

For typical geophysical data (temperature anomalies, humidity fields, wind
speeds), the stable algorithms prevent catastrophic cancellation that would
corrupt variance estimates at high reduction factors.
