# API Reference: Accumulators

Online statistics accumulators with numerically stable parallel merge
capabilities.

## Types

```@docs
MergeableStatistic
VarianceAccumulator
CovarianceAccumulator
RawMomentsAccumulator
JointMomentsResult
```

## Fitting

```@docs
fit!
nobs
```

## Merging

```@docs
merge
merge_many
merge_all
```

## Extracting Values

Accumulators extend `Statistics.mean`, `Statistics.var`, and `Statistics.cov`:

```julia
acc = VarianceAccumulator{Float64}()
fit!(acc, data)

Statistics.mean(acc)              # Sample mean
Statistics.var(acc)               # Bessel-corrected variance
Statistics.var(acc; corrected=false)  # Population variance

acc_xy = CovarianceAccumulator{Float64}()
fit!(acc_xy, x, y)
Statistics.cov(acc_xy)            # Sample covariance
```

## Algorithms

| Accumulator | Algorithm | Merge complexity |
|-------------|-----------|-----------------|
| `VarianceAccumulator` | Welford (online) + Chan (parallel) | O(1) |
| `CovarianceAccumulator` | Pebay (parallel extension of Chan) | O(1) |
| `RawMomentsAccumulator` | Direct summation of `x^k` | O(K) |

### Sufficient statistics stored

| Accumulator | State |
|-------------|-------|
| `VarianceAccumulator{T}` | `count::Int`, `mean::T`, `sum_sq_dev::T` |
| `CovarianceAccumulator{T}` | `count::Int`, `mean_x::T`, `mean_y::T`, `co_moment::T` |
| `RawMomentsAccumulator{T,K}` | `count::Int`, `moments::NTuple{K,T}` |

### Merge correctness guarantee

For any partition of data into blocks A and B:

```julia
acc_full = VarianceAccumulator{Float64}()
fit!(acc_full, [A; B])

acc_a = VarianceAccumulator{Float64}()
fit!(acc_a, A)
acc_b = VarianceAccumulator{Float64}()
fit!(acc_b, B)
merged = merge(acc_a, acc_b)

@assert Statistics.mean(merged) ≈ Statistics.mean(acc_full)
@assert Statistics.var(merged) ≈ Statistics.var(acc_full)
```

This holds to floating-point precision regardless of block sizes or data
distributions.
