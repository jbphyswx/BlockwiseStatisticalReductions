# API Reference: Public API

High-level convenience functions for common blockwise operations.  These
accept raw arrays and return results directly — no plan construction needed.

## Blockwise Statistics

```@docs
blockwise_stats
blockwise_mean
blockwise_variance
blockwise_std
blockwise_covariance
blockwise_moments
```

## Multi-Resolution (High-Level)

```@docs
multiresolution_stats
```

## Product Coarsening

```@docs
product_mean
product_moments
product_variance
covariance_from_moments
variance_from_moments
blockwise_product_mean
blockwise_product_moments
```

## Hybrid Mode

```@docs
hybrid_reduction
execute_hybrid
HybridReductionSpec
HybridReductionResult
```
