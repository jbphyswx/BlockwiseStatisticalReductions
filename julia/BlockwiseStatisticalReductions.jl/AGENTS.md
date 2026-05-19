# BlockwiseStatisticalReductions.jl Agent Guide

## Import Convention (CRITICAL)

**All external package imports MUST use fully qualified function calls.**

### Correct Pattern:
```julia
using Foo: Foo
# Then use: Foo.bar()

using Statistics: Statistics
# Then use: Statistics.mean(x)

using OnlineStats: OnlineStats
# Then use: OnlineStats.Mean(), OnlineStats.fit!()
```

### Forbidden Patterns:
```julia
# NEVER use bare using
using Foo

# NEVER use import
import Foo
import Foo: bar

# NEVER use selective using (without package prefix)
using Foo: bar, baz
```

### Why This Matters:
- **Maintainability**: Clear origin of every function call
- **Conflict Avoidance**: Multiple packages define `mean`, `sum`, `fit`, etc.
  - `Statistics.mean` vs `OnlineStats.mean`
  - `Base.sum` vs `OnlineStats.sum`
  - `StatsBase.fit` vs `OnlineStats.fit`
- **Explicit Dependencies**: No ambiguity about which package provides what

### Examples of Fully Qualified Calls:
```julia
# Statistics
Statistics.mean(x)
Statistics.var(x; corrected=true)
Statistics.std(x)

# StatsBase
StatsBase.fit(StatsBase.Histogram, data, edges)
StatsBase.quantile(x, p)

# OnlineStats
stat = OnlineStats.Mean(Float64)
OnlineStats.fit!(stat, x)
OnlineStats.value(stat)
OnlineStats.merge!(stat1, stat2)

# RollingFunctions
RollingFunctions.rolling(func, x, window)

# Distributed
Distributed.workers()
Distributed.@distributed
```

## Enforcement

All code reviews must verify:
1. No bare `using Package` statements
2. No `import` statements
3. All function calls use `Package.function()` format

Violations should be treated as blocking issues.
