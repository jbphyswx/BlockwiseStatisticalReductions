# Examples

Each script is self-contained and runnable:

```bash
julia --project=examples examples/01_basic.jl
```

| Script | Shows |
|---|---|
| `01_basic.jl` | one array, several tile sizes, reading the results |
| `02_multifield.jl` | variances and a covariance of two fields in one pass |
| `03_sliding.jl` | overlapping and anchored windows |
| `04_prepared.jl` | one plan reused over many inputs, allocation-free |
| `05_labelled.jl` | axis names, physical coordinates, sizes in physical units |
| `06_weights.jl` | weighted statistics and skipping gaps |
| `07_backends.jl` | threaded and GPU backends |
| `08_distributed.jl` | a tensor split across worker processes |
