# BlockwiseStatisticalReductions.jl

Mergeable statistics (count, sum, mean, variance, covariance, extrema, moments) of an N-D tensor over
many window sizes at once. Windows are tiles or overlapping/anchored windows per axis; a planner builds
a tree of reductions (coarsen along divisors, compose along doubling chains) so the data is touched as
close to once as possible. Runs on CPU (serial, threaded), GPU (KernelAbstractions / CUDA) and across
processes (MPI, Distributed) through the ComputationalBackends.jl backend vocabulary.

Version 0.2.0 is a ground-up rebuild in progress. `AGENTS.md` describes the layering; the public API
lands with the `api/` layer.
