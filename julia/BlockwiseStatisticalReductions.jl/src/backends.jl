# ─────────────────────────────────────────────────────────────────────────────
# Execution-backend taxonomy
# ─────────────────────────────────────────────────────────────────────────────
#
# Names mirror the jbphyswx ecosystem (`SerialBackend`, `ThreadedBackend`,
# `GPUBackend{B}`, `AutoBackend`, `DistributedBackend{Inner}`, `MPIBackend{Inner}`),
# with two orthogonal concerns:
#
#   * Local compute backend — what one process/rank computes on:
#       `SerialBackend`    (always available; pure-Julia loops in core)
#       `ThreadedBackend`  (OhMyThreads extension)
#       `GPUBackend{B}`    (KernelAbstractions / CUDA extensions; `B` is a KA backend)
#
#   * Distribution wrapper — how work is split across processes, parametric over the
#     inner local backend so e.g. `DistributedBackend{GPUBackend{...}}` (multi-node,
#     multi-GPU) and `MPIBackend{ThreadedBackend}` (hybrid) are expressible. The wrapper
#     owns only "partition / merge"; `inner` owns the compute:
#       `DistributedBackend{Inner}` (Distributed extension)
#       `MPIBackend{Inner}`         (future MPI extension)
#
# `AutoBackend` resolves to the best *available* local backend at plan time (see
# `resolve_backend`). Heavy backend code lives in extensions; core defines only the
# dispatch types, a few helpers, and the always-available `SerialBackend` behaviour.

"""
    AbstractExecutionBackend

Supertype for everything that selects *how* a compiled plan is executed. Pass an
instance as the `backend` keyword to [`reduce_stats`](@ref) / [`run!`](@ref).
"""
abstract type AbstractExecutionBackend end

"Serial single-threaded CPU compute. Always available; no extension required."
struct SerialBackend <: AbstractExecutionBackend end

"Multithreaded CPU compute over output cells and independent DAG nodes. Requires `using OhMyThreads`."
struct ThreadedBackend <: AbstractExecutionBackend
    "Number of tasks per parallel region; `0` lets the scheduler choose (`nthreads`-based)."
    ntasks::Int
end
ThreadedBackend() = ThreadedBackend(0)

"""
    GPUBackend{B}(backend::B)

GPU compute on KernelAbstractions backend object `B` (e.g. a `CUDABackend`). Requires the
corresponding extension (`using CUDA`, or another KernelAbstractions device).
"""
struct GPUBackend{B} <: AbstractExecutionBackend
    backend::B
end

"Resolve to the best available local backend at plan time (see [`resolve_backend`](@ref))."
struct AutoBackend <: AbstractExecutionBackend end

"""
    DistributedBackend{Inner}(inner::Inner = SerialBackend())

Distribute the base pass across worker processes, each running `inner` locally, then merge
per-worker accumulators with the exact Chan/Pebay `merge`. Requires `using Distributed`.
"""
struct DistributedBackend{Inner<:AbstractExecutionBackend} <: AbstractExecutionBackend
    inner::Inner
end
DistributedBackend() = DistributedBackend(SerialBackend())

"""
    MPIBackend{Inner}(inner::Inner = SerialBackend())

Distribute the base pass across MPI ranks, each running `inner` locally. Parametric over the
inner local backend (not CPU-only: `MPIBackend{GPUBackend{...}}` targets multi-GPU). Requires a
future MPI extension.
"""
struct MPIBackend{Inner<:AbstractExecutionBackend} <: AbstractExecutionBackend
    inner::Inner
end
MPIBackend() = MPIBackend(SerialBackend())

"""
    local_backend(backend) -> AbstractExecutionBackend

The per-process compute backend: `inner` for distribution wrappers, the backend itself otherwise.
"""
local_backend(b::AbstractExecutionBackend) = b
local_backend(b::DistributedBackend) = b.inner
local_backend(b::MPIBackend) = b.inner

"`true` if `backend` distributes work across processes/ranks."
is_distributed(::AbstractExecutionBackend) = false
is_distributed(::DistributedBackend) = true
is_distributed(::MPIBackend) = true

# Set true by the OhMyThreads extension's __init__ so `AutoBackend` can prefer threads when loaded.
const _THREADING_AVAILABLE = Ref(false)

"""
    resolve_backend(backend) -> AbstractExecutionBackend

Concretize `AutoBackend` to an available backend at plan time. Core resolves `Auto` to
`SerialBackend`; the OhMyThreads extension overrides this to prefer `ThreadedBackend` when more
than one thread is available. All other backends resolve to themselves (distribution wrappers
resolve their `inner`).
"""
resolve_backend(b::AbstractExecutionBackend) = b
resolve_backend(::AutoBackend) =
    (_THREADING_AVAILABLE[] && Threads.nthreads() > 1) ? ThreadedBackend() : SerialBackend()
resolve_backend(b::DistributedBackend) = DistributedBackend(resolve_backend(b.inner))
resolve_backend(b::MPIBackend) = MPIBackend(resolve_backend(b.inner))
