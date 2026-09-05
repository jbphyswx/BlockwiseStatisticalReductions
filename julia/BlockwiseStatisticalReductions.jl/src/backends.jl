using ComputationalBackends: ComputationalBackends as CB

extension_trigger(::CB.AbstractThreadedBackend) = "OhMyThreads"
extension_trigger(::CB.AbstractGPUBackend) = "KernelAbstractions"
extension_trigger(::CB.AbstractMPIBackend) = "MPI"
extension_trigger(::CB.AbstractDistributedBackend) = "Distributed"

missing_extension(b::CB.AbstractExecutionBackend) =
    throw(ArgumentError("backend $(typeof(b)) requires `using $(extension_trigger(b))` to load its BlockwiseStatisticalReductions extension"))
missing_extension(b::CB.AbstractAutoBackend) =
    throw(ArgumentError("$(typeof(b)) must be resolved to a concrete backend before kernels run"))

"""
    kernel_limits(backend, N) -> KernelLimits

Planner limits for a backend on `N`-dimensional data (defined after the planner types are loaded).
"""
function kernel_limits end

# Set by each extension as it loads, so `AutoBackend` can prefer a backend that is actually available.
const THREADS_LOADED = Ref(false)
const GPU_LOADED = Ref(false)

"""
    resolve_backend(backend, fields::Tuple) -> backend

Concretize a backend for `fields`. Everything but `AutoBackend` resolves to itself;
`AutoBackend` picks the GPU backend when the fields are device arrays and its extension is loaded,
else threads when OhMyThreads is loaded and more than one is available, else serial.
ComputationalBackends deliberately leaves this to the consumer.
"""
resolve_backend(b::CB.AbstractExecutionBackend, fields::Tuple) = b
function resolve_backend(::CB.AbstractAutoBackend, fields::Tuple)
    GPU_LOADED[] && any(CB.is_gpu_array, fields) && return gpu_backend(fields)
    THREADS_LOADED[] && Threads.nthreads() > 1 && return CB.ThreadedBackend()
    return CB.SerialBackend()
end

"The GPU backend for `fields`; defined by the KernelAbstractions extension."
function gpu_backend end
