module BlockwiseStatisticalReductionsCUDAExt

# CUDA specialization for BlockwiseStatisticalReductions (`GPUBackend{CUDABackend}`).
#
# Bundles KernelAbstractions (the kernels are written once, device-agnostically) with
# CUDA-specific glue: `CuArray` accumulator buffers, device selection, and any
# CUDA-only fast paths. Triggers only when BOTH KernelAbstractions and CUDA are loaded.
# Filled in during the GPU phase.

end # module
