module BlockwiseStatisticalReductionsKernelAbstractionsExt

# Device-agnostic GPU (`GPUBackend{B}`) execution for BlockwiseStatisticalReductions
# via KernelAbstractions.jl (works across CUDA / ROCm / Metal / oneAPI backends).
#
# Provides KernelAbstractions `@kernel` implementations of the base reduction and the
# cross-scale merge, launched one DAG layer at a time with accumulator arrays kept
# resident on device. Filled in during the GPU phase.

end # module
