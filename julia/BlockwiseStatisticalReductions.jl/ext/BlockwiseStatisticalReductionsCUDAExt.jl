module BlockwiseStatisticalReductionsCUDAExt

# CUDA specialization. The kernels are the device-agnostic KernelAbstractions ones; what is CUDA-specific
# is what the planner should believe about the hardware, which is read from the device rather than
# assumed. Every reading is guarded: an attribute a driver reports as 0 (some are deprecated on newer
# architectures) falls back to the generic device limits rather than poisoning the cost model.

using CUDA: CUDA
using ComputationalBackends: ComputationalBackends as CB
using BlockwiseStatisticalReductions: BlockwiseStatisticalReductions as BSR

const CUDAGPU = CB.GPUBackend{CUDA.CUDABackend}

# Peak bandwidth from the memory clock and bus width; a streaming kernel reaches roughly 80% of it.
function _bandwidth(dev::CUDA.CuDevice)
    clock = CUDA.attribute(dev, CUDA.DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE)          # kHz
    bus = CUDA.attribute(dev, CUDA.DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH)      # bits
    (clock > 0 && bus > 0) || return nothing
    return 0.8 * 2 * (clock * 1e3) * (bus / 8)                                    # double data rate, bytes/s
end

function _lanes(dev::CUDA.CuDevice)
    sms = CUDA.attribute(dev, CUDA.DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
    threads = CUDA.attribute(dev, CUDA.DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)
    (sms > 0 && threads > 0) || return nothing
    return sms * threads
end

function BSR.kernel_limits(b::CUDAGPU, N::Int)
    generic = invoke(BSR.kernel_limits, Tuple{CB.AbstractGPUBackend,Int}, b, N)
    CUDA.functional() || return generic
    dev = CUDA.device()
    bandwidth = _bandwidth(dev)
    lanes = _lanes(dev)
    (bandwidth === nothing || lanes === nothing) && return generic
    # One merge per lane per cycle is optimistic; the planner only needs the ratio to bandwidth to be right.
    merges = 1e9 * lanes
    return BSR.KernelLimits(max(generic.min_cells, 8 * lanes), generic.max_tile_elements,
                            bandwidth, merges, merges, generic.scan_ok)
end

end
