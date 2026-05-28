module CUDAExt

using BlockwiseStatisticalReductions: BlockwiseStatisticalReductions
using CUDA: CUDA

# GPU backend implementation
function BlockwiseStatisticalReductions.best_backend(arr::CUDA.CuArray)
    return GPUBackend()
end

# Rolling views on GPU - use CUDA kernels
function BlockwiseStatisticalReductions.rolling_views(arr::CUDA.CuArray{T,N}, config::BlockwiseStatisticalReductions.WindowConfig{N}) where {T,N}
    # For GPU arrays, we need to materialize windows on GPU
    # This is a simplified implementation - full implementation would use custom kernels
    
    sz = size(arr)
    window_sz = config.sizes
    strides = config.strides
    
    # Calculate output dimensions
    if config.padding == :valid
        out_dims = ntuple(i -> max(0, div(sz[i] - window_sz[i], strides[i]) + 1), N)
    elseif config.padding == :same
        out_dims = ntuple(i -> div(sz[i] - 1, strides[i]) + 1, N)
    else
        out_dims = ntuple(i -> div(sz[i] + window_sz[i] - 2, strides[i]) + 1, N)
    end
    
    # Create output array that holds all windows (may be large!)
    # For memory-efficient version, would use custom iterator or chunked processing
    output = CUDA.zeros(T, (window_sz..., out_dims...))
    
    # Launch kernel to fill windows
    kernel = @cuda launch=false rolling_window_kernel!(output, arr, window_sz, strides, config.padding)
    config_cuda = CUDA.launch_configuration(kernel.fun)
    threads = min(prod(out_dims), config_cuda.threads)
    blocks = cld(prod(out_dims), threads)
    
    kernel(output, arr, window_sz, strides, config.padding; threads=threads, blocks=blocks)
    
    return GPURollingIterator(output, out_dims, window_sz)
end

function rolling_window_kernel!(output, arr, window_sz, strides, padding)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    
    sz = size(arr)
    out_dims = ntuple(i -> div(sz[i] - window_sz[i], strides[i]) + 1, ndims(arr))
    
    if idx <= prod(out_dims)
        # Convert linear index to multi-dimensional output index
        out_idx = Tuple(CUDA.CartesianIndices(out_dims)[idx])
        
        # Calculate window start
        starts = ntuple(i -> 1 + (out_idx[i] - 1) * strides[i], ndims(arr))
        
        # Copy window to output
        ntuple(ndims(arr)) do i
            for w_i in 1:window_sz[i]
                src_idx = starts[i] + w_i - 1
                if src_idx <= sz[i]
                    # Copy element
                    output[w_i, out_idx...] = arr[ntuple(j -> j == i ? src_idx : starts[j], ndims(arr))...]
                end
            end
        end
    end
    
    return nothing
end

# GPU rolling iterator
struct GPURollingIterator{T,N}
    data::CUDA.CuArray{T}
    out_dims::NTuple{N,Int}
    window_sz::NTuple{N,Int}
end

Base.length(iter::GPURollingIterator) = prod(iter.out_dims)

function Base.iterate(iter::GPURollingIterator{T,N}, state=1) where {T,N}
    state > length(iter) && return nothing
    
    # Get window at position state
    out_idx = Tuple(CUDA.CartesianIndices(iter.out_dims)[state])
    window_indices = ntuple(i -> (1:iter.window_sz[i], out_idx[i]), N)
    
    view_obj = @view iter.data[window_indices...]
    metadata = Dict{Symbol,Any}(
        :out_index => out_idx,
        :gpu => true
    )
    
    return (view_obj, metadata), state + 1
end

# Tree reduce on GPU
function BlockwiseStatisticalReductions.tree_reduce_impl(items::CuVector{T}, op, backend::GPUBackend) where T
    # Use CUDA's built-in reduce
    if op == (+)
        return CUDA.sum(items)
    elseif op == (*)
        return CUDA.prod(items)
    elseif op == (max)
        return CUDA.maximum(items)
    elseif op == (min)
        return CUDA.minimum(items)
    else
        # Fall back to CPU for custom operations
        cpu_items = Vector(items)
        return tree_reduce_impl(cpu_items, op, CPUBackend())
    end
end

# Parallel map for GPU
function BlockwiseStatisticalReductions.execute_parallel(backend::GPUBackend, f, items)
    # Launch kernel with f
    # Note: f must be GPU-compatible (no dynamic dispatch)
    error("Generic parallel execution on GPU not yet implemented")
end

end
