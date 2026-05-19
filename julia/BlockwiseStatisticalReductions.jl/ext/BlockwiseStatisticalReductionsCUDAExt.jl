"""
CUDA extension for BlockwiseStatisticalReductions.

GPU-accelerated kernels for NVIDIA GPUs (A100, H100 targets).

To use: ensure CUDA.jl is loaded before using these functions.
"""

module BlockwiseStatisticalReductionsCUDAExt

using CUDA: CUDA
using BlockwiseStatisticalReductions: BlockwiseStatisticalReductions, WindowConfig

# Kernel configuration
const DEFAULT_THREADS = (16, 16, 1)
const MAX_THREADS = 256

"""
    cuda_blockwise_mean!(out, data, window_sizes)

CUDA kernel for blockwise mean computation.
"""
function cuda_blockwise_mean!(
    out::CUDA.CuArray{T,3},
    data::CUDA.CuArray{T,3},
    window_sizes::NTuple{3,Int}
) where {T}
    
    nx, ny, nz = size(out)
    fx, fy, fz = window_sizes
    
    # Launch configuration
    threads = (8, 8, 4)
    blocks = (cld(nx, threads[1]), cld(ny, threads[2]), cld(nz, threads[3]))
    
    kernel = @cuda launch=false _mean_kernel!(out, data, window_sizes, nx, ny, nz)
    kernel(out, data, window_sizes, nx, ny, nz; threads=threads, blocks=blocks)
    
    return out
end

function _mean_kernel!(
    out::CUDA.CuDeviceArray{T,3},
    data::CUDA.CuDeviceArray{T,3},
    window_sizes::NTuple{3,Int},
    nx::Int, ny::Int, nz::Int
) where {T}
    
    i = (CUDA.blockIdx().x - 1) * CUDA.blockDim().x + CUDA.threadIdx().x
    j = (CUDA.blockIdx().y - 1) * CUDA.blockDim().y + CUDA.threadIdx().y
    k = (CUDA.blockIdx().z - 1) * CUDA.blockDim().z + CUDA.threadIdx().z
    
    if i <= nx && j <= ny && k <= nz
        fx, fy, fz = window_sizes
        
        # Block start indices
        i0 = (i - 1) * fx + 1
        j0 = (j - 1) * fy + 1
        k0 = (k - 1) * fz + 1
        
        # Sum over block
        s = zero(T)
        for kk in 0:fz-1
            for jj in 0:fy-1
                for ii in 0:fx-1
                    s += data[i0+ii, j0+jj, k0+kk]
                end
            end
        end
        
        out[i, j, k] = s / (fx * fy * fz)
    end
    
    return nothing
end

"""
    cuda_blockwise_variance!(out, data, window_sizes; corrected=true)

CUDA kernel for blockwise variance using parallel Welford algorithm.
"""
function cuda_blockwise_variance!(
    out::CUDA.CuArray{T,3},
    data::CUDA.CuArray{T,3},
    window_sizes::NTuple{3,Int};
    corrected::Bool=true
) where {T}
    
    nx, ny, nz = size(out)
    fx, fy, fz = window_sizes
    
    threads = (8, 8, 4)
    blocks = (cld(nx, threads[1]), cld(ny, threads[2]), cld(nz, threads[3]))
    
    kernel = @cuda launch=false _variance_kernel!(out, data, window_sizes, corrected, nx, ny, nz)
    kernel(out, data, window_sizes, corrected, nx, ny, nz; threads=threads, blocks=blocks)
    
    return out
end

function _variance_kernel!(
    out::CUDA.CuDeviceArray{T,3},
    data::CUDA.CuDeviceArray{T,3},
    window_sizes::NTuple{3,Int},
    corrected::Bool,
    nx::Int, ny::Int, nz::Int
) where {T}
    
    i = (CUDA.blockIdx().x - 1) * CUDA.blockDim().x + CUDA.threadIdx().x
    j = (CUDA.blockIdx().y - 1) * CUDA.blockDim().y + CUDA.threadIdx().y
    k = (CUDA.blockIdx().z - 1) * CUDA.blockDim().z + CUDA.threadIdx().z
    
    if i <= nx && j <= ny && k <= nz
        fx, fy, fz = window_sizes
        block_size = fx * fy * fz
        
        i0 = (i - 1) * fx + 1
        j0 = (j - 1) * fy + 1
        k0 = (k - 1) * fz + 1
        
        # Welford's algorithm
        n = 0
        mean = zero(T)
        m2 = zero(T)
        
        for kk in 0:fz-1
            for jj in 0:fy-1
                for ii in 0:fx-1
                    x = data[i0+ii, j0+jj, k0+kk]
                    n += 1
                    delta = x - mean
                    mean += delta / n
                    delta2 = x - mean
                    m2 += delta * delta2
                end
            end
        end
        
        denom = corrected ? block_size - 1 : block_size
        out[i, j, k] = denom > 0 ? m2 / denom : zero(T)
    end
    
    return nothing
end

"""
    cuda_product_mean!(out, x, y, window_sizes)

CUDA kernel for product mean <x*y> without intermediate allocation.
"""
function cuda_product_mean!(
    out::CUDA.CuArray{T,3},
    x::CUDA.CuArray{T,3},
    y::CUDA.CuArray{T,3},
    window_sizes::NTuple{3,Int}
) where {T}
    
    nx, ny, nz = size(out)
    fx, fy, fz = window_sizes
    
    threads = (8, 8, 4)
    blocks = (cld(nx, threads[1]), cld(ny, threads[2]), cld(nz, threads[3]))
    
    kernel = @cuda launch=false _product_mean_kernel!(out, x, y, window_sizes, nx, ny, nz)
    kernel(out, x, y, window_sizes, nx, ny, nz; threads=threads, blocks=blocks)
    
    return out
end

function _product_mean_kernel!(
    out::CUDA.CuDeviceArray{T,3},
    x::CUDA.CuDeviceArray{T,3},
    y::CUDA.CuDeviceArray{T,3},
    window_sizes::NTuple{3,Int},
    nx::Int, ny::Int, nz::Int
) where {T}
    
    i = (CUDA.blockIdx().x - 1) * CUDA.blockDim().x + CUDA.threadIdx().x
    j = (CUDA.blockIdx().y - 1) * CUDA.blockDim().y + CUDA.threadIdx().y
    k = (CUDA.blockIdx().z - 1) * CUDA.blockDim().z + CUDA.threadIdx().z
    
    if i <= nx && j <= ny && k <= nz
        fx, fy, fz = window_sizes
        
        i0 = (i - 1) * fx + 1
        j0 = (j - 1) * fy + 1
        k0 = (k - 1) * fz + 1
        
        s = zero(T)
        for kk in 0:fz-1
            for jj in 0:fy-1
                for ii in 0:fx-1
                    s += x[i0+ii, j0+jj, k0+kk] * y[i0+ii, j0+jj, k0+kk]
                end
            end
        end
        
        out[i, j, k] = s / (fx * fy * fz)
    end
    
    return nothing
end

# Export GPU functions
export cuda_blockwise_mean!, cuda_blockwise_variance!, cuda_product_mean!

end  # module
