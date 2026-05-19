"""
    Reusable buffer pool for zero-allocation hierarchical reductions.

Preallocate arrays for each reduction level to avoid per-operation allocation.
This is essential for performance in hot loops and multi-level reductions.
"""

#
# Buffer pool implementation
#

"""
    BufferPool

Pool of preallocated arrays for hierarchical reductions.

# Fields
- `buffers::Dict{NTuple{N,Int}, Vector{Array{T,N}}}`: Buffers indexed by shape
- `max_buffers_per_shape::Int`: Maximum buffers to keep per shape
"""
mutable struct BufferPool{T,N}
    buffers::Dict{NTuple{N,Int}, Vector{Array{T,N}}}
    max_buffers_per_shape::Int
    
    function BufferPool{T,N}(max_buffers::Int=4) where {T,N}
        new(Dict{NTuple{N,Int}, Vector{Array{T,N}}}(), max_buffers)
    end
end

BufferPool(T::Type, N::Int; max_buffers::Int=4) = BufferPool{T,N}(max_buffers)

"""
    acquire!(pool::BufferPool{T,N}, shape::NTuple{N,Int}) where {T,N}

Acquire a buffer of the given shape from the pool, or allocate new if none available.

# Returns
Array of requested shape, either from pool or newly allocated.
"""
function acquire!(pool::BufferPool{T,N}, shape::NTuple{N,Int}) where {T,N}
    if haskey(pool.buffers, shape) && !isempty(pool.buffers[shape])
        return pop!(pool.buffers[shape])
    end
    
    # No buffer available, allocate new
    return Array{T,N}(undef, shape)
end

"""
    release!(pool::BufferPool{T,N}, buffer::Array{T,N}) where {T,N}

Return a buffer to the pool for reuse.

If pool is full for this shape, buffer is dropped (will be GC'd).
"""
function release!(pool::BufferPool{T,N}, buffer::Array{T,N}) where {T,N}
    shape = size(buffer)
    
    if !haskey(pool.buffers, shape)
        pool.buffers[shape] = Array{T,N}[]
    end
    
    # Only keep if under limit
    if length(pool.buffers[shape]) < pool.max_buffers_per_shape
        push!(pool.buffers[shape], buffer)
    end
    
    return nothing
end

"""
    with_buffer!(f::Function, shape::NTuple{N,Int}, pool::BufferPool{T,N}) where {T,N}

Execute function with a pooled buffer, automatically releasing it afterwards.

# Example
```julia
pool = BufferPool(Float64, 3)
result = with_buffer!((10, 10, 10), pool) do buf
    # Use buf for computation
    compute_something!(buf, data)
    return sum(buf)
end
```
"""
function with_buffer!(f::Function, shape::NTuple{N,Int}, pool::BufferPool{T,N}) where {T,N}
    buffer = acquire!(pool, shape)
    try
        return f(buffer)
    finally
        release!(pool, buffer)
    end
end

#
# Level-aware buffer pool for multi-resolution reductions
#

"""
    LevelBufferPool{T,N}

Buffer pool organized by reduction level, with automatic shape tracking.

# Fields
- `pools::Dict{Int, BufferPool{T,N}}`: Separate pool per reduction level
- `level_shapes::Dict{Int, NTuple{N,Int}}`: Expected shapes per level
"""
mutable struct LevelBufferPool{T,N}
    pools::Dict{Int, BufferPool{T,N}}
    level_shapes::Dict{Int, NTuple{N,Int}}
    
    function LevelBufferPool{T,N}() where {T,N}
        new(Dict{Int, BufferPool{T,N}}(), Dict{Int, NTuple{N,Int}}())
    end
end

LevelBufferPool(T::Type, N::Int) = LevelBufferPool{T,N}()

"""
    register_level!(pool::LevelBufferPool{T,N}, level::Int, shape::NTuple{N,Int}) where {T,N}

Register a reduction level with its expected output shape.
"""
function register_level!(pool::LevelBufferPool{T,N}, level::Int, shape::NTuple{N,Int}) where {T,N}
    pool.level_shapes[level] = shape
    if !haskey(pool.pools, level)
        pool.pools[level] = BufferPool{T,N}()
    end
    return pool
end

"""
    acquire_level!(pool::LevelBufferPool{T,N}, level::Int) where {T,N}

Acquire buffer for specified reduction level.
"""
function acquire_level!(pool::LevelBufferPool{T,N}, level::Int) where {T,N}
    shape = pool.level_shapes[level]
    return acquire!(pool.pools[level], shape)
end

"""
    release_level!(pool::LevelBufferPool{T,N}, level::Int, buffer::Array{T,N}) where {T,N}

Release buffer back to level's pool.
"""
function release_level!(pool::LevelBufferPool{T,N}, level::Int, buffer::Array{T,N}) where {T,N}
    return release!(pool.pools[level], buffer)
end

#
# Convenience API
#

"""
    create_buffer_pool_for_factors(T::Type, input_shape::NTuple{N,Int}, factors::Vector{Int}) where N

Create a LevelBufferPool configured for multi-resolution reduction.

# Example
```julia
pool = create_buffer_pool_for_factors(Float64, (100, 100, 50), [2, 4, 8, 10])

# Use in reduction
buf = acquire_level!(pool, 2)
# ... compute 2x reduction into buf ...
release_level!(pool, 2, buf)
```
"""
function create_buffer_pool_for_factors(T::Type, input_shape::NTuple{N,Int}, factors::AbstractVector{Int}) where N
    pool = LevelBufferPool(T, N)
    
    for factor in factors
        level_shape = ntuple(i -> div(input_shape[i], factor), N)
        register_level!(pool, factor, level_shape)
    end
    
    return pool
end
