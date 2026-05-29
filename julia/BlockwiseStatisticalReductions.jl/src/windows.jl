"""
    validate_window_config(array_shape::NTuple{N,Int}, config::WindowConfig{N}; strict::Bool=false) where N

Validate that window configuration divides evenly into array dimensions.

For blockwise (non-overlapping) windows with :valid padding, checks that:
- `array_shape[i] == n * config.sizes[i]` for some integer n

For rolling windows with overlap, checks that:
- `array_shape[i] == config.sizes[i] + (n-1) * config.strides[i]` for some integer n

Throws an error if dimensions don't divide evenly when strict=true.
"""
function validate_window_config(array_shape::NTuple{N,Int}, config::WindowConfig{N}; strict::Bool=false) where N
    if !strict
        return true
    end
    
    for i in 1:N
        arr_dim = array_shape[i]
        win_dim = config.sizes[i]
        stride = config.strides[i]
        
        if config.padding == :valid
            if stride == win_dim  # Non-overlapping (blockwise)
                # For exact tiling: array_dim must be multiple of window_dim
                if arr_dim % win_dim != 0
                    error("Blockwise window: dimension $i has size $arr_dim which is not divisible by window size $win_dim. " *
                          "Use padding=:same or :full for partial windows, or strict=false to skip this check.")
                end
            else  # Overlapping rolling windows
                # Check that windows land exactly: (arr_dim - win_dim) must be divisible by stride
                if (arr_dim - win_dim) % stride != 0
                    error("Rolling window: dimension $i with size $arr_dim, window $win_dim, stride $stride " *
                          "does not divide evenly. Final window would be partial. " *
                          "Use padding=:same/:full or strict=false to skip this check.")
                end
            end
        end
    end
    return true
end

"""
    rolling_views(array::AbstractArray{T,N}, config::WindowConfig{N}; strict::Bool=false) where {T,N}

Create a lazy iterator of views into `array` for each rolling window position.

Returns an iterator of `(view, metadata)` where metadata is a Dict with 
:indices (the range in each dimension) and :center (the center index).

If strict=true, validates that windows divide evenly into array dimensions.
"""
function rolling_views(array::AbstractArray{T,N}, config::WindowConfig{N}; strict::Bool=false) where {T,N}
    sz = size(array)
    window_sz = config.sizes
    strides = config.strides
    
    # Validate exact divisibility for blockwise operations
    validate_window_config(sz, config; strict=strict)
    
    # Calculate output dimensions based on padding mode
    if config.padding == :valid
        out_dims = ntuple(i -> max(0, div(sz[i] - window_sz[i], strides[i]) + 1), N)
        offset = ntuple(i -> 0, N)
    elseif config.padding == :same
        out_dims = ntuple(i -> div(sz[i] - 1, strides[i]) + 1, N)
        offset = ntuple(i -> div(window_sz[i] - 1, 2), N)
    elseif config.padding == :full
        out_dims = ntuple(i -> div(sz[i] + window_sz[i] - 2, strides[i]) + 1, N)
        offset = ntuple(i -> -(window_sz[i] - 1), N)
    else
        error("Unknown padding mode: $(config.padding)")
    end
    
    # Check if any dimension is 0 (window larger than array)
    if any(d == 0 for d in out_dims)
        return ()
    end
    
    # Create the iterator
    return RollingViewIterator(array, config, out_dims, offset, sz)
end

struct RollingViewIterator{T,N,A<:AbstractArray{T,N},C<:WindowConfig{N}}
    array::A
    config::C
    out_dims::NTuple{N,Int}
    offset::NTuple{N,Int}
    array_size::NTuple{N,Int}
end

Base.length(iter::RollingViewIterator{T,N}) where {T,N} = prod(iter.out_dims)
Base.eltype(::Type{RollingViewIterator{T,N,A,C}}) where {T,N,A,C} = Tuple{SubArray{T,N,A},Dict{Symbol,Any}}

function Base.iterate(iter::RollingViewIterator{T,N}, state=ntuple(i->0, N)) where {T,N}
    # state holds current multi-dimensional index in output space (0-based)
    total = length(iter)
    
    # Calculate output index
    if state[1] == 0
        # First iteration - start at index 1
        out_idx = ntuple(i -> 1, N)
        linear_idx = 1
    else
        # Convert 0-based state to 1-based linear index
        # state is (i, j, k) in 0..(out_dim-1)
        linear_idx = LinearIndices(iter.out_dims)[state...]
        linear_idx += 1  # Move to next position
        linear_idx > total && return nothing
        out_idx = CartesianIndices(iter.out_dims)[linear_idx].I
    end
    
    # Calculate window start and end in array coordinates
    starts = ntuple(N) do i
        s = 1 + (out_idx[i] - 1) * iter.config.strides[i] + iter.offset[i]
        clamp(s, 1, iter.array_size[i])
    end
    
    ends = ntuple(N) do i
        e = starts[i] + iter.config.sizes[i] - 1
        clamp(e, 1, iter.array_size[i])
    end
    
    # Create the view
    indices = ntuple(N) do i
        starts[i]:ends[i]
    end
    view_obj = @view iter.array[indices...]
    
    # Metadata
    metadata = Dict{Symbol,Any}(
        :indices => indices,
        :center => ntuple(N) do i
            c = starts[i] + div(iter.config.sizes[i] - 1, 2)
            clamp(c, 1, iter.array_size[i])
        end,
        :out_index => out_idx,
        :window_size => size(view_obj),
        :output_shape => iter.out_dims,
        :position => out_idx
    )
    
    # Next state
    next_linear = linear_idx
    next_state = CartesianIndices(iter.out_dims)[next_linear].I
    
    return (view_obj, metadata), next_state
end

"""
    tiled_blocks(array::AbstractArray{T,N}, blocksize::NTuple{N,Int}) where {T,N}

Create a lazy iterator of non-overlapping tiled blocks.

Similar to `rolling_views` but with stride equal to window size.
"""
function tiled_blocks(array::AbstractArray{T,N}, blocksize::NTuple{N,Int}) where {T,N}
    config = WindowConfig(blocksize, blocksize, :valid)
    return rolling_views(array, config)
end

tiled_blocks(array::AbstractArray, blocksize::Int...) = tiled_blocks(array, blocksize)

"""
    apply_windowed(f, array::AbstractArray, config::WindowConfig; 
                   combine=nothing, backend::AbstractExecutionBackend=CPUBackend())

Apply function `f` to each window of `array`, optionally combining results.

If `combine` is provided, results are combined pairwise using `combine`.
"""
function apply_windowed(f, array::AbstractArray, config::WindowConfig; 
                        combine=nothing, backend::AbstractExecutionBackend=CPUBackend())
    iter = rolling_views(array, config)
    
    if combine === nothing
        # Just collect all results
        return collect(f(view, meta) for (view, meta) in iter)
    else
        # Reduce using combine function
        # This is a tree reduction
        results = collect(f(view, meta) for (view, meta) in iter)
        return tree_reduce_impl(results, combine, backend)
    end
end

"""
    tree_reduce_impl(items, op, backend::CPUBackend)

Tree reduction implementation for CPU backend.
"""
function tree_reduce_impl(items, op, backend::CPUBackend)
    if length(items) == 0
        return nothing
    elseif length(items) == 1
        return items[1]
    end
    
    # Pairwise reduction
    T = eltype(items)
    while length(items) > 1
        new_items = Vector{T}(undef, div(length(items) + 1, 2))
        
        # Process pairs in parallel if multi-threaded
        if backend.nthreads > 1 && length(items) >= backend.nthreads * 2
            Base.Threads.@threads for i in 1:div(length(items), 2)
                idx1 = 2*i - 1
                idx2 = 2*i
                new_items[i] = op(items[idx1], items[idx2])
            end
            # Handle odd element
            if isodd(length(items))
                new_items[end] = items[end]
            end
        else
            for i in 1:div(length(items), 2)
                idx1 = 2*i - 1
                idx2 = 2*i
                new_items[i] = op(items[idx1], items[idx2])
            end
            if isodd(length(items))
                new_items[end] = items[end]
            end
        end
        
        items = new_items
    end
    
    return items[1]
end
