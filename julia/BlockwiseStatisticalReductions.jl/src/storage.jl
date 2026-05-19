"""
    store!(storage::MemoryStorage, key::UInt64, value)

Store a value in memory cache.
"""
function store!(storage::MemoryStorage, key::UInt64, value)
    storage.cache[key] = value
    return key
end

"""
    retrieve(storage::MemoryStorage, key::UInt64)

Retrieve a value from memory cache.
"""
function retrieve(storage::MemoryStorage, key::UInt64)
    return get(storage.cache, key, nothing)
end

"""
    haskey(storage::MemoryStorage, key::UInt64)

Check if key exists in memory cache.
"""
Base.haskey(storage::MemoryStorage, key::UInt64) = haskey(storage.cache, key)

"""
    store!(storage::DiskStorage, key::UInt64, value, metadata=nothing)

Store a value to disk with optional metadata.
"""
function store!(storage::DiskStorage, key::UInt64, value, metadata=nothing)
    filename = joinpath(storage.dir, "$(key).$(storage.format)")
    
    if storage.format == :jld2
        # JLD2 extension will provide this
        error("JLD2 format requires loading JLD2Ext extension")
    else
        # Use Julia's built-in serialization
        open(filename, "w") do io
            Serialization.serialize(io, (value, metadata))
        end
    end
    
    storage.cache[key] = filename
    return filename
end

"""
    retrieve(storage::DiskStorage, key::UInt64)

Retrieve a value from disk storage.
"""
function retrieve(storage::DiskStorage, key::UInt64)
    filename = get(storage.cache, key, nothing)
    if filename === nothing
        filename = joinpath(storage.dir, "$(key).$(storage.format)")
        isfile(filename) || return nothing
    end
    
    if storage.format == :jld2
        error("JLD2 format requires loading JLD2Ext extension")
    else
        open(filename, "r") do io
            value, metadata = Serialization.deserialize(io)
            return value
        end
    end
end

Base.haskey(storage::DiskStorage, key::UInt64) = 
    haskey(storage.cache, key) || isfile(joinpath(storage.dir, "$(key).$(storage.format)"))

"""
    clear!(storage::AbstractStorage)

Clear all stored values.
"""
function clear!(storage::MemoryStorage)
    empty!(storage.cache)
    return storage
end

function clear!(storage::DiskStorage)
    empty!(storage.cache)
    for file in readdir(storage.dir)
        rm(joinpath(storage.dir, file))
    end
    return storage
end
