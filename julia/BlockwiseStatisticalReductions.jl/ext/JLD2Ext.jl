module JLD2Ext

using BlockwiseStatisticalReductions
using JLD2: JLD2

"""
    BlockwiseStatisticalReductions.store!(storage::DiskStorage, key::UInt64, value, metadata)

Store to disk using JLD2 format.
"""
function BlockwiseStatisticalReductions.store!(storage::DiskStorage, key::UInt64, value, metadata=nothing)
    if storage.format != :jld2
        # Fallback to parent implementation
        return invoke(store!, Tuple{DiskStorage,UInt64,Any,Any}, storage, key, value, metadata)
    end
    
    filename = joinpath(storage.dir, "$(key).jld2")
    
    JLD2.jldsave(filename; value=value, metadata=metadata, key=key)
    
    storage.cache[key] = filename
    return filename
end

"""
    BlockwiseStatisticalReductions.retrieve(storage::DiskStorage, key::UInt64)

Retrieve from disk using JLD2 format.
"""
function BlockwiseStatisticalReductions.retrieve(storage::DiskStorage, key::UInt64)
    filename = get(storage.cache, key, nothing)
    if filename === nothing
        filename = joinpath(storage.dir, "$(key).jld2")
        isfile(filename) || return nothing
    end
    
    data = JLD2.load(filename)
    return data["value"]
end

"""
    BlockwiseStatisticalReductions.retrieve_with_metadata(storage::DiskStorage, key::UInt64)

Retrieve both value and metadata from JLD2 storage.
"""
function retrieve_with_metadata(storage::DiskStorage, key::UInt64)
    filename = get(storage.cache, key, nothing)
    if filename === nothing
        filename = joinpath(storage.dir, "$(key).jld2")
        isfile(filename) || return nothing
    end
    
    data = JLD2.load(filename)
    return (data["value"], get(data, "metadata", nothing))
end

end
