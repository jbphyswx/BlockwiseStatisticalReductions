"""
    WindowConfig{D}

Configuration for D-dimensional rolling or tiled windows.

# Fields
- `sizes::NTuple{D,Int}`: Window size in each dimension (e.g., (100, 100, 100))
- `strides::NTuple{D,Int}`: Step size between windows (default: 1 for rolling, window size for tiled)
- `padding::Symbol`: Padding mode (:valid, :same, :full)
"""
struct WindowConfig{D}
    sizes::NTuple{D,Int}
    strides::NTuple{D,Int}
    padding::Symbol
    
    function WindowConfig{D}(sizes::NTuple{D,Int}, 
                              strides::NTuple{D,Int}=ntuple(i->1, D),
                              padding::Symbol=:valid) where D
        @assert all(s > 0 for s in sizes) "Window sizes must be positive"
        @assert all(s > 0 for s in strides) "Strides must be positive"
        @assert padding in (:valid, :same, :full) "Invalid padding mode"
        new{D}(sizes, strides, padding)
    end
end

WindowConfig(sizes::NTuple{D,Int}, args...) where D = WindowConfig{D}(sizes, args...)
WindowConfig(sizes::Int...; kwargs...) = WindowConfig(sizes; kwargs...)
WindowConfig(sizes::Tuple; strides=sizes, padding=:valid) = 
    WindowConfig(sizes, strides, padding)

Base.ndims(::WindowConfig{D}) where D = D
Base.size(cfg::WindowConfig) = cfg.sizes

"""
    AbstractPlanNode

Base type for nodes in a reduction plan graph.
"""
abstract type AbstractPlanNode end

"""
    WindowNode

Plan node representing a windowing operation.
"""
struct WindowNode{C<:WindowConfig} <: AbstractPlanNode
    config::C
    id::UInt64
end

"""
    StatsNode{S}

Plan node representing a statistical reduction.
Type parameter S is a Symbol stored as Val{S} for type stability.
"""
struct StatsNode{S} <: AbstractPlanNode
    stat_type::Val{S}
    dims::Union{Colon,Vector{Int}}
    id::UInt64
    
    StatsNode(stat_type::Val{S}, dims, id) where S = new{S}(stat_type, dims, id)
end

"""
    TreeNode

Plan node representing a tree reduction (e.g., pairwise merge).
"""
struct TreeNode <: AbstractPlanNode
    arity::Int
    id::UInt64
end

"""
    UserNode{F}

Plan node representing a user-defined reduction function.
"""
struct UserNode{F} <: AbstractPlanNode
    f::F
    output_type::Type
    id::UInt64
end

"""
    MergeNode{F}

Plan node representing a merge of multiple branches in a DAG.
"""
struct MergeNode{F} <: AbstractPlanNode
    input_ids::Vector{UInt64}
    merge_fn::F
    id::UInt64
end

"""
    ReductionPlan

Graph structure representing a tree of reduction operations.
"""
mutable struct ReductionPlan
    nodes::Vector{AbstractPlanNode}
    edges::Dict{UInt64,Vector{UInt64}}
    inputs::Vector{UInt64}
    outputs::Vector{UInt64}
    
    function ReductionPlan()
        new(AbstractPlanNode[], Dict{UInt64,Vector{UInt64}}(), UInt64[], UInt64[])
    end
end

"""
    ReductionResult{T,M}

Holds computed statistics with attached metadata.

# Fields
- `data::T`: The computed result (scalar, array, or OnlineStat object)
- `metadata::M`: Dictionary with keys like :origin_indices, :window_config, 
  :operation_history, :count, :disk_path
"""
struct ReductionResult{T,M<:AbstractDict}
    data::T
    metadata::M
end

ReductionResult(data) = ReductionResult(data, Dict{Symbol,Any}())

Base.getindex(r::ReductionResult, key::Symbol) = r.metadata[key]
Base.haskey(r::ReductionResult, key::Symbol) = haskey(r.metadata, key)

"""
    AbstractExecutionBackend

Base type for execution backends (CPU, GPU, Distributed, etc.).
"""
abstract type AbstractExecutionBackend end

"""
    CPUBackend

Single-threaded or multi-threaded CPU execution.
"""
struct CPUBackend <: AbstractExecutionBackend
    nthreads::Int
end

CPUBackend() = CPUBackend(Threads.nthreads())

"""
    DistributedBackend

Execution across multiple Julia processes.
"""
struct DistributedBackend <: AbstractExecutionBackend
    procs::Vector{Int}
end

DistributedBackend() = DistributedBackend(Distributed.workers())

"""
    GPUBackend

Placeholder for GPU execution (actual implementation in extension).
"""
struct GPUBackend <: AbstractExecutionBackend
    device_id::Int
end

GPUBackend() = GPUBackend(0)

"""
    AbstractStorage

Base type for storage backends (memory or disk).
"""
abstract type AbstractStorage end

"""
    MemoryStorage

In-memory storage using a dictionary.
"""
mutable struct MemoryStorage <: AbstractStorage
    cache::Dict{UInt64,Any}
    max_size::Union{Int,Nothing}
    current_size::Int
    
    function MemoryStorage(; max_size=nothing)
        new(Dict{UInt64,Any}(), max_size, 0)
    end
end

"""
    DiskStorage

Disk-based storage with optional metadata.
"""
mutable struct DiskStorage <: AbstractStorage
    dir::String
    format::Symbol
    cache::Dict{UInt64,String}
    
    function DiskStorage(dir::String; format::Symbol=:jld2)
        isdir(dir) || mkpath(dir)
        @assert format in (:jld2, :serialization) "Unsupported format"
        new(dir, format, Dict{UInt64,String}())
    end
end

"""
    PlanCache

Cache for reduction plan intermediate results.
"""
mutable struct PlanCache{S<:AbstractStorage}
    storage::S
    hits::Int
    misses::Int
    
    function PlanCache(storage::S) where S<:AbstractStorage
        new{S}(storage, 0, 0)
    end
end

PlanCache() = PlanCache(MemoryStorage())
PlanCache(dir::String; kwargs...) = PlanCache(DiskStorage(dir; kwargs...))
