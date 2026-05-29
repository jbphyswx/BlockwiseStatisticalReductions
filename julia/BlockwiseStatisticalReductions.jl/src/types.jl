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

Plan node representing a windowing operation (legacy — use ReductionNode for new code).
"""
struct WindowNode{C<:WindowConfig} <: AbstractPlanNode
    config::C
    id::UInt64
end

"""
    ReductionNode{F}

Plan node that applies kernel function `F` to its inputs over a window.

The type parameter `F` is the concrete type of the kernel function
(e.g., `typeof(blockwise_mean!)`), enabling zero-overhead dispatch
through Julia's type system — no Symbol matching at runtime.

# Fields
- `kernel::F`: The kernel function to apply (e.g., `blockwise_mean!`)
- `config::WindowConfig`: Window sizes, strides, padding
- `output_shape::Tuple`: Pre-computed output shape for this node
- `id::UInt64`: Unique node identifier

# Example
```julia
node = ReductionNode(blockwise_mean!, WindowConfig((2,2,1),(2,2,1),:valid), (64,64,8), UInt64(1))
```
"""
struct ReductionNode{F, C<:WindowConfig} <: AbstractPlanNode
    kernel::F
    config::C
    output_shape::Tuple
    id::UInt64
end

"""
    SufficientStatsNode{F,M}

Plan node for sufficient-statistics computation and hierarchical merging.

Produces multiple output arrays (e.g., mean + M2 for variance towers).
- At the base level: `compute_kernel` extracts sufficient statistics from raw data
- At coarser levels: `merge_kernel` merges sufficient statistics from finer levels

# Fields
- `compute_kernel::F`: Kernel for base level (e.g., `blockwise_mean_M2!`)
- `merge_kernel::M`: Kernel for merge levels (e.g., `blockwise_merge_mean_M2!`)
- `config::WindowConfig`: Window sizes, strides, padding
- `output_shape::Tuple`: Pre-computed output shape per array
- `count_per_block::Int`: Number of samples in each finest-level block (for merge formula)
- `n_outputs::Int`: Number of output arrays (e.g., 2 for mean+M2)
- `is_base::Bool`: Whether this is the base level (use compute_kernel) or merge level
- `id::UInt64`: Unique node identifier
"""
struct SufficientStatsNode{F, M, C<:WindowConfig} <: AbstractPlanNode
    compute_kernel::F
    merge_kernel::M
    config::C
    output_shape::Tuple
    count_per_block::Int
    n_outputs::Int
    is_base::Bool
    id::UInt64
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
    ExecutionStep

Pre-compiled execution step: which node to run and where its inputs come from.

- `node`: The plan node to execute
- `input_indices`: Indices into the results vector for this node's inputs.
  Empty means this is a root node (uses the original input array).
- `result_index`: Where to store this node's output in the results vector.
"""
struct ExecutionStep
    node::AbstractPlanNode
    input_indices::Vector{Int}
    result_index::Int
end

"""
    ReductionPlan

Graph structure representing a tree of reduction operations.

After `finalize_plan`, the `execution_sequence` field contains a pre-compiled
flat execution order that eliminates all runtime graph traversal overhead.
"""
mutable struct ReductionPlan
    nodes::Vector{AbstractPlanNode}
    edges::Dict{UInt64,Vector{UInt64}}
    inputs::Vector{UInt64}
    outputs::Vector{UInt64}
    # Pre-compiled execution sequence (populated by finalize_plan)
    execution_sequence::Vector{ExecutionStep}
    output_indices::Vector{Int}
    
    function ReductionPlan()
        new(AbstractPlanNode[], Dict{UInt64,Vector{UInt64}}(), UInt64[], UInt64[],
            ExecutionStep[], Int[])
    end
end

"""
    ReductionResult{T}

Holds computed reduction results with minimal metadata.
Type-stable and allocation-minimal - no Dict overhead.

# Fields
- `data::T`: The computed result (array, scalar, or OnlineStat)
- `shape::Tuple`: Original input shape (for verification)
"""
struct ReductionResult{T}
    data::T
    shape::Tuple
end

Base.size(r::ReductionResult) = size(r.data)
Base.eltype(r::ReductionResult{T}) where T = eltype(r.data)

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

#
# ─── Pretty printing ──────────────────────────────────────────────────────────
#

function Base.show(io::IO, w::WindowConfig{D}) where D
    blockwise = w.sizes == w.strides
    if blockwise
        print(io, "WindowConfig{$D}(block=$(w.sizes), :$(w.padding))")
    else
        print(io, "WindowConfig{$D}(size=$(w.sizes), stride=$(w.strides), :$(w.padding))")
    end
end

function Base.show(io::IO, node::WindowNode)
    w = node.config
    blockwise = w.sizes == w.strides
    if blockwise
        print(io, "WindowNode(id=$(node.id), block=$(w.sizes))")
    else
        print(io, "WindowNode(id=$(node.id), size=$(w.sizes), stride=$(w.strides))")
    end
end

function Base.show(io::IO, node::ReductionNode)
    fname = nameof(node.kernel)
    w = node.config
    print(io, "ReductionNode(id=$(node.id), $(fname), block=$(w.sizes) → $(node.output_shape))")
end

function Base.show(io::IO, node::SufficientStatsNode)
    kname = node.is_base ? nameof(node.compute_kernel) : nameof(node.merge_kernel)
    phase = node.is_base ? "compute" : "merge"
    w = node.config
    print(io, "SufficientStatsNode(id=$(node.id), $(kname) [$(phase)], block=$(w.sizes) → $(node.output_shape), $(node.n_outputs) outputs)")
end
Base.show(io::IO, node::TreeNode) = print(io, "TreeNode(id=$(node.id), arity=$(node.arity))")
Base.show(io::IO, node::UserNode) = print(io, "UserNode(id=$(node.id), f=$(node.f))")
Base.show(io::IO, node::MergeNode) = print(io, "MergeNode(id=$(node.id), inputs=$(node.input_ids))")

function Base.show(io::IO, step::ExecutionStep)
    if isempty(step.input_indices)
        print(io, "Step $(step.result_index): $(step.node) ← input")
    else
        print(io, "Step $(step.result_index): $(step.node) ← steps $(step.input_indices)")
    end
end

function Base.show(io::IO, r::ReductionResult)
    d = r.data
    if d isa AbstractArray
        print(io, "ReductionResult($(typeof(d).name.name){$(eltype(d)),$(ndims(d))}, size=$(size(d)))")
    else
        print(io, "ReductionResult($(typeof(d)))")
    end
end

function Base.summary(io::IO, plan::ReductionPlan)
    n_nodes = length(plan.nodes)
    n_outputs = length(plan.outputs)
    n_steps = length(plan.execution_sequence)
    compiled = n_steps > 0 ? "compiled" : "not compiled"
    print(io, "ReductionPlan: $n_nodes nodes, $n_outputs outputs ($compiled)")
end

function Base.show(io::IO, plan::ReductionPlan)
    summary(io, plan)
end

function Base.show(io::IO, ::MIME"text/plain", plan::ReductionPlan)
    summary(io, plan)
    isempty(plan.nodes) && return

    # Build lookup tables from the plan's own data
    id_to_node = Dict{UInt64, AbstractPlanNode}(n.id => n for n in plan.nodes)
    output_set = Set(plan.outputs)
    child_ids_of = plan.edges

    # Find root nodes (not a child of any other node)
    all_children = Set{UInt64}()
    for kids in values(child_ids_of)
        union!(all_children, kids)
    end
    roots = UInt64[n.id for n in plan.nodes if !(n.id in all_children)]

    # Print DAG as indented tree
    println(io)
    println(io, "  DAG:")
    visited = Set{UInt64}()
    for rid in roots
        _print_dag_node(io, rid, id_to_node, child_ids_of, output_set, visited, 2)
    end
end

function _print_dag_node(io::IO, nid::UInt64, id_to_node, child_ids_of, output_set, visited, depth; max_depth=6)
    indent = "  " ^ depth
    node = get(id_to_node, nid, nothing)
    node === nothing && return
    marker = nid in output_set ? " ★" : ""

    if nid in visited
        println(io, indent, "└ ", node, marker, " (see above)")
        return
    end
    push!(visited, nid)

    println(io, indent, "├ ", node, marker)
    kids = get(child_ids_of, nid, UInt64[])
    if depth >= max_depth && !isempty(kids)
        println(io, indent, "│ └ … ", length(kids), " children (truncated)")
        return
    end
    for kid in kids
        _print_dag_node(io, kid, id_to_node, child_ids_of, output_set, visited, depth + 1; max_depth=max_depth)
    end
end
