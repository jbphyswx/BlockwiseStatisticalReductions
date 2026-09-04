"""
    KernelLimits

What a backend's kernels want from a plan: `min_cells` per launch for full parallelism,
`max_tile_elements` per base box, the `fusion_tile` edge (cells per axis; `0` disables fusion), and the
roofline rates `bandwidth` (bytes/s) and `merge_rate` (merges/s) the cost model divides by.
"""
struct KernelLimits
    min_cells::Int
    max_tile_elements::Int
    fusion_tile::Int
    bandwidth::Float64
    merge_rate::Float64
end

"Bytes moved and merges performed by one derivation."
struct Cost
    bytes::Float64
    merges::Float64
end
Base.:+(a::Cost, b::Cost) = Cost(a.bytes + b.bytes, a.merges + b.merges)
Base.zero(::Type{Cost}) = Cost(0.0, 0.0)
"Roofline time of a cost under the limits."
seconds(c::Cost, l::KernelLimits) = max(c.bytes / l.bandwidth, c.merges / l.merge_rate)

# Bytes a node's storage occupies per cell (uniform counts are not stored).
acc_cell_bytes(::Type{A}, uniform::Bool) where {A<:AbstractAccumulator} = sizeof(A) - (uniform ? _count_bytes(A) : 0)
_count_bytes(::Type{A}) where {A} = sum(k -> fieldname(A, k) === :n ? sizeof(fieldtype(A, k)) : _count_bytes(fieldtype(A, k)), 1:fieldcount(A); init = 0)
_count_bytes(::Type{T}) where {T<:Tuple} = sum(_count_bytes, fieldtypes(T); init = 0)
_count_bytes(::Type) = 0

"Cost of computing `node` by `how` from the plan's nodes; `in_bytes` per input element over all fields."
function cost(::Base_, node::Node, nodes, in_bytes::Int, acc_bytes::Int)
    v = volume(node.window)
    return Cost(cells(node) * (v * in_bytes + acc_bytes), cells(node) * v)
end
function cost(d::Coarsen, node::Node, nodes, in_bytes::Int, acc_bytes::Int)
    k = prod(d.k)
    return Cost(cells(node) * acc_bytes * (k + 1), cells(node) * k)
end
cost(::Compose, node::Node, nodes, in_bytes::Int, acc_bytes::Int) = Cost(3.0 * cells(node) * acc_bytes, Float64(cells(node)))
cost(::Restride, node::Node, nodes, in_bytes::Int, acc_bytes::Int) = zero(Cost)
function cost(d::Scan, node::Node, nodes, in_bytes::Int, acc_bytes::Int)
    pc = cells(nodes[d.parent])
    return Cost((pc + cells(node)) * acc_bytes, 3.0 * pc)
end

kernel_limits(::CB.AbstractSerialBackend, N::Int) = KernelLimits(1, 4096, N == 1 ? 4096 : N == 2 ? 64 : 16, 15e9, 1e9)
kernel_limits(b::CB.AbstractExecutionBackend, N::Int) = missing_extension(b)
