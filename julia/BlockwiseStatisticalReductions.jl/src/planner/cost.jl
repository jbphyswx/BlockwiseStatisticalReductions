"""
    KernelLimits

What a backend's kernels want from a plan: `min_cells` per launch for full parallelism,
`max_tile_elements` per base box, the `fusion_tile` edge (cells per axis; `0` disables fusion), and the
roofline rates the cost model divides by — `bandwidth` (bytes/s), `merge_rate` (merges/s for merges that
are independent per cell, so they pipeline and vectorize) and `serial_merge_rate` (merges/s along a
carried dependency, which neither vectorizes nor overlaps).
"""
struct KernelLimits
    min_cells::Int
    max_tile_elements::Int
    fusion_tile::Int
    bandwidth::Float64
    merge_rate::Float64
    serial_merge_rate::Float64
    scan_ok::Bool
end
KernelLimits(min_cells, max_tile_elements, fusion_tile, bandwidth, merge_rate, serial_merge_rate) =
    KernelLimits(min_cells, max_tile_elements, fusion_tile, bandwidth, merge_rate, serial_merge_rate, true)

"Bytes moved, independent merges, and dependency-carried merges performed by one derivation."
struct Cost
    bytes::Float64
    merges::Float64
    serial_merges::Float64
end
Cost(bytes::Real, merges::Real) = Cost(Float64(bytes), Float64(merges), 0.0)
Base.:+(a::Cost, b::Cost) = Cost(a.bytes + b.bytes, a.merges + b.merges, a.serial_merges + b.serial_merges)
Base.zero(::Type{Cost}) = Cost(0.0, 0.0, 0.0)
"Roofline time of a cost under the limits: whichever resource the derivation saturates."
seconds(c::Cost, l::KernelLimits) =
    max(c.bytes / l.bandwidth, c.merges / l.merge_rate, c.serial_merges / l.serial_merge_rate)

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
# The two-stack scan amortizes about three merges per cell, but each one waits on the previous cell's
# stack state, so they land on `serial_merge_rate` rather than `merge_rate`.
function cost(d::Scan, node::Node, nodes, in_bytes::Int, acc_bytes::Int)
    pc = cells(nodes[d.parent])
    return Cost((pc + cells(node)) * acc_bytes, 0.0, 3.0 * pc)
end

# Rates measured on this package's serial kernels (benchmark/gates_kernels.jl): streaming bandwidth,
# independent merges near one per cycle, and the scan's dependency-carried merges about 5x slower (calibrated so the scan gate's measured/modelled ratio is ~1).
kernel_limits(::CB.AbstractSerialBackend, N::Int) = KernelLimits(1, 4096, N == 1 ? 4096 : N == 2 ? 64 : 16, 15e9, 1e9, 2e8)
kernel_limits(b::CB.AbstractExecutionBackend, N::Int) = missing_extension(b)
