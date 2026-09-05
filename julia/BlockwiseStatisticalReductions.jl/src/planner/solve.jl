# Greedy Steiner sharing: start from the targets, repeatedly add the candidate whose own cost is most
# outweighed by the savings it creates as a parent, then schedule and assign storage.

struct _Search{N}
    nodes::Vector{Node{N}}
    in_bytes::Int
    acc_bytes::Int
    limits::KernelLimits
end
_seconds(s::_Search, d::Derivation, i::Int) = seconds(cost(d, s.nodes[i], s.nodes, s.in_bytes, s.acc_bytes), s.limits)

# Windows equal off `axis` share a key; compose partners are looked up by it.
_offaxis_key(w::Window{N}, axis::Int) where {N} = ntuple(e -> e == axis ? nothing : w[e], Val(N))

# The cheapest derivation of node `i` from the materialized set (ids `M`), and its time.
function _best(s::_Search{N}, i::Int, M::Vector{Int}, keys::Dict) where {N}
    node = s.nodes[i]
    best_d::Derivation = Base_()
    best_t = _seconds(s, best_d, i)
    oversized = volume(node.window) > s.limits.max_tile_elements
    for q in M
        q == i && continue
        for d in _derivations_from(s, q, i, M, keys)
            t = _seconds(s, d, i)
            if t < best_t || (oversized && best_d isa Base_ && d isa Coarsen)
                best_t, best_d = t, d
            end
        end
    end
    return best_d, best_t
end

# Every derivation of `i` that uses `q` as a parent.
function _derivations_from(s::_Search{N}, q::Int, i::Int, M::Vector{Int}, keys::Dict) where {N}
    out = Derivation[]
    pw, cw = s.nodes[q].window, s.nodes[i].window
    if all(a -> can_coarsen(pw[a], cw[a]), 1:N)
        k = ntuple(a -> cw[a].size ÷ pw[a].size, Val(N))
        # The kernel writes the whole coarsened grid, so the child has to be it; a child holding only some
        # of its windows is reached by coarsening to the full grid and restriding onto the subset.
        all(a -> canonicalize(coarsen_result(pw[a], k[a])) == canonicalize(cw[a]), 1:N) && push!(out, Coarsen{N}(q, k))
    end
    if pw != cw && all(a -> can_restride(pw[a], cw[a]), 1:N)
        push!(out, Restride{N}(q, ntuple(a -> index_map(pw[a].pos, cw[a].pos, 0), Val(N))))
    end
    for axis in 1:N
        all(e -> e == axis || pw[e] == cw[e], 1:N) || continue
        pa, ca = pw[axis], cw[axis]
        if s.limits.scan_ok && _is_identity(pa) && _is_dense(ca) && ca.size > 1
            push!(out, Scan(q, axis, ca.size, ca.partial))
        end
        for r in get(keys, _offaxis_key(cw, axis), Int[])
            rw = s.nodes[r].window[axis]
            if pa.size + rw.size == ca.size && can_compose(pa, rw, ca)
                push!(out, Compose(q, r, axis, index_map(pa.pos, ca.pos, 0), index_map(rw.pos, ca.pos, pa.size)))
            end
            if rw.size + pa.size == ca.size && r != q && can_compose(rw, pa, ca)
                push!(out, Compose(r, q, axis, index_map(rw.pos, ca.pos, 0), index_map(pa.pos, ca.pos, rw.size)))
            end
        end
    end
    return out
end

function _keys(s::_Search{N}, M::Vector{Int}) where {N}
    keys = Dict{Any,Vector{Int}}()
    for q in M, axis in 1:N
        push!(get!(keys, _offaxis_key(s.nodes[q].window, axis), Int[]), q)
    end
    return keys
end

# Nodes a candidate could serve as a parent: the only ones whose cost adding it can change.
function _servable(s::_Search{N}, c::Int, of::Vector{Int}, keys::Dict) where {N}
    out = Int[]
    for i in of
        i == c && continue
        isempty(_derivations_from(s, c, i, Int[], keys)) || push!(out, i)
    end
    return out
end

"""
    plan(shape, targets::Vector{Window{N}}; backend, in_bytes, acc_bytes, memory_limit, chains) -> Plan{N}

Derivation DAG producing every target window from raw fields of `shape` at minimal roofline cost under
`kernel_limits(backend, N)`. `in_bytes` is the byte size of one observation over all fields and
`acc_bytes` the storage per accumulator cell. Dense doubling chains are streamed candidates (`chains`);
when the peak live storage would exceed `memory_limit` they are dropped and the plan is rebuilt.
"""
function plan(shape::NTuple{N,Int}, targets::AbstractVector{<:Window{N}}; backend::CB.AbstractExecutionBackend = CB.SerialBackend(),
              in_bytes::Int = 8, acc_bytes::Int = 24, memory_limit::Int = typemax(Int), chains::Bool = true) where {N}
    isempty(targets) && throw(ArgumentError("no target windows"))
    for w in targets
        map(aw -> aw.extent, w) == shape || throw(DimensionMismatch("target window extents $(map(aw -> aw.extent, w)) do not match shape $shape"))
    end
    limits = kernel_limits(backend, N)
    canonical = unique!(Window{N}[map(canonicalize, w) for w in targets])
    p = _solve(shape, canonical, limits, in_bytes, acc_bytes, chains)
    if peak_bytes(p, acc_bytes) > memory_limit
        chains || throw(ArgumentError("plan needs $(peak_bytes(p, acc_bytes)) bytes of storage, above the memory limit $memory_limit"))
        return plan(shape, targets; backend, in_bytes, acc_bytes, memory_limit, chains = false)
    end
    return p
end

function _solve(shape::NTuple{N,Int}, targets::Vector{Window{N}}, limits::KernelLimits, in_bytes::Int, acc_bytes::Int, chains::Bool) where {N}
    cands = candidates(targets, shape, limits; chains)
    nodes = Node{N}[Node(w, true) for w in targets]
    append!(nodes, Node{N}[Node(w, false) for w in cands])
    s = _Search{N}(nodes, in_bytes, acc_bytes, limits)
    M = collect(1:length(targets))
    pool = collect(length(targets)+1:length(nodes))
    # Splitters that keep base boxes within the tile limit are mandatory, not optional.
    mandatory = [k for k in pool if nodes[k].window in splitter_candidates(targets, shape, limits.max_tile_elements)]
    append!(M, mandatory)
    filter!(k -> !(k in mandatory), pool)
    keys = _keys(s, M)
    best = Dict{Int,Tuple{Derivation,Float64}}(i => _best(s, i, M, keys) for i in M)
    # A candidate can only change the cost of the nodes it can parent, so each round scores just those.
    servable = Dict(c => _servable(s, c, M, _keys(s, [c])) for c in pool)
    filter!(c -> !isempty(servable[c]), pool)
    while !isempty(pool)
        gain_best, pick, pick_best = 0.0, 0, Dict{Int,Tuple{Derivation,Float64}}()
        for c in pool
            Mc = vcat(M, c)
            keysc = _keys(s, Mc)
            own_d, own_t = _best(s, c, Mc, keysc)
            gain = -own_t
            updated = Dict{Int,Tuple{Derivation,Float64}}(c => (own_d, own_t))
            for i in servable[c]
                d, t = _best(s, i, Mc, keysc)
                if t < best[i][2]
                    gain += best[i][2] - t
                    updated[i] = (d, t)
                end
            end
            if gain > gain_best + 1e-12
                gain_best, pick, pick_best = gain, c, updated
            end
        end
        pick == 0 && break
        push!(M, pick)
        deleteat!(pool, findfirst(==(pick), pool))
        merge!(best, pick_best)
        keys = _keys(s, M)
        for c in pool
            union!(servable[c], _servable(s, c, [pick], _keys(s, [c])))
        end
        # Adding a node can make a previously computed derivation cheaper for anything it can parent.
        for i in _servable(s, pick, collect(M), keys)
            d, t = _best(s, i, M, keys)
            t < best[i][2] && (best[i] = (d, t))
        end
    end
    return _assemble(shape, s, M, best, length(targets), limits)
end

# Renumber the materialized nodes, order parents before children, assign storage.
function _assemble(shape::NTuple{N,Int}, s::_Search{N}, M::Vector{Int}, best::Dict, ntargets::Int, limits::KernelLimits) where {N}
    renum = Dict(q => k for (k, q) in enumerate(M))
    nodes = [s.nodes[q] for q in M]
    how = Derivation[_renumber(best[q][1], renum) for q in M]
    computed = [k for k in eachindex(nodes) if !is_view(how[k])]
    order = sort(computed; by = k -> (prod(sizes(nodes[k])), -cells(nodes[k]), k))
    buffer = _assign_buffers(nodes, how, order)
    outputs = [renum[q] for q in 1:ntargets]
    return Plan{N}(shape, nodes, how, order, buffer, outputs)
end
_renumber(d::Base_, renum) = d
_renumber(d::Coarsen{N}, renum) where {N} = Coarsen{N}(renum[d.parent], d.k)
_renumber(d::Compose, renum) = Compose(renum[d.a], renum[d.b], d.axis, d.amap, d.bmap)
_renumber(d::Restride{N}, renum) where {N} = Restride{N}(renum[d.parent], d.sel)
_renumber(d::Scan, renum) = Scan(renum[d.parent], d.axis, d.size, d.partial)

# Views own no storage; intermediates of equal shape reuse a buffer once their last consumer has run.
function _assign_buffers(nodes::Vector{Node{N}}, how::Vector{Derivation}, order::Vector{Int}) where {N}
    buffer = zeros(Int, length(nodes))
    last_use = Dict{Int,Int}()
    for (pos, k) in enumerate(order)
        for q in parents(how[k])
            base = is_view(how[q]) ? how[q].parent : q
            last_use[base] = max(get(last_use, base, 0), pos)
        end
    end
    free = Dict{NTuple{N,Int},Vector{Int}}()
    nbuf = 0
    for (pos, k) in enumerate(order)
        shp = nodes[k].shape
        pool = get(free, shp, Int[])
        if !nodes[k].requested && !isempty(pool)
            buffer[k] = pop!(pool)
        else
            nbuf += 1
            buffer[k] = nbuf
        end
        for q in order[1:pos]
            (nodes[q].requested || buffer[q] == 0) && continue
            get(last_use, q, 0) == pos && push!(get!(free, nodes[q].shape, Int[]), buffer[q])
        end
    end
    return buffer
end

"Bytes of accumulator storage the plan holds at its peak, for `acc_bytes` per cell."
function peak_bytes(p::Plan{N}, acc_bytes::Int) where {N}
    sizes_by_buffer = Dict{Int,Int}()
    for k in eachindex(p.nodes)
        p.buffer[k] == 0 && continue
        sizes_by_buffer[p.buffer[k]] = max(get(sizes_by_buffer, p.buffer[k], 0), cells(p.nodes[k]))
    end
    return sum(values(sizes_by_buffer); init = 0) * acc_bytes
end
