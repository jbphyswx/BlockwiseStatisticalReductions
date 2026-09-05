Base.show(io::IO, r::ScaleResults{N}) where {N} =
    print(io, "ScaleResults{", N, "}(", r.input_shape, ": ", length(r), " window(s), ", join(_statnames(r), ", "), ")")

function Base.show(io::IO, ::MIME"text/plain", r::ScaleResults{N}) where {N}
    println(io, "ScaleResults over input ", r.input_shape)
    println(io, "  statistics: ", join(_statnames(r), ", "))
    println(io, "  ", length(r), " window(s):")
    for (w, res) in zip(r.windows, r.results)
        println(io, "    ", join(map(_describe_axis, w), " × "), "  → ", size(first(res)))
    end
    print(io, "  index by size, e.g. r[", map(aw -> aw.size, r.windows[1]), "].", first(_statnames(r)))
end
_statnames(r::ScaleResults) = r.statnames

Base.show(io::IO, p::Prepared{N}) where {N} =
    print(io, "Prepared{", N, "}(", p.input_shape, ": ", length(p.fieldnames), " field(s), ",
          length(p.plan.outputs), " window(s), ", nameof(typeof(p.backend)), ")")

function Base.show(io::IO, ::MIME"text/plain", p::Prepared{N}) where {N}
    println(io, "Prepared request over input ", p.input_shape)
    println(io, "  fields: ", join(p.fieldnames, ", "))
    println(io, "  statistics: ", join(map(name, p.stats), ", "))
    println(io, "  backend: ", p.backend)
    print(io, "  ")
    explain(io, p)
end

Base.show(io::IO, w::AxisWindow) = print(io, _describe_axis(w))
Base.show(io::IO, p::Plan{N}) where {N} =
    print(io, "Plan{", N, "}(", p.input_shape, ": ", length(p.nodes), " nodes, ", length(p.outputs),
          " requested, ", length(base_nodes(p)), " base pass(es))")
Base.show(io::IO, ::MIME"text/plain", p::Plan) = explain(io, p)
