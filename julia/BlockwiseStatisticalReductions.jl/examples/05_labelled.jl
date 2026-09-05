# Axis names, physical coordinates, and window sizes given in physical units.
using BlockwiseStatisticalReductions
using Random

Random.seed!(5)
x = randn(64, 64, 20)

# 500 m horizontal cells, and a vertical axis whose cells grow with height.
z_edges = cumsum([0.0; fill(20.0, 5); fill(40.0, 5); fill(80.0, 10)])
sp = (Regular(500.0), Regular(500.0), Edges(z_edges))
names = (:x, :y, :z)

# Horizontal windows of at most 4 km, the vertical axis left alone.
spec = ScaleSet((x = Dyadic(; max = Length(4000.0)), y = Dyadic(; max = Length(4000.0))))
r = blockstats(x, spec; stats = (Mean(), Var()), dimnames = names, spacing = sp)

println("sizes: ", scales(r))
w = first(windows(r))
g = geometry(r, w)
println("axis names:            ", g.names)
println("first x-cell bounds:   ", g.bounds[1][1])
println("first z-cell bounds:   ", g.bounds[3][1], "  (native, since z is unreduced)")

# A vertical coarsening whose output cells span the sum of the native cells they cover.
rv = blockstats(x, [(1, 1, 4)]; stats = (Mean(),), dimnames = names, spacing = sp)
println("z bounds after 4× coarsening: ", geometry(rv, (1, 1, 4)).bounds[3])
