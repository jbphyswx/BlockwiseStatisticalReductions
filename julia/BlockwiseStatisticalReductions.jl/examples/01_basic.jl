# Several tile sizes of one array, in one shared pass over the data.
using BlockwiseStatisticalReductions
using Random

Random.seed!(1)
x = randn(512, 512)

r = blockstats(x, [2, 4, 8, 16, 32, 64]; stats = (Mean(), Var()))

println("sizes produced: ", scales(r))
println("mean at 8×8:    ", size(r[(8, 8)].mean))
println("var  at 64×64:  ", size(r[(64, 64)].var))
println()

# Where does one output cell come from?
g = geometry(r, (8, 8))
println("output cell (1,1) covers input ", g.ranges[1][1], " × ", g.ranges[2][1])
println()

# What did the planner do?
explain(r)
