# Overlapping and anchored windows: the same machinery as tiles.
using BlockwiseStatisticalReductions
using Random

Random.seed!(3)
x = randn(128, 128)

# 16×16 windows every 4 cells.
sliding = blockstats(x, ScaleSet(Sizes([16]); placement = Stride(4)); stats = (Mean(), Var()))
println("sliding 16×16 stride 4: ", size(sliding[(16, 16)].mean))

# Every dense window size 2…8 at once. The planner builds a doubling chain and composes.
dense = blockstats(x, ScaleSet(Every(; min = 2, max = 8); placement = Dense()); stats = (Mean(),))
println("dense sizes: ", scales(dense))

# Windows at hand-picked anchors, and windows spread evenly across the axis.
anchored = blockstats(x, ScaleSet(Sizes([32]); placement = Anchors([0, 20, 50, 96])); stats = (Mean(),))
println("anchored origins: ", geometry(anchored, (32, 32)).origins[1])

spread = blockstats(x, ScaleSet(Sizes([32]); placement = Spread(4)); stats = (Mean(),))
println("spread origins:   ", geometry(spread, (32, 32)).origins[1])
