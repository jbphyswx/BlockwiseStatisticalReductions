# Overlapping (sliding) windows, and covariance of a field pair.
#
#   julia --project=examples examples/sliding_windows.jl

using BlockwiseStatisticalReductions
using Random: Random
using Statistics: Statistics

Random.seed!(2)
data = randn(200, 200)

# 16×16 windows stepped every 4 cells (overlapping). Keyed by window size.
r = reduce_stats(data, [Sliding((16, 16); stride = (4, 4)), Sliding((32, 32); stride = (8, 8))];
                 stats = (Mean(), Var()))
@show r
println("sliding mean at 16×16/stride4: ", size(r[(16, 16)].mean), " array")

# stride == window reproduces the non-overlapping blockwise result:
sld = reduce_stats(data, Sliding((8, 8)); stats = (Mean(),))            # default stride == window
blk = reduce_stats(data, (8, 8); stats = (Mean(),))
println("sliding(stride=window) == blockwise: ", sld[(8, 8)].mean ≈ blk[(8, 8)].mean)

# Covariance of two fields, sliding:
x = randn(150, 150)
y = 0.7 .* x .+ 0.3 .* randn(150, 150)
rc = reduce_stats(x, y, Sliding((10, 10); stride = (5, 5)); stats = (Cov(),))
println("mean sliding covariance(x,y) at 10×10/stride5: ", round(Statistics.mean(rc[(10, 10)].cov); digits = 3))
