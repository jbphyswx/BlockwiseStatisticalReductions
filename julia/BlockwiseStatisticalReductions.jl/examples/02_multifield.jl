# Two fields, their variances and their covariance — one pass, not three.
using BlockwiseStatisticalReductions
using Random, Statistics

Random.seed!(2)
u = randn(256, 256)
w = 0.5 .* u .+ 0.5 .* randn(256, 256)

r = blockstats((u = u, w = w), [8, 16];
               stats = (var_u = Var(:u), var_w = Var(:w), flux = Cov(:u, :w),
                        n = Count(), corr = Corr(:u, :w)))

nt = r[(8, 8)]
println("statistics: ", statnames(r))
println("mean correlation over 8×8 tiles: ", mean(nt.corr))
println()

# The raw numerators, for results that will be merged again outside this package.
raw = blockstats((u = u, w = w), [8];
                 stats = (M2_u = Component(Var(:u), :M2), C = Component(Cov(:u, :w), :C), n = Count()))
println("M2 and C are sums, not averages: ", raw[(8, 8)].M2_u[1, 1], "  ", raw[(8, 8)].C[1, 1])
