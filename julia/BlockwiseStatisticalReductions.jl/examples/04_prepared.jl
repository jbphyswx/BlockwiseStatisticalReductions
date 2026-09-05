# One plan, many inputs. `prepare` does everything that depends only on the request.
using BlockwiseStatisticalReductions
using ComputationalBackends: ComputationalBackends as CB
using Random

Random.seed!(4)
p = prepare(zeros(256, 256), [4, 8, 16]; stats = (Mean(), Var()), backend = CB.SerialBackend())

# The steady state allocates nothing at all.
x = randn(256, 256)
blockstats!(p, x)
println("bytes allocated per call: ", @allocated blockstats!(p, x))

# The result aliases the handle and is overwritten by the next call — copy what you need.
means = Float64[]
for _ in 1:5
    r = blockstats!(p, randn(256, 256))
    push!(means, sum(r[(16, 16)].mean))
end
println("five inputs through one plan: ", round.(means; digits = 3))
