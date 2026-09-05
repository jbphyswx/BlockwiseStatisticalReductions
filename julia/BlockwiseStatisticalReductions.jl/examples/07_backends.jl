# The same request on several backends.
using BlockwiseStatisticalReductions
using ComputationalBackends: ComputationalBackends as CB
using OhMyThreads: OhMyThreads
using KernelAbstractions: KernelAbstractions as KA
using Random

Random.seed!(7)
x = randn(512, 512)
stats = (Mean(), Var())

serial = blockstats(x, [8, 16]; stats = stats, backend = CB.SerialBackend())
threaded = blockstats(x, [8, 16]; stats = stats, backend = CB.ThreadedBackend())
gpu = blockstats(x, [8, 16]; stats = stats, backend = CB.GPUBackend(KA.CPU()))

println("threads available: ", Threads.nthreads())
println("threaded matches serial exactly: ", threaded[(8, 8)].var == serial[(8, 8)].var)
println("KA.CPU matches serial exactly:   ", gpu[(8, 8)].var == serial[(8, 8)].var)

# `AutoBackend` is the default and picks for you: a device backend for device arrays,
# threads when more than one is available, else serial.
auto = blockstats(x, [8]; stats = stats)
println("auto agrees: ", auto[(8, 8)].mean ≈ serial[(8, 8)].mean)
