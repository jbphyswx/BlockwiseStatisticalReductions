using BlockwiseStatisticalReductions: BlockwiseStatisticalReductions as BSR
using Printf: Printf
using Random: Random

# Roofline-relative performance gates. A gate measures a ratio (time of the case / time of a streaming
# `sum` over the same bytes, measured in this process) and passes when ratio <= threshold.
#
#   julia --project=benchmark benchmark/gates.jl [--only=group1,group2] [--list]
#
# Run these on an otherwise idle machine, and run each group in the process it describes:
#
#   julia --project=benchmark benchmark/gates.jl --only=kernels,kernels-f32,planner,executor
#   julia -t8 --project=benchmark benchmark/gates.jl --only=threaded
#
# Every ratio is against a streaming reference measured in the same process, so a loaded machine inflates
# both terms but not equally: a busy machine has been seen to push a kernel gate 30% above its quiet
# value, and the serial groups measure 20-70% higher inside a multithreaded process than a single-threaded
# one, whose allocator and garbage collector they are written against.
#
# Exit status is nonzero if any selected gate fails or if a selected group has no gates.

struct Gate{R, FT}
    group::String
    name::String
    threshold::FT
    run::R            # () -> ratio::Float64
end

const GATES = Gate[]

gate(f::Function, group::AbstractString, name::AbstractString, threshold::Real) =
    push!(GATES, Gate(String(group), String(name), Float64(threshold), f))

best(f, n::Int = 5) = minimum((GC.gc(); @elapsed f()) for _ in 1:n)

roofline(A::AbstractArray) = (sum(A); best(() -> sum(A)))

function parse_args(args)
    only = String[]
    list = false
    for a in args
        if startswith(a, "--only=")
            append!(only, split(a[8:end], ','; keepempty = false))
        elseif a == "--list"
            list = true
        else
            error("unknown argument $a")
        end
    end
    return only, list
end

function main(args = ARGS)
    Random.seed!(0)
    only, list = parse_args(args)
    selected = isempty(only) ? GATES : filter(g -> g.group in only, GATES)
    groups = isempty(only) ? unique(g.group for g in GATES) : only
    if list
        for g in selected
            Printf.@printf("%-12s %-48s <= %.2f\n", g.group, g.name, g.threshold)
        end
        return 0
    end
    failed = 0
    for grp in groups
        gs = filter(g -> g.group == grp, selected)
        if isempty(gs)
            println("group $grp: no gates registered — FAIL")
            failed += 1
            continue
        end
        for g in gs
            ratio = g.run()
            ok = ratio <= g.threshold
            failed += !ok
            Printf.@printf("%-12s %-48s ratio %7.3f  threshold %6.2f  %s\n", g.group, g.name, ratio, g.threshold, ok ? "ok" : "FAIL")
        end
    end
    println(failed == 0 ? "all gates passed" : "$failed gate(s) failed")
    return failed == 0 ? 0 : 1
end

include("gates_kernels.jl")
include("gates_multiscale.jl")
include("gates_threaded.jl")

if abspath(PROGRAM_FILE) == @__FILE__
    exit(main())
end