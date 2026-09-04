# Multi-scale gates: whole plans (planner + workspace + executor), serial.
#
# Two criteria, both derived rather than invented:
#   planner  — modelled cost of the plan / modelled cost of computing every target by its own base pass.
#              Below 1 means the plan shares work; how far below is set by what the targets allow.
#   executor — measured time / modelled time of the same plan. Checks the kernels hit the cost model.
# A single streaming `sum` is not a lower bound for a multi-scale request: every requested node must at
# least be written, and targets with no common divisor (distinct prime tile sizes) each need their own
# pass over the input.

function _plan_for(shape, T, targets, stats, fields)
    names = ntuple(i -> Symbol(:f, i), fields)
    C, _, _, _ = BSR.assemble(stats, names, T, BSR.accumulation_eltype(T))
    in_bytes = fields * sizeof(T)
    p = BSR.plan(shape, targets; in_bytes, acc_bytes = sizeof(C))
    return p, C, in_bytes
end

function _modelled(p, in_bytes, acc_bytes, N)
    limits = BSR.kernel_limits(CB.SerialBackend(), N)
    return sum(BSR.seconds(BSR.cost(p.how[k], p.nodes[k], p.nodes, in_bytes, acc_bytes), limits) for k in p.order)
end
function _all_base(p, targets, in_bytes, acc_bytes, N)
    limits = BSR.kernel_limits(CB.SerialBackend(), N)
    return sum(BSR.seconds(BSR.cost(BSR.Base_(), BSR.Node(w, true), p.nodes, in_bytes, acc_bytes), limits) for w in targets)
end

"Register a planner-sharing gate and an executor-efficiency gate for one multi-scale request."
function scale_gates(name, shape, T, targets_fn, stats; fields = 1, sharing, efficiency)
    N = length(shape)
    gate("planner", name, sharing) do
        targets = targets_fn(shape)
        p, C, in_bytes = _plan_for(shape, T, targets, stats, fields)
        return _modelled(p, in_bytes, sizeof(C), N) / _all_base(p, targets, in_bytes, sizeof(C), N)
    end
    gate("executor", name, efficiency) do
        data = ntuple(_ -> randn(T, shape...), fields)
        targets = targets_fn(shape)
        p, C, in_bytes = _plan_for(shape, T, targets, stats, fields)
        ws = BSR.allocate(p, C, data[1])
        BSR.run!(ws, p, data, SERIAL)
        t = best(() -> BSR.run!(ws, p, data, SERIAL))
        return t / _modelled(p, in_bytes, sizeof(C), N)
    end
end

# Sharing thresholds sit just above what the request allows: a tower is one base pass plus a coarsen
# chain (its floor is set by the traffic of the chain, which fusion will cut); a set of tile sizes needs
# one base pass per size with no common divisor; dense windows are dominated by their own output writes.
# Executor thresholds are 1.5: the cost model is calibrated against these kernels, so a plan that hits
# the model lands near or below 1.
scale_gates("6 scales [2..64] mean+var 4096² Float64", (4096, 4096), Float64,
            s -> BSR.resolve([2, 4, 8, 16, 32, 64], s), (BSR.Mean(), BSR.Var()); sharing = 0.30, efficiency = 1.2)
# Every tile size 2..64: the 18 prime sizes have no common divisor, so ~18 base passes are unavoidable;
# the 45 composite sizes must all come from coarsening.
scale_gates("tile sizes 2..64 mean 4096² Float64", (4096, 4096), Float64,
            s -> BSR.resolve(collect(2:64), s), (BSR.Mean(),); sharing = 0.35, efficiency = 1.5)
# Prime sizes times dyadic multiples: one base pass per prime, the rest coarsens.
scale_gates("prime tiles {3,5,7,11,13}x{1,2,4,8} mean 4096² Float64", (4096, 4096), Float64,
            s -> BSR.resolve([p * q for p in (3, 5, 7, 11, 13) for q in (1, 2, 4, 8)], s), (BSR.Mean(),);
            sharing = 0.30, efficiency = 1.5)
# Dense (stride-1) windows: outputs are as large as the input, so the write traffic dominates.
scale_gates("15 dense sizes 2..16 mean 1024² Float64", (1024, 1024), Float64,
            s -> [(BSR.strided(1024, w, 1, BSR.Truncate()), BSR.strided(1024, w, 1, BSR.Truncate())) for w in 2:16],
            (BSR.Mean(),); sharing = 0.25, efficiency = 1.5)
scale_gates("anchored Spread(3) 5 sizes mean 4096² Float64", (4096, 4096), Float64,
            s -> BSR.resolve(BSR.ScaleSet(BSR.Sizes([16, 32, 64, 128, 256]); placement = BSR.Spread(3)), s),
            (BSR.Mean(),); sharing = 1.0, efficiency = 1.5)
scale_gates("3-D 512x512x128 Float32, 15 anisotropic tiles, mean", (512, 512, 128), Float32,
            s -> BSR.resolve([(f, f, 1) for f in (2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32, 40, 48, 64)], s),
            (BSR.Mean(),); sharing = 0.28, efficiency = 1.5)

gate("planner", "peak workspace bytes / input bytes, 6 scales mean+var 4096²", 1.0) do
    shape = (4096, 4096)
    C, _, _, _ = BSR.assemble((BSR.Mean(), BSR.Var()), (:x,), Float64, Float64)
    p = BSR.plan(shape, BSR.resolve([2, 4, 8, 16, 32, 64], shape); acc_bytes = sizeof(C))
    return BSR.peak_bytes(p, sizeof(C)) / (prod(shape) * 8)
end

gate("planner", "plan build time for tile sizes 2..64 on 4096² (seconds)", 0.5) do
    shape = (4096, 4096)
    targets = BSR.resolve(collect(2:64), shape)
    BSR.plan(shape, targets)
    return best(() -> BSR.plan(shape, targets), 3)
end

gate("executor", "zero allocation: run! (bytes)", 0.0) do
    x = randn(1024, 1024)
    C, _, _, _ = BSR.assemble((BSR.Mean(), BSR.Var()), (:x,), Float64, Float64)
    p = BSR.plan((1024, 1024), BSR.resolve([2, 4, 8, 16], (1024, 1024)))
    ws = BSR.allocate(p, C, x)
    BSR.run!(ws, p, (x,), SERIAL)
    return Float64(@allocated BSR.run!(ws, p, (x,), SERIAL))
end
