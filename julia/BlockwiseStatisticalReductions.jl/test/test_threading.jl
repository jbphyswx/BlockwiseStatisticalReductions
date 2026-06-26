using OhMyThreads: OhMyThreads   # loads BlockwiseStatisticalReductionsOhMyThreadsExt
Random.seed!(7)

@testset "threading (OhMyThreads)" begin
    @test BSR._THREADING_AVAILABLE[]        # extension __init__ ran

    data = randn(192, 192)
    plan = tower_plan((192, 192); base_factor = (2, 2), steps = ([2, 3], [2]), maxfactor = (48, 48))

    # threaded run! is bit-identical to serial (per-cell independent work)
    bser = allocate_tower(plan, VarAcc{Float64}); run!(bser, plan, data, SerialBackend())
    bthr = allocate_tower(plan, VarAcc{Float64}); run!(bthr, plan, data, ThreadedBackend())
    for i in eachindex(plan.steps)
        @test step_result(bser, i) == step_result(bthr, i)
    end

    # reduce_stats with a threaded backend matches serial
    rser = reduce_stats(data, [4, 8, 16]; stats = (Mean(), Var(), Min()), backend = SerialBackend())
    rthr = reduce_stats(data, [4, 8, 16]; stats = (Mean(), Var(), Min()), backend = ThreadedBackend())
    for f in factors(rser)
        @test rser[f].mean == rthr[f].mean && rser[f].var == rthr[f].var && rser[f].min == rthr[f].min
    end

    # covariance threaded
    x = randn(160, 160); y = randn(160, 160)
    cser = reduce_stats(x, y, [10, 20]; stats = (Cov(),), backend = SerialBackend())
    cthr = reduce_stats(x, y, [10, 20]; stats = (Cov(),), backend = ThreadedBackend())
    for f in factors(cser)
        @test cser[f].cov == cthr[f].cov
    end

    # AutoBackend prefers threads only when more than one is available
    if Threads.nthreads() > 1
        @test resolve_backend(AutoBackend()) isa ThreadedBackend
    else
        @test resolve_backend(AutoBackend()) isa SerialBackend
    end
end
