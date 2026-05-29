using Random: Random

"""
    make_test_array(shape::Tuple; T=Float64, seed=1234)

Create a test array with reproducible random values.
"""
function make_test_array(shape::Tuple; T=Float64, seed=1234)
    rng = Random.MersenneTwister(seed)
    return rand(rng, T, shape...)
end

"""
    make_test_array_gpu(shape::Tuple; T=Float64, seed=1234)

Create a GPU test array (requires CUDA extension to be loaded).
"""
function make_test_array_gpu(shape::Tuple; T=Float64, seed=1234)
    if isdefined(BlockwiseStatisticalReductions, :CuArray)
        arr = make_test_array(shape; T=T, seed=seed)
        return BlockwiseStatisticalReductions.CuArray(arr)
    else
        error("CUDA not available")
    end
end

"""
    @test_backend_consistency expr

Test that an expression produces consistent results across backends.
"""
macro test_backend_consistency(expr)
    quote
        # CPU backend
        cpu_result = $(esc(expr))
        
        # OhMyThreads backend (if available)
        omt_result = nothing
        @static if isdefined(BlockwiseStatisticalReductions, :OhMyThreadsBackend)
            omt_result = $(esc(expr))
        end
        
        # GPU backend (if CUDA available)
        gpu_result = nothing
        @static if isdefined(BlockwiseStatisticalReductions, :CuArray)
            gpu_result = $(esc(expr))
        end
        
        # Compare results
        if omt_result !== nothing
            @test isapprox(cpu_result, omt_result; rtol=1e-10)
        end
        
        if gpu_result !== nothing
            @test isapprox(cpu_result, gpu_result; rtol=1e-6)
        end
        
        cpu_result
    end
end

"""
    benchmark_window_iteration(arr, config; n=10)

Benchmark window iteration performance.
"""
function benchmark_window_iteration(arr, config; n=10)
    times = Float64[]
    
    for _ in 1:n
        gc_state = gc_enable(false)
        t0 = time_ns()
        
        count = 0
        for (view, meta) in rolling_views(arr, config)
            count += 1
        end
        
        t1 = time_ns()
        gc_enable(gc_state)
        
        push!(times, (t1 - t0) / 1e9)
    end
    
    return (mean=mean(times), std=std(times), min=minimum(times), max=maximum(times), n_windows=count)
end
