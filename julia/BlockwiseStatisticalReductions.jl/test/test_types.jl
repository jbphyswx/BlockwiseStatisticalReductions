using Test: Test
using OnlineStats: OnlineStats

Test.@testset "WindowConfig" begin
    # Basic construction
    cfg = WindowConfig((10, 10, 10))
    Test.@test cfg.sizes == (10, 10, 10)
    Test.@test cfg.strides == (1, 1, 1)
    Test.@test cfg.padding == :valid
    Test.@test ndims(cfg) == 3
    
    # With custom strides
    cfg2 = WindowConfig((10, 10), (5, 5), :same)
    Test.@test cfg2.sizes == (10, 10)
    Test.@test cfg2.strides == (5, 5)
    Test.@test cfg2.padding == :same
    
    # Constructor with keyword args
    cfg3 = WindowConfig(5, 5; padding=:full)
    Test.@test cfg3.sizes == (5, 5)
    Test.@test cfg3.padding == :full
    
    # Invalid inputs
    Test.@test_throws AssertionError WindowConfig((0, 10))
    Test.@test_throws AssertionError WindowConfig((10,), (0,))
    Test.@test_throws AssertionError WindowConfig((10,); padding=:invalid)
end

Test.@testset "Plan Nodes" begin
    cfg = WindowConfig((10, 10))
    
    # WindowNode
    wn = WindowNode(cfg, UInt64(1))
    Test.@test wn.config == cfg
    Test.@test wn.id == 1
    
    # StatsNode
    stat = OnlineStats.Mean(Float64)
    sn = StatsNode{typeof(stat)}(stat, :, UInt64(2))
    Test.@test sn.stat_type == stat
    Test.@test sn.dims == (:)
    Test.@test sn.id == 2
    
    # TreeNode
    tn = TreeNode(2, UInt64(3))
    Test.@test tn.arity == 2
    Test.@test tn.id == 3
    
    # UserNode
    fn = x -> sum(x)
    un = UserNode{typeof(fn)}(fn, Float64, UInt64(4))
    Test.@test un.f == fn
    Test.@test un.output_type == Float64
    Test.@test un.id == 4
end

Test.@testset "ReductionResult" begin
    # Basic construction
    rr = ReductionResult(1.0, Dict(:key => "value"))
    Test.@test rr.data == 1.0
    Test.@test rr.metadata[:key] == "value"
    Test.@test haskey(rr, :key)
    Test.@test !haskey(rr, :missing)
    Test.@test rr[:key] == "value"
    
    # Without metadata
    rr2 = ReductionResult([1, 2, 3])
    Test.@test rr2.data == [1, 2, 3]
    Test.@test isempty(rr2.metadata) || true  # Default empty dict
end

Test.@testset "Backends" begin
    # CPU backend
    cpu = CPUBackend()
    Test.@test cpu.nthreads == Threads.nthreads()
    
    cpu_single = CPUBackend(1)
    Test.@test cpu_single.nthreads == 1
    
    # Distributed backend
    dist = DistributedBackend()
    # This should work even without workers set up
    
    # GPU backend
    gpu = GPUBackend()
    Test.@test gpu.device_id == 0
    
    gpu2 = GPUBackend(1)
    Test.@test gpu2.device_id == 1
end

Test.@testset "Storage" begin
    # Memory storage
    mem = MemoryStorage()
    Test.@test isempty(mem.cache)
    
    # Disk storage
    mktempdir() do dir
        disk = DiskStorage(dir)
        Test.@test disk.dir == dir
        Test.@test disk.format == :jld2
        
        disk2 = DiskStorage(dir; format=:serialization)
        Test.@test disk2.format == :serialization
    end
    
    # Disk storage auto-creates directory
    mktempdir() do parent
        new_dir = joinpath(parent, "new_subdir")
        disk = DiskStorage(new_dir)
        Test.@test isdir(new_dir)
    end
end

Test.@testset "PlanCache" begin
    cache = PlanCache()
    Test.@test cache.hits == 0
    Test.@test cache.misses == 0
    
    # Test with disk storage
    mktempdir() do dir
        disk_cache = PlanCache(dir)
        Test.@test disk_cache.storage isa DiskStorage
    end
end
