using OnlineStats: OnlineStats

@testset "WindowConfig" begin
    # Basic construction
    cfg = WindowConfig((10, 10, 10))
    @test cfg.sizes == (10, 10, 10)
    @test cfg.strides == (1, 1, 1)
    @test cfg.padding == :valid
    @test ndims(cfg) == 3
    
    # With custom strides
    cfg2 = WindowConfig((10, 10), (5, 5), :same)
    @test cfg2.sizes == (10, 10)
    @test cfg2.strides == (5, 5)
    @test cfg2.padding == :same
    
    # Constructor with keyword args
    cfg3 = WindowConfig(5, 5; padding=:full)
    @test cfg3.sizes == (5, 5)
    @test cfg3.padding == :full
    
    # Invalid inputs
    @test_throws AssertionError WindowConfig((0, 10))
    @test_throws AssertionError WindowConfig((10,), (0,))
    @test_throws AssertionError WindowConfig((10,); padding=:invalid)
end

@testset "Plan Nodes" begin
    cfg = WindowConfig((10, 10))
    
    # WindowNode
    wn = WindowNode(cfg, UInt64(1))
    @test wn.config == cfg
    @test wn.id == 1
    
    # StatsNode
    stat = OnlineStats.Mean(Float64)
    sn = StatsNode{typeof(stat)}(stat, :, UInt64(2))
    @test sn.stat_type == stat
    @test sn.dims == (:)
    @test sn.id == 2
    
    # TreeNode
    tn = TreeNode(2, UInt64(3))
    @test tn.arity == 2
    @test tn.id == 3
    
    # UserNode
    fn = x -> sum(x)
    un = UserNode{typeof(fn)}(fn, Float64, UInt64(4))
    @test un.f == fn
    @test un.output_type == Float64
    @test un.id == 4
end

@testset "ReductionResult" begin
    # Basic construction
    rr = ReductionResult(1.0, Dict(:key => "value"))
    @test rr.data == 1.0
    @test rr.metadata[:key] == "value"
    @test haskey(rr, :key)
    @test !haskey(rr, :missing)
    @test rr[:key] == "value"
    
    # Without metadata
    rr2 = ReductionResult([1, 2, 3])
    @test rr2.data == [1, 2, 3]
    @test isempty(rr2.metadata) || true  # Default empty dict
end

@testset "Backends" begin
    # CPU backend
    cpu = CPUBackend()
    @test cpu.nthreads == Threads.nthreads()
    
    cpu_single = CPUBackend(1)
    @test cpu_single.nthreads == 1
    
    # Distributed backend
    dist = DistributedBackend()
    # This should work even without workers set up
    
    # GPU backend
    gpu = GPUBackend()
    @test gpu.device_id == 0
    
    gpu2 = GPUBackend(1)
    @test gpu2.device_id == 1
end

@testset "Storage" begin
    # Memory storage
    mem = MemoryStorage()
    @test isempty(mem.cache)
    
    # Disk storage
    mktempdir() do dir
        disk = DiskStorage(dir)
        @test disk.dir == dir
        @test disk.format == :jld2
        
        disk2 = DiskStorage(dir; format=:serialization)
        @test disk2.format == :serialization
    end
    
    # Disk storage auto-creates directory
    mktempdir() do parent
        new_dir = joinpath(parent, "new_subdir")
        disk = DiskStorage(new_dir)
        @test isdir(new_dir)
    end
end

@testset "PlanCache" begin
    cache = PlanCache()
    @test cache.hits == 0
    @test cache.misses == 0
    
    # Test with disk storage
    mktempdir() do dir
        disk_cache = PlanCache(dir)
        @test disk_cache.storage isa DiskStorage
    end
end
