@testset "PlanCache basic" begin
    cache = PlanCache()
    
    # Initial state
    @test cache.hits == 0
    @test cache.misses == 0
    
    # Stats
    stats = cache_stats(cache)
    @test stats.hits == 0
    @test stats.misses == 0
    @test stats.hit_rate == 0.0
end

@testset "get_or_compute!" begin
    cache = PlanCache()
    key = UInt64(1)
    
    # First call - should compute
    result1 = get_or_compute!(cache, key) do
        [1, 2, 3]
    end
    @test result1 == [1, 2, 3]
    @test cache.misses == 1
    @test cache.hits == 0
    
    # Second call - should use cache
    result2 = get_or_compute!(cache, key) do
        [4, 5, 6]  # Should not be called
    end
    @test result2 == [1, 2, 3]  # Same as first
    @test cache.misses == 1
    @test cache.hits == 1
    
    # Stats updated
    stats = cache_stats(cache)
    @test stats.hit_rate == 0.5
end

@testset "cache_key" begin
    cfg = WindowConfig((10, 10))
    node = WindowNode(cfg, UInt64(1))
    
    key1 = cache_key(node, (100, 100))
    key2 = cache_key(node, (100, 100))
    key3 = cache_key(node, (200, 200))
    
    # Same config + shape = same key
    @test key1 == key2
    # Different shape = different key
    @test key1 != key3
    
    # Different node types
    stat = Mean{Float64}()
    snode = StatsNode{typeof(stat)}(stat, :, UInt64(2))
    skey = cache_key(snode, (100, 100))
    
    @test key1 != skey
end

@testset "invalidate!" begin
    cache = PlanCache()
    key = UInt64(1)
    
    get_or_compute!(cache, key) do
        "test"
    end
    @test haskey(cache.storage, key)
    
    invalidate!(cache, key)
    @test !haskey(cache.storage, key)
    
    # Invalidating non-existent key should not error
    invalidate!(cache, UInt64(999))
end

@testset "invalidate_all!" begin
    cache = PlanCache()
    
    for i in 1:5
        get_or_compute!(cache, UInt64(i)) do
            i
        end
    end
    
    @test length(cache.storage.cache) == 5
    
    invalidate_all!(cache)
    @test isempty(cache.storage.cache)
end

@testset "DiskCache" begin
    mktempdir() do dir
        cache = PlanCache(dir; format=:serialization)
        key = UInt64(1)
        
        # Store via compute
        result = get_or_compute!(cache, key) do
            rand(100, 100)
        end
        @test size(result) == (100, 100)
        @test cache.misses == 1
        
        # Retrieve from disk
        result2 = get_or_compute!(cache, key) do
            zeros(100, 100)  # Should not be called
        end
        @test result2 == result
        @test cache.hits == 1
    end
end
