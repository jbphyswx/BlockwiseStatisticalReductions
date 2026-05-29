Test.@testset "PlanCache basic" begin
    cache = BlockwiseStatisticalReductions.PlanCache()
    
    # Initial state
    Test.@test cache.hits == 0
    Test.@test cache.misses == 0
    
    # Stats
    stats = BlockwiseStatisticalReductions.cache_stats(cache)
    Test.@test stats.hits == 0
    Test.@test stats.misses == 0
    Test.@test stats.hit_rate == 0.0
end

Test.@testset "get_or_compute!" begin
    cache = BlockwiseStatisticalReductions.PlanCache()
    key = UInt64(1)
    
    # First call - should compute
    result1 = BlockwiseStatisticalReductions.get_or_compute!(cache, key) do
        [1, 2, 3]
    end
    Test.@test result1 == [1, 2, 3]
    Test.@test cache.misses == 1
    Test.@test cache.hits == 0
    
    # Second call - should use cache
    result2 = BlockwiseStatisticalReductions.get_or_compute!(cache, key) do
        [4, 5, 6]  # Should not be called
    end
    Test.@test result2 == [1, 2, 3]  # Same as first
    Test.@test cache.misses == 1
    Test.@test cache.hits == 1
    
    # Stats updated
    stats = BlockwiseStatisticalReductions.cache_stats(cache)
    Test.@test stats.hit_rate == 0.5
end

Test.@testset "cache_key" begin
    cfg = BlockwiseStatisticalReductions.WindowConfig((10, 10))
    node = BlockwiseStatisticalReductions.WindowNode(cfg, UInt64(1))
    
    key1 = BlockwiseStatisticalReductions.cache_key(node, (100, 100))
    key2 = BlockwiseStatisticalReductions.cache_key(node, (100, 100))
    key3 = BlockwiseStatisticalReductions.cache_key(node, (200, 200))
    
    # Same config + shape = same key
    Test.@test key1 == key2
    # Different shape = different key
    Test.@test key1 != key3
    
    # Different node types
    rnode = BlockwiseStatisticalReductions.ReductionNode(BlockwiseStatisticalReductions.blockwise_mean!, cfg, (10, 10), UInt64(2))
    rkey = BlockwiseStatisticalReductions.cache_key(rnode, (100, 100))

    Test.@test key1 != rkey
end

Test.@testset "invalidate!" begin
    cache = BlockwiseStatisticalReductions.PlanCache()
    key = UInt64(1)
    
    BlockwiseStatisticalReductions.get_or_compute!(cache, key) do
        "test"
    end
    Test.@test haskey(cache.storage, key)
    
    BlockwiseStatisticalReductions.invalidate!(cache, key)
    Test.@test !haskey(cache.storage, key)
    
    # Invalidating non-existent key should not error
    BlockwiseStatisticalReductions.invalidate!(cache, UInt64(999))
end

Test.@testset "invalidate_all!" begin
    cache = BlockwiseStatisticalReductions.PlanCache()
    
    for i in 1:5
        BlockwiseStatisticalReductions.get_or_compute!(cache, UInt64(i)) do
            i
        end
    end
    
    Test.@test length(cache.storage.cache) == 5
    
    BlockwiseStatisticalReductions.invalidate_all!(cache)
    Test.@test isempty(cache.storage.cache)
end

Test.@testset "DiskCache" begin
    mktempdir() do dir
        cache = BlockwiseStatisticalReductions.PlanCache(dir; format=:serialization)
        key = UInt64(1)
        
        # Store via compute
        result = BlockwiseStatisticalReductions.get_or_compute!(cache, key) do
            rand(100, 100)
        end
        Test.@test size(result) == (100, 100)
        Test.@test cache.misses == 1
        
        # Retrieve from disk
        result2 = BlockwiseStatisticalReductions.get_or_compute!(cache, key) do
            zeros(100, 100)  # Should not be called
        end
        Test.@test result2 == result
        Test.@test cache.hits == 1
    end
end
