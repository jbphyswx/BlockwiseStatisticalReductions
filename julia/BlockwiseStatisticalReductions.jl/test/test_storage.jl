Test.@testset "MemoryStorage" begin
    storage = BlockwiseStatisticalReductions.MemoryStorage()
    
    # Store and retrieve
    key = UInt64(1)
    value = [1, 2, 3]
    
    BlockwiseStatisticalReductions.store!(storage, key, value)
    Test.@test haskey(storage, key)
    
    retrieved = BlockwiseStatisticalReductions.retrieve(storage, key)
    Test.@test retrieved == value
    
    # Non-existent key
    Test.@test !haskey(storage, UInt64(999))
    Test.@test BlockwiseStatisticalReductions.retrieve(storage, UInt64(999)) === nothing
end

Test.@testset "MemoryStorage with max_size" begin
    storage = BlockwiseStatisticalReductions.MemoryStorage(max_size=1000)
    Test.@test storage.max_size == 1000
    
    # Store multiple items
    for i in 1:10
        BlockwiseStatisticalReductions.store!(storage, UInt64(i), rand(100))
    end
    
    Test.@test length(storage.cache) == 10
end

Test.@testset "DiskStorage (serialization)" begin
    mktempdir() do dir
        storage = BlockwiseStatisticalReductions.DiskStorage(dir; format=:serialization)
        
        # Store
        key = UInt64(42)
        value = Dict("array" => [1, 2, 3], "metadata" => "test")
        metadata = Dict(:timestamp => time())
        
        filename = BlockwiseStatisticalReductions.store!(storage, key, value, metadata)
        Test.@test isfile(filename)
        Test.@test haskey(storage, key)
        
        # Retrieve
        retrieved = BlockwiseStatisticalReductions.retrieve(storage, key)
        Test.@test retrieved["array"] == value["array"]
        
        # Non-existent key
        Test.@test !haskey(storage, UInt64(999))
        Test.@test BlockwiseStatisticalReductions.retrieve(storage, UInt64(999)) === nothing
    end
end

Test.@testset "clear! storage" begin
    # Memory
    mem = BlockwiseStatisticalReductions.MemoryStorage()
    BlockwiseStatisticalReductions.store!(mem, UInt64(1), "test")
    Test.@test haskey(mem, UInt64(1))
    
    BlockwiseStatisticalReductions.clear!(mem)
    Test.@test !haskey(mem, UInt64(1))
    Test.@test isempty(mem.cache)
    
    # Disk
    mktempdir() do dir
        disk = BlockwiseStatisticalReductions.DiskStorage(dir; format=:serialization)
        BlockwiseStatisticalReductions.store!(disk, UInt64(1), "test")
        
        BlockwiseStatisticalReductions.clear!(disk)
        Test.@test isempty(disk.cache)
        Test.@test isempty(readdir(dir))
    end
end

Test.@testset "DiskStorage roundtrip with complex data" begin
    mktempdir() do dir
        storage = BlockwiseStatisticalReductions.DiskStorage(dir; format=:serialization)
        
        # Complex nested structure
        data = Dict(
            :matrix => rand(10, 10),
            :vector => 1:100,
            :nested => Dict(:a => 1, :b => [1, 2, 3]),
            :nothing_val => nothing
        )
        
        key = UInt64(123)
        BlockwiseStatisticalReductions.store!(storage, key, data)
        
        retrieved = BlockwiseStatisticalReductions.retrieve(storage, key)
        Test.@test retrieved[:matrix] == data[:matrix]
        Test.@test retrieved[:vector] == data[:vector]
        Test.@test retrieved[:nested][:a] == 1
    end
end
