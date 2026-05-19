@testset "MemoryStorage" begin
    storage = MemoryStorage()
    
    # Store and retrieve
    key = UInt64(1)
    value = [1, 2, 3]
    
    BlockwiseStatisticalReductions.store!(storage, key, value)
    @test haskey(storage, key)
    
    retrieved = BlockwiseStatisticalReductions.retrieve(storage, key)
    @test retrieved == value
    
    # Non-existent key
    @test !haskey(storage, UInt64(999))
    @test BlockwiseStatisticalReductions.retrieve(storage, UInt64(999)) === nothing
end

@testset "MemoryStorage with max_size" begin
    storage = MemoryStorage(max_size=1000)
    @test storage.max_size == 1000
    
    # Store multiple items
    for i in 1:10
        BlockwiseStatisticalReductions.store!(storage, UInt64(i), rand(100))
    end
    
    @test length(storage.cache) == 10
end

@testset "DiskStorage (serialization)" begin
    mktempdir() do dir
        storage = DiskStorage(dir; format=:serialization)
        
        # Store
        key = UInt64(42)
        value = Dict("array" => [1, 2, 3], "metadata" => "test")
        metadata = Dict(:timestamp => time())
        
        filename = BlockwiseStatisticalReductions.store!(storage, key, value, metadata)
        @test isfile(filename)
        @test haskey(storage, key)
        
        # Retrieve
        retrieved = BlockwiseStatisticalReductions.retrieve(storage, key)
        @test retrieved["array"] == value["array"]
        
        # Non-existent key
        @test !haskey(storage, UInt64(999))
        @test BlockwiseStatisticalReductions.retrieve(storage, UInt64(999)) === nothing
    end
end

@testset "clear! storage" begin
    # Memory
    mem = MemoryStorage()
    BlockwiseStatisticalReductions.store!(mem, UInt64(1), "test")
    @test haskey(mem, UInt64(1))
    
    BlockwiseStatisticalReductions.clear!(mem)
    @test !haskey(mem, UInt64(1))
    @test isempty(mem.cache)
    
    # Disk
    mktempdir() do dir
        disk = DiskStorage(dir; format=:serialization)
        BlockwiseStatisticalReductions.store!(disk, UInt64(1), "test")
        
        BlockwiseStatisticalReductions.clear!(disk)
        @test isempty(disk.cache)
        @test isempty(readdir(dir))
    end
end

@testtestset "DiskStorage roundtrip with complex data" begin
    mktempdir() do dir
        storage = DiskStorage(dir; format=:serialization)
        
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
        @test retrieved[:matrix] == data[:matrix]
        @test retrieved[:vector] == data[:vector]
        @test retrieved[:nested][:a] == 1
    end
end
