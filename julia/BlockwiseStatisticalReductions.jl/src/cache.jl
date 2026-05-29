"""
    cache_key(node::AbstractPlanNode, input_shape)

Generate a unique cache key for a plan node and input configuration.
Uses operation semantics (not node.id) so identical operations share cache entries.
"""
cache_key(node::WindowNode, input_shape) = hash((:window, node.config.sizes, node.config.strides, node.config.padding, input_shape))
cache_key(node::ReductionNode{F}, input_shape) where F = hash((:reduction, F, node.config.sizes, node.config.strides, node.config.padding, input_shape))
cache_key(node::SufficientStatsNode{F,M}, input_shape) where {F,M} = hash((:sufstats, F, M, node.is_base, node.config.sizes, node.count_per_block, input_shape))
cache_key(node::TreeNode, input_shape) = hash((:tree, node.arity, input_shape))
cache_key(node::UserNode{F}, input_shape) where F = hash((:user, F, node.output_type, input_shape))

"""
    get_or_compute!(cache::PlanCache, key::UInt64, f)

Get cached value or compute and store it.
"""
function get_or_compute!(cache::PlanCache, key::UInt64, f)
    result = retrieve(cache.storage, key)
    
    if result !== nothing
        cache.hits += 1
        return result
    end
    
    cache.misses += 1
    result = f()
    store!(cache.storage, key, result)
    return result
end

"""
    cache_stats(cache::PlanCache)

Return cache hit/miss statistics.
"""
cache_stats(cache::PlanCache) = (hits=cache.hits, misses=cache.misses, hit_rate=cache.hits / max(1, cache.hits + cache.misses))

"""
    invalidate!(cache::PlanCache, key::UInt64)

Remove a specific entry from cache.
"""
function invalidate!(cache::PlanCache{MemoryStorage}, key::UInt64)
    delete!(cache.storage.cache, key)
    return cache
end

function invalidate!(cache::PlanCache{DiskStorage}, key::UInt64)
    if haskey(cache.storage.cache, key)
        filename = cache.storage.cache[key]
        isfile(filename) && rm(filename)
        delete!(cache.storage.cache, key)
    end
    return cache
end

"""
    invalidate_all!(cache::PlanCache)

Clear all cached entries.
"""
invalidate_all!(cache::PlanCache) = (clear!(cache.storage); cache)
