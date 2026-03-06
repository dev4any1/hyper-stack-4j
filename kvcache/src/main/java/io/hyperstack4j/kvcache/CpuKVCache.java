package io.hyperstack4j.kvcache;

import com.github.benmanes.caffeine.cache.Cache;
import com.github.benmanes.caffeine.cache.Caffeine;

import java.util.Optional;
import java.util.concurrent.atomic.AtomicLong;

/**
 * CPU-tier KV cache backed by Caffeine (W-TinyLFU eviction).
 *
 * Bounded by maximum number of blocks (not bytes directly).
 * Use maxBlocks = estimatedHeapBudget / averageBlockSize to size appropriately.
 *
 * Caffeine handles eviction automatically — GC-aware, no off-heap complexity.
 * Thread-safe — Caffeine's internal striped locking handles concurrency.
 */
public final class CpuKVCache implements KVCache {

    private final Cache<KVKey, KVBlock> cache;
    private final AtomicLong            totalBytesStored = new AtomicLong(0);

    public CpuKVCache(long maxBlocks) {
        if (maxBlocks < 1) throw new IllegalArgumentException("maxBlocks must be >= 1");
        this.cache = Caffeine.newBuilder()
                .maximumSize(maxBlocks)
                .removalListener((key, block, cause) -> {
                    if (block instanceof KVBlock b) totalBytesStored.addAndGet(-b.sizeBytes());
                })
                .build();
    }

    @Override
    public void put(KVKey key, KVBlock block) {
        KVBlock existing = cache.getIfPresent(key);
        if (existing != null) totalBytesStored.addAndGet(-existing.sizeBytes());
        cache.put(key, block);
        totalBytesStored.addAndGet(block.sizeBytes());
    }

    @Override
    public Optional<KVBlock> get(KVKey key) {
        return Optional.ofNullable(cache.getIfPresent(key));
    }

    @Override
    public void evict(String requestId) {
        // Caffeine doesn't support prefix scans — collect keys then invalidate
        cache.asMap().keySet().stream()
                .filter(k -> k.requestId().equals(requestId))
                .forEach(cache::invalidate);
    }

    @Override
    public boolean contains(KVKey key) {
        return cache.getIfPresent(key) != null;
    }

    @Override public long   size()               { return cache.estimatedSize(); }
    @Override public long   estimatedSizeBytes() { return totalBytesStored.get(); }
    @Override public String tierName()           { return "cpu"; }
}
