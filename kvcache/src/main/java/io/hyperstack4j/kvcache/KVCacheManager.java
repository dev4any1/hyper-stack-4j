package io.hyperstack4j.kvcache;

import java.time.Instant;
import java.util.Optional;
import java.util.logging.Logger;

/**
 * Unified KV cache facade — orchestrates GPU and CPU tiers.
 *
 * Write policy: write-through to both tiers.
 * Read policy:  GPU first, fall back to CPU, promote on CPU hit.
 * Eviction:     per-tier (Caffeine on CPU, LRU slab on GPU).
 * Cleanup:      evict(requestId) removes from both tiers on request completion.
 *
 * Thread-safe — each tier implementation is independently thread-safe.
 */
public final class KVCacheManager {

    private static final Logger log = Logger.getLogger(KVCacheManager.class.getName());

    private final GpuKVCache    gpuCache;
    private final CpuKVCache    cpuCache;
    private final PrefixCache   prefixCache;

    public KVCacheManager(GpuKVCache gpuCache, CpuKVCache cpuCache) {
        if (gpuCache == null) throw new IllegalArgumentException("gpuCache must not be null");
        if (cpuCache == null) throw new IllegalArgumentException("cpuCache must not be null");
        this.gpuCache    = gpuCache;
        this.cpuCache    = cpuCache;
        this.prefixCache = new PrefixCache();
    }

    /**
     * Store a KV block. Written to GPU tier first, then CPU tier.
     */
    public void put(KVKey key, KVBlock block) {
        gpuCache.put(key, block);
        cpuCache.put(key, block);
    }

    /**
     * Retrieve a KV block. Checks GPU first, falls back to CPU.
     * On CPU hit, promotes block back to GPU tier.
     */
    public Optional<KVBlock> get(KVKey key) {
        // GPU hit — fast path
        Optional<KVBlock> gpuHit = gpuCache.get(key);
        if (gpuHit.isPresent()) return gpuHit;

        // CPU hit — promote back to GPU
        Optional<KVBlock> cpuHit = cpuCache.get(key);
        if (cpuHit.isPresent()) {
            log.fine("KV cache CPU→GPU promotion: " + key);
            gpuCache.put(key, cpuHit.get());
            return cpuHit;
        }

        return Optional.empty();
    }

    /**
     * Evict all blocks for a completed request from both tiers.
     */
    public void evict(String requestId) {
        gpuCache.evict(requestId);
        cpuCache.evict(requestId);
    }

    /**
     * Store a prefix in the prefix trie.
     *
     * @param tokens     full token sequence
     * @param prefixLen  how many tokens to register as cached prefix
     * @param cacheKey   reference key for KV lookup
     */
    public void cachePrefix(int[] tokens, int prefixLen, String cacheKey) {
        prefixCache.cachePrefix(tokens, prefixLen, cacheKey);
    }

    /**
     * Find the longest matching cached prefix for the given token sequence.
     */
    public PrefixCache.PrefixMatch findLongestPrefix(int[] tokens) {
        return prefixCache.findLongestPrefix(tokens);
    }

    // ── Stats ─────────────────────────────────────────────────────────────────

    public long gpuBlockCount()       { return gpuCache.size(); }
    public long cpuBlockCount()       { return cpuCache.size(); }
    public long gpuBytesUsed()        { return gpuCache.estimatedSizeBytes(); }
    public long cpuBytesUsed()        { return cpuCache.estimatedSizeBytes(); }
    public long gpuVramBudgetBytes()  { return gpuCache.vramBudgetBytes(); }
}
