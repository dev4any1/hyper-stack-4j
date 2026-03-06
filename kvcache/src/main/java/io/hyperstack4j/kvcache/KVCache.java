package io.hyperstack4j.kvcache;

import java.util.Optional;

/**
 * Single-tier KV cache contract.
 *
 * Implementations:
 *   CpuKVCache   — Caffeine, bounded by JVM heap (-Xmx)
 *   GpuKVCache   — JCuda CudaBuffer, bounded by GPU VRAM
 *
 * All methods are thread-safe.
 */
public interface KVCache {

    /**
     * Store a KV block. Evicts entries if capacity is exceeded.
     */
    void put(KVKey key, KVBlock block);

    /**
     * Retrieve a KV block, or empty if not present.
     */
    Optional<KVBlock> get(KVKey key);

    /**
     * Evict all blocks for a given requestId (called when request completes).
     */
    void evict(String requestId);

    /**
     * Whether this cache contains a block for the given key.
     */
    boolean contains(KVKey key);

    /**
     * Current number of blocks stored.
     */
    long size();

    /**
     * Approximate memory used by cached data in bytes.
     */
    long estimatedSizeBytes();

    /**
     * Human-readable tier name for logging e.g. "cpu", "gpu".
     */
    String tierName();
}
