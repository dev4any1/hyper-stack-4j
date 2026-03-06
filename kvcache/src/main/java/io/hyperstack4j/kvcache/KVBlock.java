package io.hyperstack4j.kvcache;

import java.time.Instant;

/**
 * A single KV cache block — attention keys + values for one transformer layer
 * of one request sequence.
 *
 * Stored as raw bytes (fp16 or fp32 depending on model config).
 * Keys and values are stored interleaved in a single byte array:
 *   [key_head_0 ... key_head_N | val_head_0 ... val_head_N]
 *
 * Size formula (fp16):
 *   2 x heads x head_dim x seq_len x 2 bytes = sizeBytes
 *
 * For LLaMA3-8B at 1000 tokens: ~512KB per layer per request
 */
public record KVBlock(
        KVKey   key,
        byte[]  data,          // serialized keys + values
        int     sequenceLen,   // number of tokens cached
        int     layerIndex,
        Instant createdAt,
        Instant lastAccessedAt
) {

    public KVBlock {
        if (key == null)   throw new IllegalArgumentException("key must not be null");
        if (data == null || data.length == 0)
            throw new IllegalArgumentException("data must not be empty");
        if (sequenceLen < 1)
            throw new IllegalArgumentException("sequenceLen must be >= 1");
    }

    /** Size of this block in bytes. */
    public int sizeBytes() {
        return data.length;
    }

    /** Return a copy with updated lastAccessedAt. */
    public KVBlock accessed() {
        return new KVBlock(key, data, sequenceLen, layerIndex, createdAt, Instant.now());
    }
}
