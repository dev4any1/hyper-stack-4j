package io.hyperstack4j.kvcache;

/**
 * Composite cache key — requestId + layerIndex.
 *
 * Each inference node caches KV pairs only for its own assigned layers. The
 * layerIndex identifies which transformer layer this block belongs to.
 */
public record KVKey(String requestId, int layerIndex) {

	public KVKey {
		if (requestId == null || requestId.isBlank())
			throw new IllegalArgumentException("requestId must not be blank");
		if (layerIndex < 0)
			throw new IllegalArgumentException("layerIndex must be >= 0");
	}

	@Override
	public String toString() {
		return requestId + "@layer" + layerIndex;
	}
}
