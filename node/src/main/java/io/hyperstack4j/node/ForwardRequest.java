package io.hyperstack4j.node;

/**
 * Input to a single node's forward pass computation.
 *
 * For the first node (hasEmbeddings=true): activations is null, tokenIds is
 * set. For subsequent nodes: activations carries the float[] from the previous
 * node.
 */
public record ForwardRequest(String requestId, int[] tokenIds, // non-null for first node only
		float[] activations, // non-null for subsequent nodes
		int startPosition // KV cache position (for incremental decode)
) {

	/** First node in the pipeline — takes raw token IDs. */
	public static ForwardRequest withTokens(String requestId, int[] tokenIds, int startPosition) {
		if (tokenIds == null || tokenIds.length == 0)
			throw new IllegalArgumentException("tokenIds must not be empty");
		return new ForwardRequest(requestId, tokenIds, null, startPosition);
	}

	/** Subsequent nodes — take activations from previous node. */
	public static ForwardRequest withActivations(String requestId, float[] activations, int startPosition) {
		if (activations == null || activations.length == 0)
			throw new IllegalArgumentException("activations must not be empty");
		return new ForwardRequest(requestId, null, activations, startPosition);
	}

	public boolean isFirstNode() {
		return tokenIds != null;
	}
}
