package io.hyperstack4j.node;

import io.hyperstack4j.registry.ShardAssignment;

/**
 * Runtime context for this node's shard assignment. Derived from the ShardMap
 * computed by the registry.
 *
 * Tells the ForwardPassHandler: - which layers to compute - whether to run
 * embedding lookup first - whether to run output projection last
 */
public record ShardContext(String nodeId, int startLayer, int endLayer, boolean hasEmbeddings,
		boolean hasOutputProjection, int vocabSize, int hiddenDim, int numHeads) {

	public ShardContext {
		if (startLayer < 0)
			throw new IllegalArgumentException("startLayer must be >= 0");
		if (endLayer <= startLayer)
			throw new IllegalArgumentException("endLayer must be > startLayer");
		if (vocabSize < 1)
			throw new IllegalArgumentException("vocabSize must be >= 1");
		if (hiddenDim < 1)
			throw new IllegalArgumentException("hiddenDim must be >= 1");
		if (numHeads < 1)
			throw new IllegalArgumentException("numHeads must be >= 1");
	}

	/** Number of transformer layers this node owns. */
	public int layerCount() {
		return endLayer - startLayer;
	}

	/** Build from a ShardAssignment + model metadata. */
	public static ShardContext from(ShardAssignment assignment, int vocabSize, int hiddenDim, int numHeads) {
		return new ShardContext(assignment.nodeId(), assignment.startLayer(), assignment.endLayer(),
				assignment.hasEmbeddings(), assignment.hasOutputProjection(), vocabSize, hiddenDim, numHeads);
	}
}
