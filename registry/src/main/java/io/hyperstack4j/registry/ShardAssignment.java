package io.hyperstack4j.registry;

import java.io.Serializable;

/**
 * Assignment of a contiguous layer range to a specific node. Part of a ShardMap
 * — one ShardAssignment per node in the pipeline.
 *
 * Pipeline order is determined by startLayer ascending. The node with the
 * lowest startLayer holds the embedding table. The node with the highest
 * endLayer holds the output projection.
 */
public record ShardAssignment(String nodeId, String host, int grpcPort, int startLayer, // inclusive
		int endLayer, // exclusive
		boolean hasEmbeddings, // true for the first node in pipeline
		boolean hasOutputProjection // true for the last node in pipeline
) implements Serializable {

	public ShardAssignment {
		if (nodeId == null || nodeId.isBlank())
			throw new IllegalArgumentException("nodeId must not be blank");
		if (startLayer < 0)
			throw new IllegalArgumentException("startLayer must be >= 0");
		if (endLayer <= startLayer)
			throw new IllegalArgumentException("endLayer must be > startLayer");
	}

	/** Number of transformer layers assigned to this node. */
	public int layerCount() {
		return endLayer - startLayer;
	}

	/** gRPC target address for this node. */
	public String grpcTarget() {
		return host + ":" + grpcPort;
	}
}
