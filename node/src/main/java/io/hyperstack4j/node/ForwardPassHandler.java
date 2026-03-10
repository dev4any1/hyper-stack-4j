package io.hyperstack4j.node;

/**
 * Executes the transformer forward pass for this node's assigned layers.
 *
 * Implementations: CyclicForwardPassHandler — deterministic fake, used in tests
 * + integration tests GpuForwardPassHandler — real JCuda/JCublas implementation
 * (GPU required)
 *
 * Thread-safe — may be called concurrently for different requests in a batch.
 */
public interface ForwardPassHandler {

	/**
	 * Execute this node's forward pass.
	 *
	 * @param request input (token IDs for first node, activations for others)
	 * @param context this node's shard assignment and model metadata
	 * @return ForwardResult with activations (intermediate) or logits (last node)
	 */
	ForwardResult forward(ForwardRequest request, ShardContext context);

	/** Whether this handler is ready to serve (shard loaded, GPU initialized). */
	boolean isReady();
}
