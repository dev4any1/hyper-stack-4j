package io.hyperstack4j.node;

/**
 * Output from a single node's forward pass.
 *
 * For intermediate nodes: activations carries the hidden state to the next node.
 * For the last node (hasOutputProjection=true): logits is set instead.
 */
public record ForwardResult(
        String  requestId,
        float[] activations,   // non-null for intermediate nodes
        float[] logits,        // non-null for last node only (float[vocabSize])
        long    computeNanos   // wall time for this node's computation
) {

    /** Intermediate node result. */
    public static ForwardResult activations(String requestId, float[] activations, long nanos) {
        return new ForwardResult(requestId, activations, null, nanos);
    }

    /** Last node result — carries final logits. */
    public static ForwardResult logits(String requestId, float[] logits, long nanos) {
        return new ForwardResult(requestId, null, logits, nanos);
    }

    public boolean isFinalNode() { return logits != null; }
}
