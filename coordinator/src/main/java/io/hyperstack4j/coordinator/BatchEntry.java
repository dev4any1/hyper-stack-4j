package io.hyperstack4j.coordinator;

/**
 * A request + its token-streaming callback, grouped for batch dispatch.
 *
 * Passed to GenerationLoop.generateBatch() — one entry per request in the batch.
 * The consumer receives tokens as they are generated, exactly as in single-request
 * generation. Batching is transparent to the streaming layer.
 */
public record BatchEntry(InferenceRequest request, TokenConsumer consumer) {

    public BatchEntry {
        if (request  == null) throw new IllegalArgumentException("request must not be null");
        if (consumer == null) throw new IllegalArgumentException("consumer must not be null");
    }
}