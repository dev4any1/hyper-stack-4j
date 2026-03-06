package io.hyperstack4j.coordinator;

/**
 * Callback interface for real-time streaming token delivery.
 *
 * Called by GenerationLoop once per generated token.
 * Implementations write the piece to a gRPC stream, SSE endpoint, etc.
 *
 * Must be non-blocking — any slow I/O should be handled asynchronously
 * by the implementation. Blocking here stalls the generation loop.
 */
@FunctionalInterface
public interface TokenConsumer {

    /**
     * Called when a new token piece is ready.
     *
     * @param piece     decoded text for this token (may be empty for special tokens)
     * @param tokenId   raw token ID
     * @param position  0-based position in the generated sequence
     */
    void onToken(String piece, int tokenId, int position);

    /** No-op consumer — useful for non-streaming (batch) generation. */
    static TokenConsumer discard() {
        return (piece, tokenId, position) -> {};
    }

    /** Collects pieces into a StringBuilder — useful for testing. */
    static TokenConsumer collecting(StringBuilder sb) {
        return (piece, tokenId, position) -> sb.append(piece);
    }
}
