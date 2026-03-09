package io.hyperstack4j.coordinator;

/**
 * Callback interface for real-time streaming token delivery.
 *
 * Called by GenerationLoop once per generated token, and also notified
 * about the prefill phase so UIs can show a progress indicator.
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

    /**
     * Called once before the prefill loop starts.
     * Default no-op — override to display a "thinking…" indicator.
     *
     * @param promptLen total number of prompt tokens (prefill steps = promptLen - 1)
     */
    default void onPrefillStart(int promptLen) {}

    /**
     * Called once after the prefill loop finishes, immediately before decode begins.
     * Default no-op — override to clear the "thinking…" indicator.
     */
    default void onPrefillComplete() {}

    /** No-op consumer — useful for non-streaming (batch) generation. */
    static TokenConsumer discard() {
        return (piece, tokenId, position) -> {};
    }

    /** Collects pieces into a StringBuilder — useful for testing. */
    static TokenConsumer collecting(StringBuilder sb) {
        return (piece, tokenId, position) -> sb.append(piece);
    }
}