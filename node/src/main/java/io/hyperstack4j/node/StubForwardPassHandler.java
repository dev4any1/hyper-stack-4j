package io.hyperstack4j.node;

import java.util.concurrent.atomic.AtomicInteger;

/**
 * Deterministic ForwardPassHandler for tests and integration testing.
 *
 * Intermediate nodes: returns a fixed-pattern float[] activation of hiddenDim size.
 * Last node: rotates the winner token through IDs 3-9, which are pre-registered
 * stub response words in StubTokenizer ("the quick brown fox jumps over lazy").
 * This guarantees every generated token decodes to a visible word regardless of
 * what prompt was given, without depending on any dynamic vocabulary registration.
 *
 * No GPU, no model weights, no JCuda. Compiles and runs anywhere.
 */
public final class StubForwardPassHandler implements ForwardPassHandler {

    /** First and last stub word token ID — must match StubTokenizer.STUB_WORD_FIRST/LAST. */
    private static final int STUB_FIRST = 3;
    private static final int STUB_LAST  = 9;
    private static final int STUB_RANGE = STUB_LAST - STUB_FIRST + 1; // 7 tokens

    private final int           fixedWinner;   // -1 = rotate, >= 0 = fixed (for targeted tests)
    private final AtomicInteger callCount = new AtomicInteger(0);

    /** Default: rotate winner through IDs 3-9 for visible streaming output. */
    public StubForwardPassHandler() {
        this.fixedWinner = -1;
    }

    /** Targeted: always use a specific winner token (for tests that check exact IDs). */
    public StubForwardPassHandler(int winnerToken) {
        this.fixedWinner = winnerToken;
    }

    @Override
    public ForwardResult forward(ForwardRequest request, ShardContext context) {
        callCount.incrementAndGet();
        long start = System.nanoTime();

        if (context.hasOutputProjection()) {
            // Last node — return logits with all mass on the winner token
            float[] logits = new float[context.vocabSize()];
            // Base winner on startPosition so the same decode step always maps to the
            // same token regardless of how many other requests ran before — this is
            // what makes the FLOAT16/INT8 compression tests deterministic.
            int winner = (fixedWinner >= 0)
                    ? fixedWinner
                    : STUB_FIRST + (request.startPosition() % STUB_RANGE);
            logits[winner] = 100.0f;
            return ForwardResult.logits(request.requestId(), logits, System.nanoTime() - start);
        } else {
            // Intermediate node — return activations (deterministic pattern)
            float[] activations = new float[context.hiddenDim()];
            for (int i = 0; i < activations.length; i++) {
                activations[i] = 0.01f * (i % 100);
            }
            return ForwardResult.activations(request.requestId(), activations, System.nanoTime() - start);
        }
    }

    @Override
    public boolean isReady() { return true; }

    public int callCount() { return callCount.get(); }
}