package io.hyperstack4j.sampler;

import java.util.concurrent.ThreadLocalRandom;

/**
 * Step 6: Final token selection.
 *
 * Two modes:
 *   greedy  — argmax: always picks the highest probability token
 *   sample  — weighted random draw over the probability distribution
 *
 * Returns the selected token ID via a single-element array.
 * The logits array is treated as probabilities at this stage (post-softmax).
 *
 * Thread-safe: uses ThreadLocalRandom, no shared mutable state.
 */
public final class SampleStep {

    public static final SampleStep INSTANCE = new SampleStep();

    private SampleStep() {}

    /**
     * Select the next token from the probability distribution.
     *
     * @param probs           probability distribution (output of softmax + filters)
     * @param params          sampling configuration
     * @return selected token ID
     */
    public int sample(float[] probs, SamplingParams params) {
        return params.greedy() ? greedy(probs) : weightedSample(probs);
    }

    // ── Greedy: argmax ────────────────────────────────────────────────────────

    private int greedy(float[] probs) {
        int best = 0;
        for (int i = 1; i < probs.length; i++) {
            if (probs[i] > probs[best]) best = i;
        }
        return best;
    }

    // ── Weighted random draw ──────────────────────────────────────────────────

    private int weightedSample(float[] probs) {
        double r = ThreadLocalRandom.current().nextDouble();
        double cumulative = 0.0;
        for (int i = 0; i < probs.length; i++) {
            cumulative += probs[i];
            if (r < cumulative) return i;
        }
        // Fallback — floating point rounding edge case: return last non-zero token
        for (int i = probs.length - 1; i >= 0; i--) {
            if (probs[i] > 0.0f) return i;
        }
        return 0;
    }
}
