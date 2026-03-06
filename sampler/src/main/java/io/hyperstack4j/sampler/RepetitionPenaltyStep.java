package io.hyperstack4j.sampler;

import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;

/**
 * Step 5: Repetition penalty.
 *
 * Reduces the probability of tokens that have already appeared in the
 * generated sequence. Applied after softmax, before final sampling.
 *
 * For each token already seen:
 *   prob[token] /= repetitionPenalty
 *
 * penalty=1.0 → no effect (disabled)
 * penalty=1.1 → mild suppression (default)
 * penalty=1.5 → strong suppression
 *
 * Skipped if penalty == 1.0 or generatedTokens is empty.
 */
public final class RepetitionPenaltyStep implements SamplingStep {

    public static final RepetitionPenaltyStep INSTANCE = new RepetitionPenaltyStep();

    private RepetitionPenaltyStep() {}

    @Override
    public float[] apply(float[] logits, SamplingParams params, int[] generatedTokens) {
        float penalty = params.repetitionPenalty();
        if (penalty == 1.0f || generatedTokens == null || generatedTokens.length == 0) {
            return logits;
        }

        // Build a set of already-seen token IDs for O(1) lookup
        Set<Integer> seen = new HashSet<>(generatedTokens.length * 2);
        for (int token : generatedTokens) seen.add(token);

        for (int tokenId : seen) {
            if (tokenId >= 0 && tokenId < logits.length) {
                logits[tokenId] /= penalty;
            }
        }

        // Re-normalize after penalty adjustment
        float sum = 0.0f;
        for (float v : logits) {
            if (v > 0.0f) sum += v;
        }
        if (sum > 0.0f) {
            for (int i = 0; i < logits.length; i++) {
                if (logits[i] > 0.0f) logits[i] /= sum;
            }
        }

        return logits;
    }
}
