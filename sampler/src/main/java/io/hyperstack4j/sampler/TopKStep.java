package io.hyperstack4j.sampler;

import java.util.Arrays;

/**
 * Step 2: Top-K filtering.
 *
 * Keeps only the K tokens with the highest logit scores. All others are set to
 * -Infinity so softmax will assign them probability 0.
 *
 * Skipped if params.topK() == 0 (disabled) or greedy=true.
 */
public final class TopKStep implements SamplingStep {

	public static final TopKStep INSTANCE = new TopKStep();

	private static final float NEG_INF = Float.NEGATIVE_INFINITY;

	private TopKStep() {
	}

	@Override
	public float[] apply(float[] logits, SamplingParams params, int[] generatedTokens) {
		if (params.greedy())
			return logits;

		int k = params.topK();
		if (k <= 0 || k >= logits.length)
			return logits;

		// Find the k-th largest value (pivot)
		float[] sorted = logits.clone();
		Arrays.sort(sorted);
		float threshold = sorted[sorted.length - k]; // k-th from the top

		// Zero out everything below threshold
		for (int i = 0; i < logits.length; i++) {
			if (logits[i] < threshold) {
				logits[i] = NEG_INF;
			}
		}
		return logits;
	}
}
