package io.hyperstack4j.sampler;

/**
 * A single step in the sampling pipeline. Stateless and thread-safe — takes
 * logits in, returns modified logits. Steps are chained: temperature → topK →
 * topP → softmax → penalty → sample
 */
@FunctionalInterface
public interface SamplingStep {

	/**
	 * Apply this step to the logits array.
	 *
	 * @param logits          raw or partially processed logit scores (modified in
	 *                        place)
	 * @param params          sampling configuration for this request
	 * @param generatedTokens tokens already generated in this sequence (for
	 *                        repetition penalty)
	 * @return the same logits array, modified in place
	 */
	float[] apply(float[] logits, SamplingParams params, int[] generatedTokens);
}
