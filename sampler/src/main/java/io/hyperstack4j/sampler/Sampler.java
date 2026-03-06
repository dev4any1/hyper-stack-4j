package io.hyperstack4j.sampler;

/**
 * Main sampling pipeline.
 *
 * Chains all steps in order:
 *   temperature → topK → softmax → topP → repetitionPenalty → sample
 *
 * Stateless and thread-safe. One instance shared across all requests.
 *
 * Usage:
 *   Sampler sampler = Sampler.create();
 *   int nextToken = sampler.sample(logits, SamplingParams.defaults(), generatedSoFar);
 */
public final class Sampler {

    private final TemperatureStep    temperatureStep;
    private final TopKStep           topKStep;
    private final SoftmaxStep        softmaxStep;
    private final TopPStep           topPStep;
    private final RepetitionPenaltyStep repetitionPenaltyStep;
    private final SampleStep         sampleStep;

    private Sampler(
            TemperatureStep temperatureStep,
            TopKStep topKStep,
            SoftmaxStep softmaxStep,
            TopPStep topPStep,
            RepetitionPenaltyStep repetitionPenaltyStep,
            SampleStep sampleStep
    ) {
        this.temperatureStep       = temperatureStep;
        this.topKStep              = topKStep;
        this.softmaxStep           = softmaxStep;
        this.topPStep              = topPStep;
        this.repetitionPenaltyStep = repetitionPenaltyStep;
        this.sampleStep            = sampleStep;
    }

    /**
     * Create a default Sampler with all standard pipeline steps.
     */
    public static Sampler create() {
        return new Sampler(
                TemperatureStep.INSTANCE,
                TopKStep.INSTANCE,
                SoftmaxStep.INSTANCE,
                TopPStep.INSTANCE,
                RepetitionPenaltyStep.INSTANCE,
                SampleStep.INSTANCE
        );
    }

    /**
     * Run the full sampling pipeline and return the next token ID.
     *
     * @param rawLogits       float[vocabSize] from the last inference node — WILL BE MUTATED
     * @param params          sampling configuration for this request
     * @param generatedTokens token IDs generated so far in this sequence (for repetition penalty)
     * @return next token ID
     */
    public int sample(float[] rawLogits, SamplingParams params, int[] generatedTokens) {
        if (rawLogits == null || rawLogits.length == 0)
            throw new IllegalArgumentException("logits must not be null or empty");
        if (params == null)
            throw new IllegalArgumentException("params must not be null");

        float[] logits = rawLogits.clone(); // defensive copy — don't mutate caller's array

        // Pipeline
        logits = temperatureStep.apply(logits, params, generatedTokens);
        logits = topKStep.apply(logits, params, generatedTokens);
        logits = softmaxStep.apply(logits, params, generatedTokens);
        logits = topPStep.apply(logits, params, generatedTokens);
        logits = repetitionPenaltyStep.apply(logits, params, generatedTokens);

        return sampleStep.sample(logits, params);
    }

    /**
     * Convenience overload — no previous tokens (start of sequence).
     */
    public int sample(float[] rawLogits, SamplingParams params) {
        return sample(rawLogits, params, new int[0]);
    }

    /**
     * Check whether a token is a stop condition.
     */
    public boolean isStopToken(int tokenId, SamplingParams params) {
        for (int stop : params.stopTokenIds()) {
            if (stop == tokenId) return true;
        }
        return false;
    }
}
