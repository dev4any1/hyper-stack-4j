package io.hyperstack4j.sampler;

/**
 * Step 1: Temperature scaling.
 *
 * Divides each logit by the temperature value.
 *   temperature → 0.0  : approaches one-hot (fully deterministic)
 *   temperature = 1.0  : no change
 *   temperature → 2.0  : flattens distribution (more random)
 *
 * Skipped if greedy=true (temperature is irrelevant for argmax).
 */
public final class TemperatureStep implements SamplingStep {

    public static final TemperatureStep INSTANCE = new TemperatureStep();

    private TemperatureStep() {}

    @Override
    public float[] apply(float[] logits, SamplingParams params, int[] generatedTokens) {
        if (params.greedy()) return logits;

        float temperature = params.temperature();
        // Treat near-zero as greedy to avoid division instability
        if (temperature < 1e-6f) return logits;

        for (int i = 0; i < logits.length; i++) {
            logits[i] /= temperature;
        }
        return logits;
    }
}
