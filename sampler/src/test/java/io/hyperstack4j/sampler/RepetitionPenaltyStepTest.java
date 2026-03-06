package io.hyperstack4j.sampler;

import org.junit.jupiter.api.Test;
import static org.assertj.core.api.Assertions.*;

class RepetitionPenaltyStepTest {

    private final RepetitionPenaltyStep step = RepetitionPenaltyStep.INSTANCE;

    @Test
    void reduces_probability_of_seen_tokens() {
        float[] probs = {0.25f, 0.25f, 0.25f, 0.25f};
        int[] generated = {0, 1}; // tokens 0 and 1 already seen
        SamplingParams params = SamplingParams.defaults().withRepetitionPenalty(1.5f);

        step.apply(probs, params, generated);

        // Tokens 0 and 1 should have lower relative probability
        assertThat(probs[0]).isLessThan(probs[2]);
        assertThat(probs[1]).isLessThan(probs[3]);
    }

    @Test
    void skips_when_penalty_is_one() {
        float[] probs = {0.25f, 0.25f, 0.25f, 0.25f};
        float[] original = probs.clone();
        SamplingParams params = SamplingParams.defaults().withRepetitionPenalty(1.0f);

        step.apply(probs, params, new int[]{0, 1, 2});

        assertThat(probs).containsExactly(original);
    }

    @Test
    void skips_when_no_generated_tokens() {
        float[] probs = {0.25f, 0.25f, 0.25f, 0.25f};
        float[] original = probs.clone();
        SamplingParams params = SamplingParams.defaults().withRepetitionPenalty(1.3f);

        step.apply(probs, params, new int[0]);

        assertThat(probs).containsExactly(original);
    }

    @Test
    void probabilities_still_sum_to_one_after_penalty() {
        float[] probs = {0.4f, 0.3f, 0.2f, 0.1f};
        SamplingParams params = SamplingParams.defaults().withRepetitionPenalty(1.3f);

        step.apply(probs, params, new int[]{0, 1});

        float sum = 0.0f;
        for (float v : probs) sum += v;
        assertThat(sum).isCloseTo(1.0f, within(1e-5f));
    }
}
