package io.hyperstack4j.coordinator;

import java.time.Duration;
import java.time.Instant;
import java.util.List;

/**
 * Completed inference result for a single request.
 *
 * For streaming use cases, tokens are delivered incrementally via TokenConsumer.
 * This record is the final summary returned when generation completes.
 */
public record GenerationResult(
        String       requestId,
        String       text,           // full decoded output
        List<Integer> tokenIds,      // all generated token IDs
        int          promptTokens,   // input token count
        int          generatedTokens,
        StopReason   stopReason,
        Instant      completedAt,
        Duration     latency
) {

    public GenerationResult {
        tokenIds = List.copyOf(tokenIds);
    }

    public enum StopReason {
        EOS_TOKEN,       // model produced end-of-sequence token
        MAX_TOKENS,      // hit samplingParams.maxTokens() limit
        STOP_TOKEN,      // hit a configured stop token ID
        ERROR            // upstream failure during generation
    }

    /** Tokens per second throughput. */
    public double tokensPerSecond() {
        double seconds = latency.toMillis() / 1000.0;
        return seconds > 0 ? generatedTokens / seconds : 0.0;
    }
}
