package io.hyperstack4j.coordinator;

import java.time.Instant;
import java.util.List;
import java.util.UUID;

import io.hyperstack4j.sampler.SamplingParams;
import io.hyperstack4j.tokenizer.ChatMessage;

/**
 * Immutable value object representing a single inference request.
 *
 * Created by the REST/gRPC layer and submitted to the RequestScheduler.
 * Carries everything the GenerationLoop needs: messages, sampling config,
 * model ID, and client-facing metadata.
 */
public record InferenceRequest(
        String          requestId,
        String          modelId,
        List<ChatMessage> messages,
        SamplingParams  samplingParams,
        RequestPriority priority,
        Instant         receivedAt
) implements Comparable<InferenceRequest> {

    public InferenceRequest {
        if (requestId == null || requestId.isBlank())
            throw new IllegalArgumentException("requestId must not be blank");
        if (modelId == null || modelId.isBlank())
            throw new IllegalArgumentException("modelId must not be blank");
        if (messages == null || messages.isEmpty())
            throw new IllegalArgumentException("messages must not be empty");
        if (samplingParams == null)
            throw new IllegalArgumentException("samplingParams must not be null");
        if (priority == null)
            throw new IllegalArgumentException("priority must not be null");

        messages = List.copyOf(messages); // defensive copy
    }

    /** Factory — generates a random requestId. */
    public static InferenceRequest of(String modelId, List<ChatMessage> messages,
                                      SamplingParams params, RequestPriority priority) {
        return new InferenceRequest(
                UUID.randomUUID().toString(),
                modelId,
                messages,
                params,
                priority,
                Instant.now()
        );
    }

    /** Higher priority = lower compareTo value = scheduled first. */
    @Override
    public int compareTo(InferenceRequest other) {
        int cmp = Integer.compare(other.priority().weight(), this.priority().weight());
        if (cmp != 0) return cmp;
        // FIFO within same priority
        return this.receivedAt().compareTo(other.receivedAt());
    }
}
