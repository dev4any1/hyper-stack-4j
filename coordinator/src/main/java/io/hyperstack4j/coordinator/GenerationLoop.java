package io.hyperstack4j.coordinator;

import io.hyperstack4j.kvcache.KVCacheManager;
import io.hyperstack4j.sampler.Sampler;
import io.hyperstack4j.tokenizer.ChatTemplateFormatter;
import io.hyperstack4j.tokenizer.Tokenizer;
import io.hyperstack4j.node.InferencePipeline;
import java.time.Duration;
import java.time.Instant;
import java.util.ArrayList;
import java.util.List;
import java.util.logging.Logger;

/**
 * Core autoregressive generation loop.
 *
 * Implements the 8-step loop from the architecture doc:
 *   1. encode prompt (chatTemplate + tokenizer)
 *   2. check prefix cache
 *   3. forward pass (full prefill or incremental from cache hit)
 *   4. sample next token
 *   5. check EOS / stop tokens
 *   6. decode token piece
 *   7. stream piece to client via TokenConsumer
 *   8. repeat
 *
 * Stateless — one shared instance, called per request on a Virtual Thread.
 * Each call is independent; all state lives on the stack.
 */
public final class GenerationLoop {

    private static final Logger log = Logger.getLogger(GenerationLoop.class.getName());

    private final Tokenizer          tokenizer;
    private final Sampler            sampler;
    private final InferencePipeline  pipeline;
    private final KVCacheManager     kvCache;

    public GenerationLoop(Tokenizer tokenizer, Sampler sampler,
                          InferencePipeline pipeline, KVCacheManager kvCache) {
        this.tokenizer = tokenizer;
        this.sampler   = sampler;
        this.pipeline  = pipeline;
        this.kvCache   = kvCache;
    }

    /**
     * Run generation for a single request.
     *
     * @param request  the inference request
     * @param consumer receives each token piece as it is generated
     * @return final GenerationResult with full text + stats
     */
    public GenerationResult generate(InferenceRequest request, TokenConsumer consumer) {
        Instant start = Instant.now();
        String requestId = request.requestId();

        // ── Step 1: Encode prompt ─────────────────────────────────────────────
        ChatTemplateFormatter formatter = ChatTemplateFormatter.forModelType(
                request.modelId().contains("llama3")  ? "llama3"  :
                request.modelId().contains("mistral") ? "mistral" :
                request.modelId().contains("gemma")   ? "gemma"   : "chatml"
        );
        String prompt    = formatter.format(request.messages());
        int[]  promptIds = tokenizer.encode(prompt);

        // ── Step 2: Check prefix cache ────────────────────────────────────────
        var prefixMatch = kvCache.findLongestPrefix(promptIds);
        int startPos = prefixMatch.isHit() ? prefixMatch.matchedTokens() : 0;

        if (prefixMatch.isHit()) {
            log.fine("Prefix cache hit: skipping " + startPos + " tokens for " + requestId);
        }

        // Build working token array (prompt IDs only at first)
        int[]        allTokens     = promptIds.clone();
        List<Integer> generatedIds = new ArrayList<>();
        StringBuilder fullText     = new StringBuilder();
        GenerationResult.StopReason stopReason = GenerationResult.StopReason.MAX_TOKENS;

        // ── Steps 3–8: Autoregressive decode loop ─────────────────────────────
        int maxTokens = request.samplingParams().maxTokens();

        for (int step = 0; step < maxTokens; step++) {

            // Step 3: Forward pass
            float[] logits = pipeline.forward(requestId, allTokens, startPos + step);

            // Step 4: Sample next token
            int[] historyArr = generatedIds.stream()
                    .mapToInt(Integer::intValue).toArray();
            int nextToken = sampler.sample(logits, request.samplingParams(), historyArr);

            // Step 5: Check stop conditions
            if (nextToken == tokenizer.eosTokenId()) {
                stopReason = GenerationResult.StopReason.EOS_TOKEN;
                break;
            }
            if (sampler.isStopToken(nextToken, request.samplingParams())) {
                stopReason = GenerationResult.StopReason.STOP_TOKEN;
                break;
            }

            // Step 6: Decode token piece
            String piece = tokenizer.decodeToken(nextToken);

            // Step 7: Stream to client
            consumer.onToken(piece, nextToken, step);
            fullText.append(piece);
            generatedIds.add(nextToken);

            // Step 8: Extend token array for next iteration
            allTokens = appendToken(allTokens, nextToken);
        }

        // Cache the prompt prefix for future requests
        if (!prefixMatch.isHit() && promptIds.length > 0) {
            kvCache.cachePrefix(promptIds, promptIds.length, requestId + ":prefix");
        }

        // Cleanup KV blocks for this request
        kvCache.evict(requestId);

        return new GenerationResult(
                requestId,
                fullText.toString(),
                generatedIds,
                promptIds.length,
                generatedIds.size(),
                stopReason,
                Instant.now(),
                Duration.between(start, Instant.now())
        );
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    private int[] appendToken(int[] tokens, int newToken) {
        int[] next = new int[tokens.length + 1];
        System.arraycopy(tokens, 0, next, 0, tokens.length);
        next[tokens.length] = newToken;
        return next;
    }
}
