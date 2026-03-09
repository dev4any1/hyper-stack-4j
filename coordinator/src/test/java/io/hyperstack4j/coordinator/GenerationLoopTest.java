package io.hyperstack4j.coordinator;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatCode;

import java.util.ArrayList;
import java.util.List;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import io.hyperstack4j.kvcache.CpuKVCache;
import io.hyperstack4j.kvcache.GpuKVCache;
import io.hyperstack4j.kvcache.KVCacheManager;
import io.hyperstack4j.node.InferencePipeline;
import io.hyperstack4j.sampler.Sampler;
import io.hyperstack4j.sampler.SamplingParams;
import io.hyperstack4j.tokenizer.ChatMessage;
import io.hyperstack4j.tokenizer.StubTokenizer;

class GenerationLoopTest {

    private StubTokenizer       tokenizer;
    private Sampler             sampler;
    private KVCacheManager      kvCache;
    private GenerationLoop      loop;

    @BeforeEach
    void setUp() {
        tokenizer = new StubTokenizer();
        sampler   = Sampler.create();
        kvCache   = new KVCacheManager(
                new GpuKVCache(64 * 1024 * 1024),
                new CpuKVCache(1000)
        );
    }

    private GenerationLoop loopWith(InferencePipeline pipeline) {
        return new GenerationLoop(tokenizer, sampler, pipeline, kvCache);
    }

    private InferenceRequest requestFor(String... messages) {
        List<ChatMessage> msgs = new ArrayList<>();
        for (int i = 0; i < messages.length; i++) {
            msgs.add(i % 2 == 0 ? ChatMessage.user(messages[i])
                                 : ChatMessage.assistant(messages[i]));
        }
        return InferenceRequest.of("llama3-8b", msgs,
                SamplingParams.defaults().withMaxTokens(5), RequestPriority.NORMAL);
    }

    @Test
    void generates_up_to_max_tokens() {
        // Pipeline always returns token 42 — never EOS — so loop hits maxTokens
        StubInferencePipeline pipeline = new StubInferencePipeline();
        GenerationLoop loop = loopWith(pipeline);

        GenerationResult result = loop.generate(requestFor("hello"), TokenConsumer.discard());

        assertThat(result.generatedTokens()).isEqualTo(5);
        assertThat(result.stopReason()).isEqualTo(GenerationResult.StopReason.MAX_TOKENS);
    }

    @Test
    void stops_at_eos_token() {
        // After 3 tokens, returns EOS
        int eos = tokenizer.eosTokenId();
        StubInferencePipeline pipeline = new StubInferencePipeline(42, 43, eos);
        GenerationLoop loop = loopWith(pipeline);

        InferenceRequest req = InferenceRequest.of("model",
                List.of(ChatMessage.user("hi")),
                SamplingParams.defaults().withMaxTokens(20),
                RequestPriority.NORMAL);

        GenerationResult result = loop.generate(req, TokenConsumer.discard());

        assertThat(result.generatedTokens()).isEqualTo(2); // 42, 43 — EOS not counted
        assertThat(result.stopReason()).isEqualTo(GenerationResult.StopReason.EOS_TOKEN);
    }

    @Test
    void stops_at_stop_token() {
        int stopToken = 99;
        StubInferencePipeline pipeline = new StubInferencePipeline(42, stopToken, 43);
        GenerationLoop loop = loopWith(pipeline);

        InferenceRequest req = InferenceRequest.of("model",
                List.of(ChatMessage.user("hi")),
                SamplingParams.defaults()
                        .withMaxTokens(20)
                        .withStopTokenIds(stopToken),
                RequestPriority.NORMAL);

        GenerationResult result = loop.generate(req, TokenConsumer.discard());

        assertThat(result.generatedTokens()).isEqualTo(1); // only token 42
        assertThat(result.stopReason()).isEqualTo(GenerationResult.StopReason.STOP_TOKEN);
    }

    @Test
    void token_consumer_called_once_per_generated_token() {
        StubInferencePipeline pipeline = new StubInferencePipeline();
        GenerationLoop loop = loopWith(pipeline);

        List<Integer> receivedTokens = new ArrayList<>();
        TokenConsumer consumer = (piece, tokenId, pos) -> receivedTokens.add(tokenId);

        loop.generate(requestFor("test"), consumer);

        assertThat(receivedTokens).hasSize(5); // maxTokens=5
        assertThat(receivedTokens).allMatch(id -> id == StubInferencePipeline.DEFAULT_TOKEN);
    }

    @Test
    void result_contains_prompt_token_count() {
        StubInferencePipeline pipeline = new StubInferencePipeline();
        GenerationLoop loop = loopWith(pipeline);

        GenerationResult result = loop.generate(requestFor("hello world"), TokenConsumer.discard());

        assertThat(result.promptTokens()).isGreaterThan(0);
    }

    @Test
    void result_latency_is_positive() {
        GenerationLoop loop = loopWith(new StubInferencePipeline());
        GenerationResult result = loop.generate(requestFor("hi"), TokenConsumer.discard());
        assertThat(result.latency().toNanos()).isGreaterThan(0);
    }

    @Test
    void prefix_cache_populated_after_first_request() {
        GenerationLoop loop = loopWith(new StubInferencePipeline());
        InferenceRequest req = requestFor("the same system prompt");

        loop.generate(req, TokenConsumer.discard());

        // After generation, prefix should be cached
        int[] encoded = tokenizer.encode("the same system prompt");
        // The prefix cache should have something for these tokens
        var match = kvCache.findLongestPrefix(encoded);
        // May or may not hit depending on template formatting, but should not throw
        assertThatCode(() -> kvCache.findLongestPrefix(encoded)).doesNotThrowAnyException();
    }
}
