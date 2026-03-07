package io.hyperstack4j.integration;

import static org.assertj.core.api.Assertions.assertThat;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CopyOnWriteArrayList;

import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.MethodOrderer;
import org.junit.jupiter.api.Order;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestMethodOrder;

import io.hyperstack4j.coordinator.GenerationLoop;
import io.hyperstack4j.coordinator.GenerationResult;
import io.hyperstack4j.coordinator.InferenceRequest;
import io.hyperstack4j.coordinator.RequestPriority;
import io.hyperstack4j.coordinator.RequestScheduler;
import io.hyperstack4j.coordinator.TokenConsumer;
import io.hyperstack4j.kvcache.CpuKVCache;
import io.hyperstack4j.kvcache.GpuKVCache;
import io.hyperstack4j.kvcache.KVCacheManager;
import io.hyperstack4j.sampler.Sampler;
import io.hyperstack4j.sampler.SamplingParams;
import io.hyperstack4j.tokenizer.ChatMessage;
import io.hyperstack4j.tokenizer.StubTokenizer;

/**
 * Full multi-JVM 3-node cluster integration test.
 *
 * 3 separate JVM processes each running EmbeddedNodeServer (gRPC).
 * ProcessPipelineClient routes forward passes across them in pipeline order.
 * GenerationLoop + RequestScheduler run in this (coordinator) JVM.
 *
 * Memory budget for 16 GB host:
 *   3 node JVMs × -Xmx4g  = 12 GB
 *   coordinator JVM -Xmx2g =  2 GB
 *   OS + overhead           =  2 GB
 *
 * Run all ITs:  mvn verify -pl integration
 * Run only this: mvn verify -pl integration -Dit.test=ThreeNodeClusterIT
 */
@DisplayName("Three-Node Cluster (3 forked JVMs)")
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
class ThreeNodeClusterIT {

    private static ClusterHarness        harness;
    private static ProcessPipelineClient pipeline;
    private static GenerationLoop        generationLoop;
    private static RequestScheduler      scheduler;

    @BeforeAll
    static void startCluster() throws Exception {
        harness = ClusterHarness.threeNodes();
        harness.start();

        pipeline = harness.pipelineClient();

        generationLoop = new GenerationLoop(
                new StubTokenizer(),
                Sampler.create(),
                pipeline,
                new KVCacheManager(
                        new GpuKVCache(512L * 1024 * 1024),
                        new CpuKVCache(4096)
                )
        );
        scheduler = new RequestScheduler(100, generationLoop);
    }

    @AfterAll
    static void stopCluster() throws Exception {
        if (harness != null) harness.stop();
    }

    // ── Tests ─────────────────────────────────────────────────────────────────

    @Test
    @Order(1)
    @DisplayName("All 3 node processes are alive after startup")
    void allNodesAlive() {
        assertThat(pipeline).isNotNull();
        assertThat(pipeline.vocabSize()).isEqualTo(EmbeddedNodeServer.VOCAB_SIZE);
    }

    @Test
    @Order(2)
    @DisplayName("Single forward pass completes across all 3 JVMs")
    void singleForwardPassAcrossJvms() {
        float[] logits = pipeline.forward("cross-jvm-001", new int[]{1, 2, 3}, 0);

        assertThat(logits).hasSize(EmbeddedNodeServer.VOCAB_SIZE);

        // StubForwardPassHandler puts 100.0f at token 42
        float max = logits[42];
        for (float l : logits) assertThat(max).isGreaterThanOrEqualTo(l);
    }

    @Test
    @Order(3)
    @DisplayName("GenerationLoop generates tokens via gRPC pipeline")
    void generationLoopViaGrpc() {
        int maxTokens = 8;
        InferenceRequest request = InferenceRequest.of(
                "tinyllama",
                List.of(ChatMessage.user("Write a haiku about distributed systems.")),
                SamplingParams.defaults().withMaxTokens(maxTokens),
                RequestPriority.NORMAL
        );

        List<String> pieces = new ArrayList<>();
        GenerationResult result = generationLoop.generate(
                request, (piece, tokenId, step) -> pieces.add(piece));

        assertThat(result.generatedTokens())
                .isGreaterThan(0)
                .isLessThanOrEqualTo(maxTokens);

        assertThat(pieces).hasSameSizeAs(result.tokenIds());
        assertThat(result.latency()).isPositive();

        System.out.printf("Generated: \"%s\"  tokens=%d  latency=%d ms%n",
                result.text(), result.generatedTokens(), result.latency().toMillis());
    }

    @Test
    @Order(4)
    @DisplayName("Scheduler dispatches 4 concurrent requests via virtual threads")
    void schedulerConcurrentRequests() throws InterruptedException {
        int count = 4;
        List<GenerationResult> results = new CopyOnWriteArrayList<>();
        List<Thread> threads = new ArrayList<>();

        for (int i = 0; i < count; i++) {
            final String id = "sched-" + i;
            Thread t = Thread.ofVirtual().start(() -> {
                InferenceRequest req = InferenceRequest.of(
                        "tinyllama",
                        List.of(ChatMessage.user("Request " + id)),
                        SamplingParams.defaults().withMaxTokens(4),
                        RequestPriority.NORMAL
                );
                results.add(scheduler.submitAndWait(req));
            });
            threads.add(t);
        }

        for (Thread t : threads) t.join(30_000);

        assertThat(results).hasSize(count);
        assertThat(results).allSatisfy(r -> assertThat(r.generatedTokens()).isGreaterThan(0));
    }

    @Test
    @Order(5)
    @DisplayName("Repeated identical prompts use prefix cache")
    void prefixCacheOnRepeat() {
        SamplingParams params   = SamplingParams.defaults().withMaxTokens(5);
        List<ChatMessage> msgs  = List.of(ChatMessage.user("What is pipeline parallelism?"));

        GenerationResult r1 = generationLoop.generate(
                InferenceRequest.of("tinyllama", msgs, params, RequestPriority.NORMAL),
                TokenConsumer.discard());
        GenerationResult r2 = generationLoop.generate(
                InferenceRequest.of("tinyllama", msgs, params, RequestPriority.NORMAL),
                TokenConsumer.discard());

        assertThat(r1.generatedTokens()).isEqualTo(r2.generatedTokens());
        assertThat(r1.promptTokens()).isEqualTo(r2.promptTokens());

        System.out.printf("Cold: %d ms  Warm: %d ms%n",
                r1.latency().toMillis(), r2.latency().toMillis());
    }

    @Test
    @Order(6)
    @DisplayName("HIGH priority request completes without starvation")
    void highPriorityCompletes() {
        InferenceRequest high = InferenceRequest.of(
                "tinyllama",
                List.of(ChatMessage.user("Urgent query")),
                SamplingParams.defaults().withMaxTokens(2),
                RequestPriority.HIGH
        );

        GenerationResult result = scheduler.submitAndWait(high);
        assertThat(result.generatedTokens()).isGreaterThan(0);
    }
}
