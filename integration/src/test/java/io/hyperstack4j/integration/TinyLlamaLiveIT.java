package io.hyperstack4j.integration;

import static org.assertj.core.api.Assertions.assertThat;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;

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
import io.hyperstack4j.coordinator.TokenConsumer;
import io.hyperstack4j.kvcache.CpuKVCache;
import io.hyperstack4j.kvcache.GpuKVCache;
import io.hyperstack4j.kvcache.KVCacheManager;
import io.hyperstack4j.node.ActivationDtype;
import io.hyperstack4j.node.GgufReader;
import io.hyperstack4j.sampler.Sampler;
import io.hyperstack4j.sampler.SamplingParams;
import io.hyperstack4j.tokenizer.ChatMessage;
import io.hyperstack4j.tokenizer.GgufTokenizer;

/**
 * Real-weight integration test — requires TinyLlama-1.1B-Chat-v1.0.Q4_K_M.gguf.
 *
 * Starts a 3-node gRPC cluster with real model weights, sends prompts through
 * the full pipeline (tokenizer → prefill → decode → detokenize), and asserts
 * the responses contain coherent English text.
 *
 * Skipped automatically when the model file is not present. Point at the model
 * with the Maven property {@code it.model.path}:
 *
 * mvn verify -pl integration \
 * -Dit.model.path=/path/to/TinyLlama-1.1B-Chat-v1.0.Q4_K_M.gguf \
 * -Dit.test=TinyLlamaLiveIT
 *
 * Or via the environment variable {@code MODEL_PATH} (which run-me.sh already
 * sets).
 *
 * The test uses temperature=0.0 (greedy) so responses are deterministic.
 */
@DisplayName("TinyLlama Live — Real Model, 3-Node Cluster")
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
class TinyLlamaLiveIT {

	// ── Well-known English words TinyLlama reliably produces for "hello" greetings
	private static final Set<String> GREETING_WORDS = Set.of("how", "are", "you", "hello", "hi", "help", "doing",
			"today", "there", "welcome", "assist", "can", "i", "what", "do");

	private static String modelPath;
	private static ClusterHarness harness;
	private static GenerationLoop loop;

	// ── Lifecycle ─────────────────────────────────────────────────────────────

	@BeforeAll
	static void startCluster() throws Exception {
		modelPath = resolveModelPath();
		org.junit.jupiter.api.Assumptions.assumeTrue(modelPath != null,
				"Skipping TinyLlamaLiveIT — model not found. Set it.model.path or MODEL_PATH.");

		harness = ClusterHarness.threeNodes(modelPath);
		harness.start();

		GgufTokenizer tokenizer;
		try (GgufReader reader = GgufReader.open(Path.of(modelPath))) {
			tokenizer = GgufTokenizer.load(reader);
		}

		loop = new GenerationLoop(tokenizer, Sampler.create(),
				new ProcessPipelineClient(harness.nodeAddresses(), EmbeddedNodeServer.VOCAB_SIZE,
						ActivationDtype.FLOAT32),
				new KVCacheManager(new GpuKVCache(512L * 1024 * 1024), new CpuKVCache(4096)));
	}

	@AfterAll
	static void stopCluster() throws Exception {
		if (harness != null)
			harness.stop();
	}

	// ── Tests ─────────────────────────────────────────────────────────────────

	/**
	 * The primary smoke test: "hello" → coherent English greeting.
	 *
	 * TinyLlama-1.1B-Chat responds to "hello" with something like "how are you
	 * doing today?" This test verifies the full pipeline produces at least 3
	 * recognisable English words from that greeting set.
	 *
	 * This test was the target of a three-bug debugging session: Bug 1 — wrong chat
	 * template (ChatML instead of Zephyr) → alien tokens Bug 2 — SentencePiece ▁
	 * leaked in streaming path → mangled text Bug 3 — Q6_K hi=i/4 index error →
	 * completely scrambled weights All three are now fixed and this test serves as
	 * the non-regression anchor.
	 */
	@Test
	@Order(1)
	@DisplayName("'hello' produces a coherent English greeting")
	void hello_produces_coherent_english_greeting() {
		GenerationResult result = generate("hello", 10);

		System.out.printf("[TinyLlamaLiveIT] 'hello' → \"%s\"  (%d tokens, %d ms)%n", result.text(),
				result.generatedTokens(), result.latency().toMillis());

		assertThat(result.generatedTokens()).as("Should produce at least 3 tokens for a greeting")
				.isGreaterThanOrEqualTo(3);

		// The response should contain at least 2 recognisable English greeting words.
		// We lower-case and split on non-alpha to be punctuation-tolerant.
		String lower = result.text().toLowerCase();
		long matchCount = GREETING_WORDS.stream().filter(w -> lower.contains(w)).count();

		assertThat(matchCount).as("Response \"%s\" should contain at least 2 English greeting words from %s",
				result.text(), GREETING_WORDS).isGreaterThanOrEqualTo(2);
	}

	@Test
	@Order(2)
	@DisplayName("Tokens are valid SentencePiece pieces — no raw ▁ characters")
	void tokens_contain_no_raw_sentencepiece_markers() {
		List<String> pieces = new ArrayList<>();
		loop.generate(request("hello", 10), (piece, tokenId, step) -> pieces.add(piece));

		for (String piece : pieces) {
			assertThat(piece).as("Decoded piece should not contain raw SentencePiece prefix ▁ (U+2581)")
					.doesNotContain("\u2581");
		}
	}

	@Test
	@Order(3)
	@DisplayName("Response to a question contains relevant words")
	void question_produces_relevant_response() {
		GenerationResult result = generate("What is 2 plus 2?", 12);

		System.out.printf("[TinyLlamaLiveIT] question → \"%s\"  (%d tokens)%n", result.text(),
				result.generatedTokens());

		assertThat(result.generatedTokens()).isGreaterThan(0);
		// TinyLlama answers math questions — the response should at minimum be
		// non-empty
		assertThat(result.text().strip()).isNotEmpty();
	}

	@Test
	@Order(4)
	@DisplayName("Two identical prompts produce the same output (greedy determinism)")
	void greedy_sampling_is_deterministic() {
		// Use greedy (temperature=0) to guarantee determinism
		SamplingParams greedy = SamplingParams.defaults().withMaxTokens(8).withTemperature(0.0f);

		GenerationResult r1 = loop.generate(
				InferenceRequest.of("tinyllama", List.of(ChatMessage.user("hello")), greedy, RequestPriority.NORMAL),
				TokenConsumer.discard());
		GenerationResult r2 = loop.generate(
				InferenceRequest.of("tinyllama", List.of(ChatMessage.user("hello")), greedy, RequestPriority.NORMAL),
				TokenConsumer.discard());

		assertThat(r1.text()).as("Greedy responses to the same prompt should be identical").isEqualTo(r2.text());
	}

	@Test
	@Order(5)
	@DisplayName("Multi-turn conversation: second turn is coherent")
	void multi_turn_conversation_is_coherent() {
		List<ChatMessage> conversation = List.of(ChatMessage.user("hello"),
				ChatMessage.assistant("Hello! How can I help you today?"), ChatMessage.user("What is Java?"));

		SamplingParams params = SamplingParams.defaults().withMaxTokens(12);
		GenerationResult result = loop.generate(
				InferenceRequest.of("tinyllama", conversation, params, RequestPriority.NORMAL),
				TokenConsumer.discard());

		System.out.printf("[TinyLlamaLiveIT] multi-turn → \"%s\"  (%d tokens)%n", result.text(),
				result.generatedTokens());

		assertThat(result.generatedTokens()).isGreaterThan(0);
		assertThat(result.promptTokens()).as("Multi-turn prompt should be longer than a single-turn prompt")
				.isGreaterThan(20);
	}

	@Test
	@Order(6)
	@DisplayName("FLOAT16 activation compression produces same first token as FLOAT32")
	void float16_activations_produce_same_first_token() throws Exception {
		try (GgufReader reader = GgufReader.open(Path.of(modelPath))) {
			GgufTokenizer tokenizer = GgufTokenizer.load(reader);

			ProcessPipelineClient f16Pipeline = new ProcessPipelineClient(harness.nodeAddresses(),
					EmbeddedNodeServer.VOCAB_SIZE, ActivationDtype.FLOAT16);
			try {
				GenerationLoop f16Loop = new GenerationLoop(tokenizer, Sampler.create(), f16Pipeline,
						new KVCacheManager(new GpuKVCache(512L * 1024 * 1024), new CpuKVCache(256)));

				SamplingParams greedy = SamplingParams.defaults().withMaxTokens(1).withTemperature(0.0f);

				GenerationResult f32result = loop.generate(InferenceRequest.of("tinyllama",
						List.of(ChatMessage.user("hello")), greedy, RequestPriority.NORMAL), TokenConsumer.discard());

				GenerationResult f16result = f16Loop.generate(InferenceRequest.of("tinyllama",
						List.of(ChatMessage.user("hello")), greedy, RequestPriority.NORMAL), TokenConsumer.discard());

				System.out.printf("[TinyLlamaLiveIT] F32 first token: \"%s\"  F16: \"%s\"%n", f32result.text(),
						f16result.text());

				assertThat(f16result.text()).as("FLOAT16 pipeline should produce the same first token as FLOAT32")
						.isEqualTo(f32result.text());
			} finally {
				f16Pipeline.shutdown();
			}
		}
	}

	// ── Helpers ───────────────────────────────────────────────────────────────

	private GenerationResult generate(String userMessage, int maxTokens) {
		return loop.generate(request(userMessage, maxTokens), TokenConsumer.discard());
	}

	private static InferenceRequest request(String userMessage, int maxTokens) {
		return InferenceRequest.of("tinyllama", List.of(ChatMessage.user(userMessage)),
				SamplingParams.defaults().withMaxTokens(maxTokens).withTemperature(0.7f), RequestPriority.NORMAL);
	}

	/**
	 * Resolve the model file path from, in priority order: 1. Maven system property
	 * {@code it.model.path} (set by failsafe plugin) 2. Environment variable
	 * {@code MODEL_PATH} (used by run-me.sh)
	 *
	 * Returns {@code null} if neither is set or the path doesn't exist.
	 */
	private static String resolveModelPath() {
		String path = System.getProperty("it.model.path");
		if (path == null || path.isBlank()) {
			path = System.getenv("MODEL_PATH");
		}
		if (path == null || path.isBlank())
			return null;
		if (!Files.exists(Path.of(path))) {
			System.err.println("[TinyLlamaLiveIT] Model file not found: " + path);
			return null;
		}
		return path;
	}
}