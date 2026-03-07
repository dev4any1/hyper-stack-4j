package io.hyperstack4j.coordinator;

import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.PriorityBlockingQueue;
import java.util.logging.Logger;

import io.hyperstack4j.coordinator.RequestScheduler.QueueFullException;

/**
 * Priority-aware request scheduler.
 *
 * Maintains a bounded PriorityBlockingQueue ordered by RequestPriority (HIGH
 * first), with FIFO ordering within the same priority level.
 *
 * Every request — streaming or blocking — is dispatched on its own Java 21
 * Virtual Thread. The caller either consumes tokens via a TokenConsumer
 * callback (streaming), or calls submitAndWait() which returns a
 * CompletableFuture and joins it.
 *
 * submitAndWait() flow: 1. CompletableFuture<GenerationResult> created and
 * registered BEFORE queue.offer() 2. Virtual thread dispatched — runs
 * generate(), completes the future on finish 3. Caller blocks on future.join()
 * — wakes up exactly when its own request is done
 *
 * This means N concurrent callers each block on their own independent future,
 * not on each other. No shared mutable state between concurrent requests.
 *
 * Thread-safe — PriorityBlockingQueue + ConcurrentHashMap handle concurrent
 * producers.
 */
public final class RequestScheduler {

	private static final Logger log = Logger.getLogger(RequestScheduler.class.getName());

	private final int maxQueueDepth;
	private final PriorityBlockingQueue<InferenceRequest> queue;
	private final GenerationLoop generationLoop;
	private final ConcurrentHashMap<String, CompletableFuture<GenerationResult>> futures = new ConcurrentHashMap<>();

	public RequestScheduler(int maxQueueDepth, GenerationLoop generationLoop) {
		if (maxQueueDepth < 1)
			throw new IllegalArgumentException("maxQueueDepth must be >= 1");
		if (generationLoop == null)
			throw new IllegalArgumentException("generationLoop must not be null");

		this.maxQueueDepth = maxQueueDepth;
		this.generationLoop = generationLoop;
		this.queue = new PriorityBlockingQueue<>(maxQueueDepth);
	}

	/**
	 * Submit a request for async streaming execution. Returns immediately — tokens
	 * delivered via the TokenConsumer. The returned future completes when
	 * generation finishes.
	 *
	 * @throws QueueFullException if the queue has reached maxQueueDepth
	 */
	public CompletableFuture<GenerationResult> submit(InferenceRequest request, TokenConsumer consumer) {
		if (queue.size() >= maxQueueDepth) {
			throw new QueueFullException("Request queue full (" + maxQueueDepth + "). Retry later.",
					estimateRetryAfterSeconds());
		}

		CompletableFuture<GenerationResult> future = new CompletableFuture<>();
		futures.put(request.requestId(), future); // register BEFORE queue.offer()
		queue.offer(request);
		dispatch(request, consumer, future);
		return future;
	}

	/**
	 * Submit and block until generation completes (non-streaming use case).
	 *
	 * Each caller blocks only on its own future — N concurrent callers run fully in
	 * parallel on separate virtual threads, independent of each other.
	 *
	 * @throws QueueFullException if the queue has reached maxQueueDepth
	 */
	public GenerationResult submitAndWait(InferenceRequest request) {
		return submit(request, TokenConsumer.discard()).join();
	}

	public int queueDepth() {
		return queue.size();
	}

	public int maxQueueDepth() {
		return maxQueueDepth;
	}

	// ── Private ───────────────────────────────────────────────────────────────

	private void dispatch(InferenceRequest request, TokenConsumer consumer,
			CompletableFuture<GenerationResult> future) {
		Thread.ofVirtual().name("gen-" + request.requestId()).start(() -> {
			try {
				GenerationResult result = generationLoop.generate(request, consumer);
				future.complete(result);
			} catch (Exception e) {
				log.warning("Generation failed for " + request.requestId() + ": " + e.getMessage());
				future.completeExceptionally(e);
			} finally {
				queue.remove(request);
				futures.remove(request.requestId());
			}
		});
	}

	private int estimateRetryAfterSeconds() {
		return Math.max(1, queue.size() * 2);
	}

	// ── Exceptions ────────────────────────────────────────────────────────────

	public static final class QueueFullException extends RuntimeException {
		private final int retryAfterSeconds;

		public QueueFullException(String message, int retryAfterSeconds) {
			super(message);
			this.retryAfterSeconds = retryAfterSeconds;
		}

		public int retryAfterSeconds() {
			return retryAfterSeconds;
		}
	}
}