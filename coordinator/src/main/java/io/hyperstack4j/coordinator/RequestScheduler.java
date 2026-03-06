package io.hyperstack4j.coordinator;

import java.util.concurrent.PriorityBlockingQueue;
import java.util.concurrent.RejectedExecutionException;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.logging.Logger;

/**
 * Priority-aware request scheduler.
 *
 * Maintains a bounded PriorityBlockingQueue ordered by RequestPriority (HIGH first),
 * with FIFO ordering within the same priority level.
 *
 * Each accepted request is dispatched on its own Java 21 Virtual Thread.
 * Virtual threads are cheap — no pooling needed, one per in-flight request.
 *
 * When the queue is full: submit() throws QueueFullException,
 * which the REST layer translates to HTTP 503 + Retry-After.
 *
 * Thread-safe — PriorityBlockingQueue handles concurrent producers.
 */
public final class RequestScheduler {

    private static final Logger log = Logger.getLogger(RequestScheduler.class.getName());

    private final int                               maxQueueDepth;
    private final PriorityBlockingQueue<InferenceRequest> queue;
    private final GenerationLoop                    generationLoop;
    private final AtomicBoolean                     running = new AtomicBoolean(false);

    public RequestScheduler(int maxQueueDepth, GenerationLoop generationLoop) {
        if (maxQueueDepth < 1)
            throw new IllegalArgumentException("maxQueueDepth must be >= 1");
        if (generationLoop == null)
            throw new IllegalArgumentException("generationLoop must not be null");

        this.maxQueueDepth  = maxQueueDepth;
        this.generationLoop = generationLoop;
        this.queue          = new PriorityBlockingQueue<>(maxQueueDepth);
    }

    /**
     * Submit a request for async execution.
     * Returns immediately — result is delivered via the TokenConsumer.
     *
     * @throws QueueFullException if the queue has reached maxQueueDepth
     */
    public void submit(InferenceRequest request, TokenConsumer consumer) {
        if (queue.size() >= maxQueueDepth) {
            throw new QueueFullException(
                    "Request queue full (" + maxQueueDepth + "). Retry later.",
                    estimateRetryAfterSeconds()
            );
        }
        queue.offer(request);
        dispatch(request, consumer);
    }

    /**
     * Submit and block until generation completes (non-streaming use case).
     */
    public GenerationResult submitAndWait(InferenceRequest request) {
        if (queue.size() >= maxQueueDepth) {
            throw new QueueFullException(
                    "Request queue full (" + maxQueueDepth + "). Retry later.",
                    estimateRetryAfterSeconds()
            );
        }
        queue.offer(request);
        try {
            return generationLoop.generate(request, TokenConsumer.discard());
        } finally {
            queue.remove(request);
        }
    }

    public int queueDepth()    { return queue.size(); }
    public int maxQueueDepth() { return maxQueueDepth; }

    // ── Private ───────────────────────────────────────────────────────────────

    private void dispatch(InferenceRequest request, TokenConsumer consumer) {
        Thread.ofVirtual()
              .name("gen-" + request.requestId())
              .start(() -> {
                  try {
                      generationLoop.generate(request, consumer);
                  } catch (Exception e) {
                      log.warning("Generation failed for " + request.requestId()
                              + ": " + e.getMessage());
                  } finally {
                      queue.remove(request);
                  }
              });
    }

    private int estimateRetryAfterSeconds() {
        // Rough estimate: assume ~2s per request, queue drains FIFO
        return Math.max(1, queue.size() * 2);
    }

    // ── Exceptions ────────────────────────────────────────────────────────────

    public static final class QueueFullException extends RuntimeException {
        private final int retryAfterSeconds;

        public QueueFullException(String message, int retryAfterSeconds) {
            super(message);
            this.retryAfterSeconds = retryAfterSeconds;
        }

        public int retryAfterSeconds() { return retryAfterSeconds; }
    }
}
