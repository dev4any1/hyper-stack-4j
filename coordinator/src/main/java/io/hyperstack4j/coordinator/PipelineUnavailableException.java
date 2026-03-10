package io.hyperstack4j.coordinator;

/**
 * Thrown by FaultTolerantPipeline when no node can serve the request.
 *
 * Two scenarios: CIRCUIT_OPEN — all node circuits were OPEN at call time (no
 * attempt made) RETRIES_EXHAUSTED — one or more nodes were tried but all threw
 * exceptions
 *
 * The REST layer converts this to HTTP 503 with a Retry-After hint. The
 * scheduler may re-queue the request or complete the future exceptionally.
 */
public final class PipelineUnavailableException extends RuntimeException {

	public enum Reason {
		/**
		 * All node circuit breakers were OPEN — request rejected without attempting a
		 * forward pass.
		 */
		CIRCUIT_OPEN,
		/** At least one attempt was made but all tried nodes threw exceptions. */
		RETRIES_EXHAUSTED
	}

	private final Reason reason;
	private final int attemptsMade;

	public PipelineUnavailableException(Reason reason, int attemptsMade, String detail) {
		super(String.format("[%s] %s (attempts: %d)", reason, detail, attemptsMade));
		this.reason = reason;
		this.attemptsMade = attemptsMade;
	}

	public PipelineUnavailableException(Reason reason, int attemptsMade, String detail, Throwable cause) {
		super(String.format("[%s] %s (attempts: %d)", reason, detail, attemptsMade), cause);
		this.reason = reason;
		this.attemptsMade = attemptsMade;
	}

	public Reason reason() {
		return reason;
	}

	public int attemptsMade() {
		return attemptsMade;
	}

	/**
	 * Whether the failure is potentially transient — worth retrying at the
	 * scheduler level.
	 */
	public boolean isRetryable() {
		return reason == Reason.RETRIES_EXHAUSTED;
	}
}
