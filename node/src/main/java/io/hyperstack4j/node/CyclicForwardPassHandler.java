package io.hyperstack4j.node;

import java.util.concurrent.atomic.AtomicInteger;

/**
 * Deterministic ForwardPassHandler for tests and integration testing.
 *
 * 
 * Intermediate nodes: returns a fixed-pattern float[] activation of hiddenDim
 * size. Last node: returns logits with all probability mass on a configurable
 * winner token.
 *
 * No GPU, no model weights, no JCuda. Compiles and runs anywhere.
 */
public final class CyclicForwardPassHandler implements ForwardPassHandler {

	private final int winnerToken; // last-node logit winner
	private final AtomicInteger callCount = new AtomicInteger(0);

	public CyclicForwardPassHandler() {
		this.winnerToken = 42;
	}

	public CyclicForwardPassHandler(int winnerToken) {
		this.winnerToken = winnerToken;
	}

	@Override
	public ForwardResult forward(ForwardRequest request, ShardContext context) {
		callCount.incrementAndGet();
		long start = System.nanoTime();

		if (context.hasOutputProjection()) {
			// Last node — return logits
			float[] logits = new float[context.vocabSize()];
			logits[winnerToken] = 100.0f;
			return ForwardResult.logits(request.requestId(), logits, System.nanoTime() - start);
		} else {
			// Intermediate node — return activations (deterministic pattern)
			float[] activations = new float[context.hiddenDim()];
			for (int i = 0; i < activations.length; i++) {
				activations[i] = 0.01f * (i % 100);
			}
			return ForwardResult.activations(request.requestId(), activations, System.nanoTime() - start);
		}
	}

	@Override
	public boolean isReady() {
		return true;
	}

	public int callCount() {
		return callCount.get();
	}
}