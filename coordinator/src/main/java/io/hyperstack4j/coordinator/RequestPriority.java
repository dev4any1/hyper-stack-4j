package io.hyperstack4j.coordinator;

/**
 * Priority level for inference requests. Weight determines effective ordering
 * in PriorityBlockingQueue. Higher weight = scheduled sooner.
 */
public enum RequestPriority {

	HIGH(3), NORMAL(1), LOW(0);

	private final int weight;

	RequestPriority(int weight) {
		this.weight = weight;
	}

	public int weight() {
		return weight;
	}
}
