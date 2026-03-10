package io.hyperstack4j.health;

import java.io.Serializable;
import java.time.Instant;

/**
 * Immutable health snapshot for a single inference node. Published by each
 * node's GPU health probe every 5s. Stored in Hazelcast IMap("node-health")
 * keyed by nodeId.
 */
public record NodeHealth(String nodeId, double vramPressure, // 0.0 → 1.0
		long vramFreeBytes, long vramTotalBytes, double temperatureCelsius, // -1.0 if unavailable
		double inferenceLatencyP99Ms, // -1.0 if no data yet
		Instant sampledAt) implements Serializable {

	public NodeHealth {
		if (nodeId == null || nodeId.isBlank())
			throw new IllegalArgumentException("nodeId must not be blank");
		if (vramPressure < 0.0 || vramPressure > 1.0)
			throw new IllegalArgumentException("vramPressure must be in [0.0, 1.0]");
	}

	/** Whether VRAM is above the warning threshold (default 90%). */
	public boolean isVramWarning(double threshold) {
		return vramPressure >= threshold;
	}

	/** Whether VRAM is above the critical threshold (default 98%). */
	public boolean isVramCritical(double threshold) {
		return vramPressure >= threshold;
	}

	/** How stale this snapshot is. */
	public long ageMillis() {
		return Instant.now().toEpochMilli() - sampledAt.toEpochMilli();
	}
}
