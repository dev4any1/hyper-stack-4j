package io.hyperstack4j.registry;

/**
 * Thrown when the cluster does not have enough free VRAM to accommodate all
 * layers of the requested model.
 */
public final class InsufficientClusterVramException extends RuntimeException {

	public InsufficientClusterVramException(String message) {
		super(message);
	}
}
