package io.hyperstack4j.health;

/**
 * Circuit breaker state machine.
 *
 * Transitions:
 *   CLOSED → OPEN      when failure rate exceeds threshold
 *   OPEN   → HALF_OPEN after waitDuration elapses
 *   HALF_OPEN → CLOSED on probe success
 *   HALF_OPEN → OPEN   on probe failure
 */
public enum CircuitState {

    /** Normal operation — calls pass through. */
    CLOSED,

    /** Failing — calls are rejected immediately. */
    OPEN,

    /** Testing recovery — limited calls allowed through. */
    HALF_OPEN
}
