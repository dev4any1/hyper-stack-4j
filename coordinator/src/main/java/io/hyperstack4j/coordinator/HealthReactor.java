package io.hyperstack4j.coordinator;

import java.util.List;
import java.util.logging.Logger;

import io.hyperstack4j.health.HealthEvent;
import io.hyperstack4j.health.HealthThresholds;
import io.hyperstack4j.health.NodeHealth;
import io.hyperstack4j.health.HealthEvaluator;

/**
 * Bridges the health subsystem to the coordinator's fault-tolerance layer.
 *
 * Owns a HealthEvaluator and reacts to the events it emits:
 *
 *   VRAM_CRITICAL → force-open the node's circuit (reject new forward passes)
 *   NODE_STALE    → force-open the node's circuit (missed health probes)
 *   NODE_RECOVERED → reset the node's circuit to CLOSED (allow traffic back)
 *   VRAM_WARNING  → logged only; eviction is handled by KVCacheManager separately
 *
 * Additionally, if all node circuits become OPEN (isFullyUnavailable()), the
 * RequestScheduler is shut down to stop accepting new requests until at least
 * one node recovers.
 *
 * ── Wiring in production ──────────────────────────────────────────────────────
 * Hazelcast EntryAddedListener on IMap("node-health") calls:
 *
 *   healthReactor.onHealthProbe(newHealth);
 *
 * That single call drives the full evaluate → react cycle.
 *
 * ── Thread Safety ─────────────────────────────────────────────────────────────
 * onHealthProbe() is safe for concurrent calls (HealthEvaluator uses
 * ConcurrentHashMap internally). Each probe evaluation is independent.
 */
public final class HealthReactor {

    private static final Logger log = Logger.getLogger(HealthReactor.class.getName());

    private final HealthEvaluator         evaluator;
    private final FaultTolerantPipeline   pipeline;
    private final RequestScheduler        scheduler;  // may be null in unit tests

    /**
     * Full constructor — reacts to health events and can shut down the scheduler
     * if the cluster becomes fully unavailable.
     */
    public HealthReactor(HealthThresholds thresholds,
                         FaultTolerantPipeline pipeline,
                         RequestScheduler scheduler) {
        if (thresholds == null) throw new IllegalArgumentException("thresholds must not be null");
        if (pipeline   == null) throw new IllegalArgumentException("pipeline must not be null");

        this.evaluator = new HealthEvaluator(thresholds);
        this.pipeline  = pipeline;
        this.scheduler = scheduler;
    }

    /**
     * Constructor without scheduler — for unit tests and wiring scenarios
     * where the scheduler isn't relevant.
     */
    public HealthReactor(HealthThresholds thresholds, FaultTolerantPipeline pipeline) {
        this(thresholds, pipeline, null);
    }

    /**
     * Accept a fresh health probe from a node and react to any state transitions.
     *
     * Called by the Hazelcast IMap listener in production; called directly in tests.
     * Safe for concurrent calls from multiple Hazelcast listener threads.
     *
     * @param probe fresh NodeHealth snapshot
     */
    public void onHealthProbe(NodeHealth probe) {
        List<HealthEvent> events = evaluator.evaluate(probe);
        for (HealthEvent event : events) {
            react(event);
        }
    }

    /**
     * Notify the reactor that a node has left the cluster entirely (e.g. Hazelcast
     * memberRemoved event). Equivalent to a permanent stale — force-opens the circuit
     * and removes state from the evaluator.
     */
    public void onNodeRemoved(String nodeId) {
        log.warning("Node removed from cluster: " + nodeId + " — force-opening circuit");
        pipeline.onNodeStale(nodeId);
        evaluator.forget(nodeId);
    }

    // ── Private ───────────────────────────────────────────────────────────────

    private void react(HealthEvent event) {
        String nodeId = event.nodeId();

        switch (event.type()) {
            case VRAM_CRITICAL -> {
                log.warning("HealthReactor: VRAM_CRITICAL on " + nodeId
                        + " — " + event.detail());
                pipeline.onVramCritical(nodeId);
                checkFullyUnavailable();
            }
            case NODE_STALE -> {
                log.warning("HealthReactor: NODE_STALE on " + nodeId
                        + " — " + event.detail());
                pipeline.onNodeStale(nodeId);
                checkFullyUnavailable();
            }
            case NODE_RECOVERED -> {
                log.info("HealthReactor: NODE_RECOVERED on " + nodeId);
                pipeline.onNodeRecovered(nodeId);
                // If scheduler was shut down due to full unavailability,
                // a new RequestScheduler would need to be created by the coordinator.
                // That lifecycle is handled by the coordinator — not here.
            }
            case VRAM_WARNING -> {
                // Warning only — eviction is handled by KVCacheManager separately.
                // Circuit stays CLOSED; the coordinator can reduce batch size here
                // in a future iteration.
                log.info("HealthReactor: VRAM_WARNING on " + nodeId
                        + " — " + event.detail());
            }
        }
    }

    /**
     * If the pipeline is fully unavailable after a circuit-open event, shut down
     * the scheduler to stop accepting new requests. The coordinator must restart
     * it when a node recovers.
     */
    private void checkFullyUnavailable() {
        if (pipeline.isFullyUnavailable() && scheduler != null) {
            log.severe("HealthReactor: ALL nodes unavailable — shutting down scheduler");
            scheduler.shutdown();
        }
    }
}
