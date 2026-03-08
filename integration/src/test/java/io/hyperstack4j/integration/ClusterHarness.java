package io.hyperstack4j.integration;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.TimeUnit;
import java.util.logging.Logger;

/**
 * Manages the lifecycle of 3 forked node JVM processes for integration testing.
 *
 * Each node is launched as a separate JVM via ProcessBuilder running NodeMain.
 * The harness waits for each node to print "READY:<nodeId>:<port>" before
 * proceeding — so tests don't start sending requests to nodes that aren't up
 * yet.
 *
 * Memory allocation per node JVM (16 GB host): -Xms512m -Xmx4g -XX:+UseZGC
 *
 * Usage: ClusterHarness harness = ClusterHarness.threeNodes(); harness.start();
 * // ... run tests using harness.pipelineClient() ... harness.stop();
 *
 * Or with try-with-resources (implements AutoCloseable): try (ClusterHarness
 * harness = ClusterHarness.threeNodes()) { harness.start(); // ... tests ... }
 */
public final class ClusterHarness implements AutoCloseable {

	private static final Logger log = Logger.getLogger(ClusterHarness.class.getName());

	/** How long to wait for each node to report READY (ms). */
	private static final long NODE_STARTUP_TIMEOUT_MS = 30_000;

	/** gRPC ports for the three nodes. */
	static final int NODE_1_PORT = 19092;
	static final int NODE_2_PORT = 19093;
	static final int NODE_3_PORT = 19094;

	/** TinyLlama-1.1B shape constants — matches EmbeddedNodeServer defaults. */
	private static final int VOCAB_SIZE = EmbeddedNodeServer.VOCAB_SIZE;
	private static final int TOTAL_LAYERS = EmbeddedNodeServer.TOTAL_LAYERS;

	private final List<NodeSpec> specs;
	private final List<Process> processes = new ArrayList<>();
	private ProcessPipelineClient pipelineClient;
	private static String modelPath; // null = stub mode

	private ClusterHarness(List<NodeSpec> specs) {
		this(specs, null);
	}

	private ClusterHarness(List<NodeSpec> specs, String pathToModel) {
		this.specs = specs;
		modelPath = pathToModel;
	}

	/**
	 * Create a standard 3-node cluster.
	 *
	 * Layer split for 22-layer TinyLlama across 3 nodes: Node-1: layers 0– 7 (8
	 * layers) + embeddings Node-2: layers 8–14 (7 layers) Node-3: layers 15–21 (7
	 * layers) + output projection
	 */
	public static ClusterHarness threeNodes() {
		return threeNodes(null);
	}

	/**
	 * Create a 3-node cluster that loads real weights from a GGUF file. Each node
	 * receives the model path as a CLI arg and loads its layer shard via
	 * CpuForwardPassHandler when loadShard() is called.
	 *
	 * @param modelPath path to a GGUF file, e.g.
	 *                  "/models/TinyLlama-1.1B-Chat.Q4_K_M.gguf" Pass null for stub
	 *                  mode (no real model).
	 */
	public static ClusterHarness threeNodes(String modelPath) {
		return new ClusterHarness(List.of(
				new NodeSpec("node-1", "localhost", NODE_1_PORT,
						new ProcessPipelineClient.ShardConfig(0, 8, true, false)),
				new NodeSpec("node-2", "localhost", NODE_2_PORT,
						new ProcessPipelineClient.ShardConfig(8, 15, false, false)),
				new NodeSpec("node-3", "localhost", NODE_3_PORT,
						new ProcessPipelineClient.ShardConfig(15, 22, false, true))),
				modelPath);
	}

	/**
	 * Fork all three node JVMs and wait until each reports READY.
	 */
	public void start() throws IOException, InterruptedException {
		for (NodeSpec spec : specs) {
			Process proc = launchNode(spec.nodeId(), spec.port());
			processes.add(proc);
			waitForReady(proc, spec.nodeId());
			log.info("Node [" + spec.nodeId() + "] is up on port " + spec.port());
		}

		// Wire up the pipeline client
		List<ProcessPipelineClient.NodeAddress> addresses = specs.stream()
				.map(s -> new ProcessPipelineClient.NodeAddress(s.host(), s.port())).toList();

		pipelineClient = new ProcessPipelineClient(addresses, VOCAB_SIZE);

		// Tell each node which shard it owns
		List<ProcessPipelineClient.ShardConfig> shards = specs.stream().map(NodeSpec::shard).toList();
		pipelineClient.loadShards(shards);

		log.info("Cluster ready — 3 nodes, " + TOTAL_LAYERS + " total layers distributed");
	}

	/**
	 * Returns the InferencePipeline that routes forward passes across the 3 nodes.
	 * Only valid after start() has been called.
	 */
	public ProcessPipelineClient pipelineClient() {
		if (pipelineClient == null)
			throw new IllegalStateException("ClusterHarness not started — call start() first");
		return pipelineClient;
	}

	/**
	 * Returns the node addresses in pipeline order. Use this to create additional
	 * ProcessPipelineClient instances (e.g. with a different ActivationDtype)
	 * pointing at the same running nodes.
	 */
	public List<ProcessPipelineClient.NodeAddress> nodeAddresses() {
		return specs.stream().map(s -> new ProcessPipelineClient.NodeAddress(s.host(), s.port())).toList();
	}

	/**
	 * Shut down all node processes and close gRPC channels.
	 */
	public void stop() throws InterruptedException {
		if (pipelineClient != null) {
			pipelineClient.shutdown();
		}
		for (Process proc : processes) {
			proc.destroy();
			proc.waitFor(5, TimeUnit.SECONDS);
		}
		processes.clear();
		log.info("Cluster stopped");
	}

	@Override
	public void close() throws Exception {
		stop();
	}

	// ── Private ───────────────────────────────────────────────────────────────

	private Process launchNode(String nodeId, int port) throws IOException {
		String javaExe = ProcessHandle.current().info().command()
				.orElse(Path.of(System.getProperty("java.home"), "bin", "java").toString());

		String classpath = System.getProperty("java.class.path");
		boolean verbose = "true".equalsIgnoreCase(System.getProperty("HYPER_VERBOSE"));

		java.util.List<String> cmd = new java.util.ArrayList<>(
				java.util.List.of(javaExe, "--enable-preview", "-Xms512m", "-Xmx4g", "-XX:+UseZGC"));

		if (!verbose) {
			// Write a temp JUL config that silences all logging in this node JVM.
			java.io.File q = java.io.File.createTempFile("hyper-quiet-", ".properties");
			q.deleteOnExit();
			try (java.io.PrintWriter pw = new java.io.PrintWriter(q)) {
				pw.println("handlers="); // no handlers -> nothing printed
				pw.println(".level=OFF");
			}
			cmd.add("-Djava.util.logging.config.file=" + q.getAbsolutePath());
		}

		cmd.addAll(java.util.List.of("-cp", classpath, "io.hyperstack4j.integration.NodeMain", nodeId,
				String.valueOf(port)));
		if (modelPath != null)
			cmd.add(modelPath);

		ProcessBuilder pb = new ProcessBuilder(cmd);
		pb.redirectErrorStream(false);
		// In quiet mode discard node stderr entirely; in verbose mode inherit it
		pb.redirectError(verbose ? ProcessBuilder.Redirect.INHERIT : ProcessBuilder.Redirect.DISCARD);

		Process proc = pb.start();
		log.info("Forked JVM for node [" + nodeId + "] PID=" + proc.pid());
		return proc;
	}

	/**
	 * Block until the process prints "READY:<nodeId>:..." or the timeout expires.
	 */
	private static void waitForReady(Process proc, String expectedNodeId) throws IOException, InterruptedException {

		long deadline = System.currentTimeMillis() + NODE_STARTUP_TIMEOUT_MS;
		try (BufferedReader reader = new BufferedReader(new InputStreamReader(proc.getInputStream()))) {

			String line;
			while ((line = reader.readLine()) != null) {
				log.fine("[" + expectedNodeId + "] stdout: " + line);
				if (line.startsWith("READY:" + expectedNodeId)) {
					return;
				}
				if (System.currentTimeMillis() > deadline) {
					throw new IOException("Node [" + expectedNodeId + "] did not become ready within "
							+ NODE_STARTUP_TIMEOUT_MS + " ms");
				}
				if (!proc.isAlive()) {
					throw new IOException("Node [" + expectedNodeId + "] process died before becoming ready (exit="
							+ proc.exitValue() + ")");
				}
			}
		}
		throw new IOException("Node [" + expectedNodeId + "] stdout closed without READY signal");
	}

	// ── Inner type ────────────────────────────────────────────────────────────

	private record NodeSpec(String nodeId, String host, int port, ProcessPipelineClient.ShardConfig shard) {
	}
}