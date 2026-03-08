package io.hyperstack4j.integration;

import java.util.logging.Logger;

/**
 * Entry point for a standalone node JVM process.
 *
 * Launched by ClusterHarness via ProcessBuilder — one JVM per node.
 * Listens on the given gRPC port and serves NodeService via EmbeddedNodeServer.
 *
 * Usage (ClusterHarness handles this automatically):
 *   java ... io.hyperstack4j.integration.NodeMain <nodeId> <port> [modelPath]
 *
 * When modelPath is supplied, EmbeddedNodeServer uses CpuForwardPassHandler
 * (real transformer math) instead of StubForwardPassHandler.
 *
 * Manual launch for debugging:
 *   mvn exec:java -pl integration \
 *       -Dexec.mainClass=io.hyperstack4j.integration.NodeMain \
 *       -Dexec.args="node-1 9092 /models/TinyLlama-1.1B-Chat-v1.0.Q4_K_M.gguf"
 */
public final class NodeMain {

    private static final Logger log = Logger.getLogger(NodeMain.class.getName());

    public static void main(String[] args) throws Exception {
        if (args.length < 2) {
            System.err.println("Usage: NodeMain <nodeId> <port> [modelPath]");
            System.exit(1);
        }

        String nodeId    = args[0];
        int    port      = Integer.parseInt(args[1]);
        String modelPath = args.length >= 3 ? args[2] : null;

        EmbeddedNodeServer server = new EmbeddedNodeServer(nodeId, port, modelPath);
        server.start();

        // Signal readiness to the parent process (ClusterHarness polls for this line)
        System.out.println("READY:" + nodeId + ":" + port);
        System.out.flush();

        log.info("Node [" + nodeId + "] running on port " + port + " — waiting for requests");
        server.blockUntilShutdown();
    }
}