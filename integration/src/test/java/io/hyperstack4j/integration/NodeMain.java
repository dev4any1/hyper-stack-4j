package io.hyperstack4j.integration;

import java.util.logging.Logger;

/**
 * Entry point for a standalone node JVM process.
 *
 * Launched by ClusterHarness via ProcessBuilder — one JVM per node.
 * Listens on the given gRPC port and serves NodeService via EmbeddedNodeServer.
 *
 * Usage (ClusterHarness handles this automatically):
 *   java ... io.hyperstack4j.integration.NodeMain <nodeId> <port>
 *
 * Manual launch for debugging:
 *   mvn exec:java -pl integration \
 *       -Dexec.mainClass=io.hyperstack4j.integration.NodeMain \
 *       -Dexec.args="node-1 9092"
 */
public final class NodeMain {

    private static final Logger log = Logger.getLogger(NodeMain.class.getName());

    public static void main(String[] args) throws Exception {
        if (args.length < 2) {
            System.err.println("Usage: NodeMain <nodeId> <port>");
            System.exit(1);
        }

        String nodeId = args[0];
        int    port   = Integer.parseInt(args[1]);

        EmbeddedNodeServer server = new EmbeddedNodeServer(nodeId, port);
        server.start();

        // Signal readiness to the parent process (ClusterHarness polls for this line)
        System.out.println("READY:" + nodeId + ":" + port);
        System.out.flush();

        log.info("Node [" + nodeId + "] running on port " + port + " — waiting for requests");
        server.blockUntilShutdown();
    }
}
