package io.hyperstack4j.integration;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.concurrent.TimeUnit;
import java.util.logging.Logger;

import io.grpc.Server;
import io.grpc.ServerBuilder;
import io.grpc.stub.StreamObserver;
import io.hyperstack4j.api.grpc.ForwardResponse;
import io.hyperstack4j.api.grpc.LoadShardRequest;
import io.hyperstack4j.api.grpc.LoadShardResponse;
import io.hyperstack4j.api.grpc.NodeServiceGrpc;
import io.hyperstack4j.api.grpc.NodeStatusRequest;
import io.hyperstack4j.api.grpc.NodeStatusResponse;
import io.hyperstack4j.api.grpc.UnloadShardRequest;
import io.hyperstack4j.api.grpc.UnloadShardResponse;
import io.hyperstack4j.kvcache.CpuKVCache;
import io.hyperstack4j.kvcache.GpuKVCache;
import io.hyperstack4j.kvcache.KVCacheManager;
import io.hyperstack4j.kvcache.LayerRange;
import io.hyperstack4j.node.ForwardResult;
import io.hyperstack4j.node.ShardContext;
import io.hyperstack4j.node.StubForwardPassHandler;
import io.hyperstack4j.registry.ShardAssignment;

/**
 * Minimal gRPC NodeService implementation backed by StubForwardPassHandler.
 *
 * Used by ThreeNodeClusterIT — each node JVM runs one of these.
 * No GPU, no real weights — deterministic stub responses only.
 */
public final class EmbeddedNodeServer {

    private static final Logger log = Logger.getLogger(EmbeddedNodeServer.class.getName());

    private final String nodeId;
    private final int    port;
    private final Server grpcServer;

    // TinyLlama-1.1B shape constants
    static final int VOCAB_SIZE   = 32_000;
    static final int HIDDEN_DIM   = 2_048;
    static final int NUM_HEADS    = 32;
    static final int TOTAL_LAYERS = 22;

    public EmbeddedNodeServer(String nodeId, int port) {
        this.nodeId = nodeId;
        this.port   = port;
        this.grpcServer = ServerBuilder.forPort(port)
                .addService(new NodeServiceImpl(nodeId))
                .build();
    }

    public void start() throws IOException {
        grpcServer.start();
        log.info("Node [" + nodeId + "] gRPC server started on port " + port);
        Runtime.getRuntime().addShutdownHook(Thread.ofVirtual().unstarted(() -> {
            try { stop(); } catch (InterruptedException e) { Thread.currentThread().interrupt(); }
        }));
    }

    public void stop() throws InterruptedException {
        if (grpcServer != null) grpcServer.shutdown().awaitTermination(5, TimeUnit.SECONDS);
    }

    public void blockUntilShutdown() throws InterruptedException {
        if (grpcServer != null) grpcServer.awaitTermination();
    }

    // ── gRPC service impl ─────────────────────────────────────────────────────

    private static final class NodeServiceImpl extends NodeServiceGrpc.NodeServiceImplBase {

        private static final long NODE_VRAM_BUDGET = 512L * 1024 * 1024; // 512MB per node cache budget

        private final String                 nodeId;
        private final StubForwardPassHandler handler = new StubForwardPassHandler();
        private volatile ShardContext        context;
        private volatile KVCacheManager      kvCache;  // layer-range-aware, set on loadShard

        NodeServiceImpl(String nodeId) {
            this.nodeId  = nodeId;
            this.context = buildDefaultContext();
            // Default: unbounded — replaced with layer-scoped cache on loadShard
            this.kvCache = new KVCacheManager(
                    new GpuKVCache(NODE_VRAM_BUDGET),
                    new CpuKVCache(256)
            );
        }

        @Override
        public void forwardPass(
                io.hyperstack4j.api.grpc.ForwardRequest request,   // fully qualified — avoids clash with node.ForwardRequest
                StreamObserver<ForwardResponse> responseObserver) {
            try {
                float[] inputActivations = bytesToFloats(request.getActivation().toByteArray());

                // Use node.ForwardRequest (the record, not the proto message)
                io.hyperstack4j.node.ForwardRequest nodeReq =
                        inputActivations.length == 0
                                ? io.hyperstack4j.node.ForwardRequest.withTokens(
                                        request.getRequestId(), new int[]{}, request.getSequencePos())
                                : io.hyperstack4j.node.ForwardRequest.withActivations(
                                        request.getRequestId(), inputActivations, request.getSequencePos());

                ForwardResult result = handler.forward(nodeReq, context);

                float[] outputFloats = result.isFinalNode() ? result.logits() : result.activations();

                ForwardResponse response = ForwardResponse.newBuilder()
                        .setRequestId(request.getRequestId())
                        .setActivation(com.google.protobuf.ByteString.copyFrom(floatsToBytes(outputFloats)))
                        .setIsLastNode(result.isFinalNode())
                        .build();

                responseObserver.onNext(response);
                responseObserver.onCompleted();

            } catch (Exception e) {
                responseObserver.onNext(ForwardResponse.newBuilder()
                        .setRequestId(request.getRequestId())
                        .setError(e.getMessage() != null ? e.getMessage() : e.getClass().getName())
                        .build());
                responseObserver.onCompleted();
            }
        }

        @Override
        public void loadShard(LoadShardRequest request,
                              StreamObserver<LoadShardResponse> responseObserver) {
            ShardAssignment assignment = new ShardAssignment(
                    nodeId, "localhost", 0,
                    request.getStartLayer(), request.getEndLayer(),
                    request.getHasEmbeddings(), request.getHasOutputProjection()
            );
            context = ShardContext.from(assignment, VOCAB_SIZE, HIDDEN_DIM, NUM_HEADS);

            // Rebuild KVCacheManager scoped to this node's layer range
            LayerRange range = LayerRange.of(request.getStartLayer(), request.getEndLayer());
            kvCache = new KVCacheManager(
                    new GpuKVCache(NODE_VRAM_BUDGET),
                    new CpuKVCache(256),
                    range
            );
            log.info("Node [" + nodeId + "] KVCache scoped to " + range);

            responseObserver.onNext(LoadShardResponse.newBuilder()
                    .setSuccess(true)
                    .setMessage("Stub shard loaded layers "
                            + request.getStartLayer() + "–" + request.getEndLayer())
                    .build());
            responseObserver.onCompleted();
        }

        @Override
        public void unloadShard(UnloadShardRequest request,
                                StreamObserver<UnloadShardResponse> responseObserver) {
            context = buildDefaultContext();
            responseObserver.onNext(UnloadShardResponse.newBuilder().setSuccess(true).build());
            responseObserver.onCompleted();
        }

        @Override
        public void getNodeStatus(NodeStatusRequest request,
                                  StreamObserver<NodeStatusResponse> responseObserver) {
            responseObserver.onNext(NodeStatusResponse.newBuilder()
                    .setNodeId(nodeId)
                    .setStatus("READY")
                    .setVramTotalBytes(4L * 1024 * 1024 * 1024)
                    .setVramFreeBytes(3L  * 1024 * 1024 * 1024)
                    .setSeedScore(1.0)
                    .build());
            responseObserver.onCompleted();
        }

        // ── helpers ───────────────────────────────────────────────────────────

        private static ShardContext buildDefaultContext() {
            ShardAssignment full = new ShardAssignment(
                    "default", "localhost", 0,
                    0, TOTAL_LAYERS, true, true
            );
            return ShardContext.from(full, VOCAB_SIZE, HIDDEN_DIM, NUM_HEADS);
        }

        private static float[] bytesToFloats(byte[] bytes) {
            if (bytes == null || bytes.length == 0) return new float[0];
            ByteBuffer buf = ByteBuffer.wrap(bytes);
            float[] floats = new float[bytes.length / 4];
            for (int i = 0; i < floats.length; i++) floats[i] = buf.getFloat();
            return floats;
        }

        private static byte[] floatsToBytes(float[] floats) {
            ByteBuffer buf = ByteBuffer.allocate(floats.length * 4);
            for (float f : floats) buf.putFloat(f);
            return buf.array();
        }
    }
}
