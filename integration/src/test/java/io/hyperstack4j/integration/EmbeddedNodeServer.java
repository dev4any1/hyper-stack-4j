package io.hyperstack4j.integration;

import java.io.IOException;
import java.nio.file.Path;
import java.util.concurrent.TimeUnit;
import java.util.logging.Logger;

import io.grpc.Server;
import io.grpc.ServerBuilder;
import io.grpc.stub.StreamObserver;
import io.hyperstack4j.api.grpc.ActivationDtype;
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
import io.hyperstack4j.node.ActivationCodec;
import io.hyperstack4j.node.CpuForwardPassHandler;
import io.hyperstack4j.node.ForwardPassHandler;
import io.hyperstack4j.node.ForwardResult;
import io.hyperstack4j.node.ShardContext;
import io.hyperstack4j.node.CyclicForwardPassHandler;
import io.hyperstack4j.registry.ShardAssignment;

/**
 * Minimal gRPC NodeService backed by CyclicForwardPassHandler.
 *
 * Used by ThreeNodeClusterIT — each node JVM runs one of these.
 * No GPU, no real weights — deterministic stub responses only.
 *
 * Activation compression:
 *   Each ForwardRequest carries a dtype field that tells this node how to decode
 *   the incoming activation bytes. The response activation (for intermediate nodes)
 *   is compressed using the same dtype. Final-node logits are always FLOAT32.
 */
public final class EmbeddedNodeServer {

    private static final Logger log = Logger.getLogger(EmbeddedNodeServer.class.getName());

    private final String nodeId;
    private final int    port;
    private final Server grpcServer;

    // TinyLlama-1.1B shape constants (used when no model file is supplied)
    static final int VOCAB_SIZE   = 32_000;
    static final int HIDDEN_DIM   = 2_048;
    static final int NUM_HEADS    = 32;
    static final int TOTAL_LAYERS = 22;

    /** Stub mode (no real model — used by integration tests). */
    public EmbeddedNodeServer(String nodeId, int port) {
        this(nodeId, port, null);
    }

    /**
     * Real-model mode.
     * @param modelPath  path to a GGUF file, or null to fall back to stub mode.
     */
    public EmbeddedNodeServer(String nodeId, int port, String modelPath) {
        this.nodeId = nodeId;
        this.port   = port;
        this.grpcServer = ServerBuilder.forPort(port)
                .addService(new NodeServiceImpl(nodeId, modelPath))
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

        private static final long NODE_VRAM_BUDGET = 512L * 1024 * 1024;

        private final String              nodeId;
        private final String              modelPath; // null = stub mode
        private volatile ForwardPassHandler handler;
        private volatile ShardContext       context;
        private volatile KVCacheManager     kvCache;

        NodeServiceImpl(String nodeId, String modelPath) {
            this.nodeId    = nodeId;
            this.modelPath = modelPath;
            this.handler   = new CyclicForwardPassHandler(); // replaced in loadShard() when real model
            this.context   = buildDefaultContext();
            this.kvCache   = new KVCacheManager(
                    new GpuKVCache(NODE_VRAM_BUDGET),
                    new CpuKVCache(256)
            );
        }

        @Override
        public void forwardPass(
                io.hyperstack4j.api.grpc.ForwardRequest request,
                StreamObserver<ForwardResponse> responseObserver) {
            try {
                // ── Decode incoming activation ──────────────────────────────
                io.hyperstack4j.node.ActivationDtype inDtype = fromProto(request.getDtype());
                byte[] rawBytes = request.getActivation().toByteArray();
                float[] inputActivations = ActivationCodec.decode(rawBytes, inDtype);

                // Build the domain-model ForwardRequest (not the proto one)
                io.hyperstack4j.node.ForwardRequest nodeReq =
                        inputActivations.length == 0
                                ? io.hyperstack4j.node.ForwardRequest.withTokens(
                                        request.getRequestId(), new int[]{}, request.getSequencePos())
                                : io.hyperstack4j.node.ForwardRequest.withActivations(
                                        request.getRequestId(), inputActivations, request.getSequencePos());

                ForwardResult result = handler.forward(nodeReq, context);

                // ── Encode outgoing activation ──────────────────────────────
                float[] outputFloats = result.isFinalNode() ? result.logits() : result.activations();

                // Final node always returns plain FLOAT32 logits (no loss allowed on vocab)
                io.hyperstack4j.node.ActivationDtype outDtype = result.isFinalNode()
                        ? io.hyperstack4j.node.ActivationDtype.FLOAT32
                        : inDtype;

                byte[] encodedOutput = ActivationCodec.encode(outputFloats, outDtype);

                ForwardResponse response = ForwardResponse.newBuilder()
                        .setRequestId(request.getRequestId())
                        .setActivation(com.google.protobuf.ByteString.copyFrom(encodedOutput))
                        .setIsLastNode(result.isFinalNode())
                        .setDtype(toProto(outDtype))
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
            ShardContext newCtx = ShardContext.from(assignment, VOCAB_SIZE, HIDDEN_DIM, NUM_HEADS);

            String msg;
            if (modelPath != null) {
                try {
                    log.info("Loading real model from: " + modelPath);
                    log.info("Shard context: layers " + request.getStartLayer() + "-" + request.getEndLayer()
                            + " embeddings=" + request.getHasEmbeddings() 
                            + " outputProj=" + request.getHasOutputProjection());
                    handler = CpuForwardPassHandler.load(Path.of(modelPath), newCtx);
                    msg = "Real shard loaded (CpuForwardPassHandler) layers "
                            + request.getStartLayer() + "–" + request.getEndLayer();
                    log.info(msg);
                } catch (Exception e) {
                    log.severe("FAILED to load real model: " + e.getMessage());
                    e.printStackTrace();  // <-- Print full stack trace
                    log.warning("Falling back to stub handler");
                    handler = new CyclicForwardPassHandler();
                    msg = "Stub shard (model load failed: " + e.getMessage() + ") layers "
                            + request.getStartLayer() + "–" + request.getEndLayer();
                }
            } else {
                handler = new CyclicForwardPassHandler();
                msg = "Stub shard loaded layers "
                        + request.getStartLayer() + "–" + request.getEndLayer();
            }
            context = newCtx;

            LayerRange range = LayerRange.of(request.getStartLayer(), request.getEndLayer());
            kvCache = new KVCacheManager(
                    new GpuKVCache(NODE_VRAM_BUDGET),
                    new CpuKVCache(256),
                    range
            );
            log.info("Node [" + nodeId + "] KVCache scoped to " + range);

            responseObserver.onNext(LoadShardResponse.newBuilder()
                    .setSuccess(true)
                    .setMessage(msg)
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

        private static io.hyperstack4j.node.ActivationDtype fromProto(ActivationDtype proto) {
            return switch (proto) {
                case FLOAT16 -> io.hyperstack4j.node.ActivationDtype.FLOAT16;
                case INT8    -> io.hyperstack4j.node.ActivationDtype.INT8;
                default      -> io.hyperstack4j.node.ActivationDtype.FLOAT32;
            };
        }

        private static ActivationDtype toProto(io.hyperstack4j.node.ActivationDtype dtype) {
            return switch (dtype) {
                case FLOAT32 -> ActivationDtype.FLOAT32;
                case FLOAT16 -> ActivationDtype.FLOAT16;
                case INT8    -> ActivationDtype.INT8;
            };
        }
    }
}