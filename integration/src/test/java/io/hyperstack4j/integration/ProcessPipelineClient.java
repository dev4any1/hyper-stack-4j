package io.hyperstack4j.integration;

import java.nio.ByteBuffer;
import java.util.List;
import java.util.concurrent.TimeUnit;
import java.util.logging.Logger;

import com.google.protobuf.ByteString;

import io.grpc.ManagedChannel;
import io.grpc.ManagedChannelBuilder;
import io.hyperstack4j.api.grpc.ForwardRequest;
import io.hyperstack4j.api.grpc.ForwardResponse;
import io.hyperstack4j.api.grpc.LoadShardRequest;
import io.hyperstack4j.api.grpc.LoadShardResponse;
import io.hyperstack4j.api.grpc.NodeServiceGrpc;
import io.hyperstack4j.node.InferencePipeline;

/**
 * InferencePipeline implementation that fans a forward pass across N remote
 * node processes over gRPC, in pipeline order.
 *
 * Used by ThreeNodeClusterIT to wire the coordinator's GenerationLoop to the
 * three forked node JVMs.
 *
 * Flow:
 *   Node-1: ForwardPass(tokens=[...], activation=empty)  → activation₁
 *   Node-2: ForwardPass(activation=activation₁)           → activation₂
 *   Node-3: ForwardPass(activation=activation₂)           → logits  (isLastNode=true)
 *
 * Thread-safe — channels are immutable after construction.
 */
public final class ProcessPipelineClient implements InferencePipeline {

    private static final Logger log = Logger.getLogger(ProcessPipelineClient.class.getName());

    private final List<NodeStub>  stubs;
    private final int             vocabSize;

    public ProcessPipelineClient(List<NodeAddress> nodes, int vocabSize) {
        this.vocabSize = vocabSize;
        this.stubs = nodes.stream()
                .map(addr -> new NodeStub(addr.host(), addr.port()))
                .toList();
    }
    private static byte[] intsToBytes(int[] ints) {
        ByteBuffer buf = ByteBuffer.allocate(ints.length * 4);
        for (int i : ints) buf.putInt(i);
        return buf.array();
    }
    @Override
    public float[] forward(String requestId, int[] tokens, int startPos) {
    	byte[] activation = intsToBytes(tokens);

        for (int i = 0; i < stubs.size(); i++) {
            NodeStub stub = stubs.get(i);

            ForwardRequest.Builder req = ForwardRequest.newBuilder()
                    .setRequestId(requestId)
                    .setModelId("stub-model")
                    .setSequencePos(startPos)
                    .setActivation(ByteString.copyFrom(activation));

            ForwardResponse response = stub.blockingStub.forwardPass(req.build());

            if (!response.getError().isEmpty()) {
                throw new RuntimeException("Node " + i + " forward pass failed: " + response.getError());
            }

            activation = response.getActivation().toByteArray();

            if (response.getIsLastNode()) {
                return bytesToFloats(activation);
            }
        }

        throw new IllegalStateException("Pipeline completed without a last-node response");
    }

    @Override
    public int vocabSize() {
        return vocabSize;
    }

    /**
     * Load a shard assignment onto each node — tells each node which layer range it owns.
     * Call once before any forward pass.
     */
    public void loadShards(List<ShardConfig> shards) {
        if (shards.size() != stubs.size()) {
            throw new IllegalArgumentException("shards.size() must equal nodes.size()");
        }
        for (int i = 0; i < stubs.size(); i++) {
            ShardConfig shard = shards.get(i);
            LoadShardRequest req = LoadShardRequest.newBuilder()
                    .setModelId("stub-model")
                    .setStartLayer(shard.startLayer())
                    .setEndLayer(shard.endLayer())
                    .setHasEmbeddings(shard.hasEmbeddings())
                    .setHasOutputProjection(shard.hasOutputProjection())
                    .build();
            LoadShardResponse response = stubs.get(i).blockingStub.loadShard(req);
            log.info("Node " + i + " shard load: " + response.getMessage());
        }
    }

    /** Shut down all gRPC channels cleanly. */
    public void shutdown() throws InterruptedException {
        for (NodeStub stub : stubs) {
            stub.channel.shutdown().awaitTermination(5, TimeUnit.SECONDS);
        }
    }

    // ── Inner types ───────────────────────────────────────────────────────────

    private static final class NodeStub {
        final ManagedChannel              channel;
        final NodeServiceGrpc.NodeServiceBlockingStub blockingStub;

        NodeStub(String host, int port) {
            this.channel = ManagedChannelBuilder.forAddress(host, port)
                    .usePlaintext()
                    .build();
            this.blockingStub = NodeServiceGrpc.newBlockingStub(channel);
        }
    }

    public record NodeAddress(String host, int port) {}

    public record ShardConfig(int startLayer, int endLayer,
                               boolean hasEmbeddings, boolean hasOutputProjection) {}

    // ── Serialization helpers ─────────────────────────────────────────────────

    private static float[] bytesToFloats(byte[] bytes) {
        ByteBuffer buf = ByteBuffer.wrap(bytes);
        float[] floats = new float[bytes.length / 4];
        for (int i = 0; i < floats.length; i++) floats[i] = buf.getFloat();
        return floats;
    }
}
