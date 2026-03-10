package io.hyperstack4j.integration;

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
import io.hyperstack4j.node.ActivationCodec;
import io.hyperstack4j.node.ActivationDtype;
import io.hyperstack4j.node.InferencePipeline;

/**
 * InferencePipeline that fans a forward pass across N remote node processes
 * over gRPC, in pipeline order, with configurable activation compression.
 *
 * Flow (3-node example): Node-1: ForwardPass(tokens=[...], activation=empty,
 * dtype=X) → encoded activation₁ Node-2: ForwardPass(activation=activation₁,
 * dtype=X) → encoded activation₂ Node-3: ForwardPass(activation=activation₂,
 * dtype=X) → logits (isLastNode=true)
 *
 * Compression: {@code activationDtype} controls how float[] tensors are encoded
 * before each gRPC send and decoded after each gRPC receive. The dtype is
 * carried in the proto message so the receiving node always knows how to
 * decode.
 *
 * FLOAT32 → no compression (baseline, default) FLOAT16 → 2× reduction,
 * negligible accuracy loss (recommended for LAN) INT8 → 4× reduction, ~1%
 * relative error (for bandwidth-constrained nodes)
 *
 * Thread-safe — channels and dtype are immutable after construction.
 */
public final class ProcessPipelineClient implements InferencePipeline {

	private static final Logger log = Logger.getLogger(ProcessPipelineClient.class.getName());

	private final List<NodeStub> stubs;
	private final int vocabSize;
	private final ActivationDtype activationDtype;

	/**
	 * Construct with FLOAT32 (no compression) — preserves backward compatibility.
	 */
	public ProcessPipelineClient(List<NodeAddress> nodes, int vocabSize) {
		this(nodes, vocabSize, ActivationDtype.FLOAT32);
	}

	/**
	 * Construct with an explicit activation dtype.
	 *
	 * @param nodes           addresses of pipeline nodes in order
	 * @param vocabSize       size of the final logit vector
	 * @param activationDtype compression format for activation tensors
	 */
	public ProcessPipelineClient(List<NodeAddress> nodes, int vocabSize, ActivationDtype activationDtype) {
		this.vocabSize = vocabSize;
		this.activationDtype = activationDtype;
		this.stubs = nodes.stream().map(addr -> new NodeStub(addr.host(), addr.port())).toList();
		log.info("ProcessPipelineClient created — dtype=" + activationDtype + ", nodes=" + nodes.size());
	}

	@Override
	public float[] forward(String requestId, int[] tokens, int startPos) {
		if (tokens == null)
			throw new IllegalArgumentException("tokenIds cannot be null");
		// First node receives raw token IDs as a byte payload; dtype is set for the
		// response
		byte[] activation = intsToBytes(tokens);

		for (int i = 0; i < stubs.size(); i++) {
			NodeStub stub = stubs.get(i);

			ForwardRequest req = ForwardRequest.newBuilder().setRequestId(requestId).setModelId("stub-model")
					.setSequencePos(startPos).setActivation(ByteString.copyFrom(activation))
					.setDtype(toProto(activationDtype)).build();

			ForwardResponse response = stub.blockingStub.forwardPass(req);

			if (!response.getError().isEmpty()) {
				throw new RuntimeException("Node " + i + " forward pass failed: " + response.getError());
			}

			ActivationDtype responseDtype = fromProto(response.getDtype());
			byte[] rawBytes = response.getActivation().toByteArray();

			if (response.getIsLastNode()) {
				// Final node always returns logits as plain FLOAT32
				return ActivationCodec.decode(rawBytes, ActivationDtype.FLOAT32);
			}

			// Intermediate node: decode the compressed activation, then re-encode for
			// the next hop using the configured dtype (enables per-hop dtype later)
			float[] decoded = ActivationCodec.decode(rawBytes, responseDtype);
			activation = ActivationCodec.encode(decoded, activationDtype);
		}

		throw new IllegalStateException("Pipeline completed without a last-node response");
	}

	@Override
	public int vocabSize() {
		return vocabSize;
	}

	/** Returns the activation dtype this client is configured to use. */
	public ActivationDtype activationDtype() {
		return activationDtype;
	}

	/**
	 * Load a shard assignment onto each node. Call once before any forward pass.
	 */
	public void loadShards(List<ShardConfig> shards) {
		if (shards.size() != stubs.size())
			throw new IllegalArgumentException("shards.size() must equal nodes.size()");

		for (int i = 0; i < stubs.size(); i++) {
			ShardConfig shard = shards.get(i);
			LoadShardRequest req = LoadShardRequest.newBuilder().setModelId("stub-model")
					.setStartLayer(shard.startLayer()).setEndLayer(shard.endLayer())
					.setHasEmbeddings(shard.hasEmbeddings()).setHasOutputProjection(shard.hasOutputProjection())
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
		final ManagedChannel channel;
		final NodeServiceGrpc.NodeServiceBlockingStub blockingStub;

		NodeStub(String host, int port) {
			this.channel = ManagedChannelBuilder.forAddress(host, port).usePlaintext().build();
			this.blockingStub = NodeServiceGrpc.newBlockingStub(channel);
		}
	}

	public record NodeAddress(String host, int port) {
	}

	public record ShardConfig(int startLayer, int endLayer, boolean hasEmbeddings, boolean hasOutputProjection) {
	}

	// ── Helpers ───────────────────────────────────────────────────────────────

	private static byte[] intsToBytes(int[] ints) {
		java.nio.ByteBuffer buf = java.nio.ByteBuffer.allocate(ints.length * 4);
		for (int v : ints)
			buf.putInt(v);
		return buf.array();
	}

	static io.hyperstack4j.api.grpc.ActivationDtype toProto(ActivationDtype dtype) {
		return switch (dtype) {
		case FLOAT32 -> io.hyperstack4j.api.grpc.ActivationDtype.FLOAT32;
		case FLOAT16 -> io.hyperstack4j.api.grpc.ActivationDtype.FLOAT16;
		case INT8 -> io.hyperstack4j.api.grpc.ActivationDtype.INT8;
		};
	}

	static ActivationDtype fromProto(io.hyperstack4j.api.grpc.ActivationDtype proto) {
		return switch (proto) {
		case FLOAT16 -> ActivationDtype.FLOAT16;
		case INT8 -> ActivationDtype.INT8;
		default -> ActivationDtype.FLOAT32; // FLOAT32 and UNRECOGNIZED
		};
	}
}