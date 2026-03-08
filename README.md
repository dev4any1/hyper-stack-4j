# 🦙 hyper-stack-4j

> Distributed Java-native LLM inference engine — commodity GPU cluster, no Python, no GIL, no Spring.

## Vision

Run large language models across a network of **affordable commodity GPUs** — replacing the need for a single expensive high-VRAM card with a cluster of machines you already have.

**16 × 4GB GPUs = 64GB total VRAM at a fraction of the cost.**

## Architecture

```
[Client] REST (Javalin) or gRPC streaming
    ↓
[Load Balancer]  HAProxy / Nginx
    ↓                         ↓
[Coordinator 1]         [Coordinator 2]
   LEADER                  STANDBY
    │
    ├── Tokenizer (DJL SentencePiece)
    ├── Scheduler (CompletableFuture, virtual threads)
    ├── Sampler (pure Java)
    ├── PrefixCache (Trie)
    └── InferencePipeline
              │
              │  gRPC       (data plane  — activations)
              │  Hazelcast  (control plane — state, events)
              │
    =============================================
    ||       10/25GbE RDMA Network            ||
    =============================================
         |          |          |          |
    [Node 1]   [Node 2]   [Node 3]  ... [Node 16]
    Layer 0-1  Layer 2-3  Layer 4-5     Layer N
    + Embed    GPU shard  GPU shard     + Output proj
```

## Modules

| Module | Responsibility |
|--------|---------------|
| `api` | OpenAPI 3.0 spec + generated JAX-RS interfaces + models. gRPC proto (`inference.proto`) for internal node communication, including `ActivationDtype` enum |
| `registry` | Model Registry + Shard Planner (Hazelcast IMap, GGUF via DJL, IMQ seed scoring) |
| `coordinator` | Coordinator + Scheduler (reactive `CompletableFuture`, Hazelcast CP leader election, Javalin REST) |
| `node` | Inference Node — runs on each GPU machine (JCuda, JCublas, gRPC server). `ActivationDtype` + `ActivationCodec` — pure-Java compression at the gRPC boundary |
| `kvcache` | KV Cache Manager — GPU tier (JCuda) + JVM heap tier (Caffeine). Prefix Trie for shared prompts |
| `tokenizer` | Tokenizer (DJL SentencePiece, chat template formatter) |
| `sampler` | Sampler — pure Java, zero deps (temperature, top-k, top-p, repetition penalty) |
| `health` | Health Monitor (Hazelcast membership events, JCuda GPU probes, Resilience4j circuit breakers) |
| `integration` | Multi-JVM cluster integration tests — forks real node processes, exercises full gRPC pipeline including FLOAT16/INT8 compression |

## Activation Compression

Activation tensors shipped between nodes are the primary network bottleneck.
At 70B scale (hidden_dim=8192, seq_len=4096) each hop costs ~64 MB over 10GbE.

`ProcessPipelineClient` accepts an `ActivationDtype` to compress activations
before each gRPC send and decompress after each receive:

| Dtype | Size/element | Relative error | Transfer (70B, 10GbE) |
|-------|-------------|---------------|-----------------------|
| `FLOAT32` | 4 B | lossless | ~51 ms/hop |
| `FLOAT16` | 2 B | ~0.1% | ~26 ms/hop |
| `INT8`    | 1 B + 4 B scale | ~1% | ~13 ms/hop |

```java
// Default — no compression
ProcessPipelineClient pipeline = new ProcessPipelineClient(nodes, vocabSize);

// FLOAT16 — recommended for LAN clusters
ProcessPipelineClient pipeline = new ProcessPipelineClient(nodes, vocabSize, ActivationDtype.FLOAT16);

// INT8 — for bandwidth-constrained community nodes
ProcessPipelineClient pipeline = new ProcessPipelineClient(nodes, vocabSize, ActivationDtype.INT8);
```

The dtype is encoded in each `ForwardRequest` proto message (`dtype` field 9)
so each node always knows how to decode its input. Final-node logits are always
returned as `FLOAT32` — no precision loss on the vocabulary distribution.

`ActivationCodec` is pure Java (no JNI, no external deps). FLOAT16 uses manual
IEEE 754 half-precision bit manipulation; INT8 uses symmetric quantisation with a
float32 scale prefix (`max(|activations|) / 127`).

## Scheduler — Reactive Design

`RequestScheduler` dispatches every request on its own Java 21 Virtual Thread and exposes a fully reactive API via `CompletableFuture`.

```java
// Streaming — returns immediately, tokens delivered via TokenConsumer callback
CompletableFuture<GenerationResult> future = scheduler.submit(request, consumer);

// Blocking — caller waits only on its own future, never on other requests
GenerationResult result = scheduler.submitAndWait(request);
```

**How `submitAndWait` works:**

```
1.  CompletableFuture<GenerationResult> created and registered in ConcurrentHashMap
2.  request.offer()'d into PriorityBlockingQueue                    (HIGH priority first)
3.  Virtual thread spawned — runs generate(), calls future.complete(result) on finish
4.  Caller blocks on future.join() — wakes up exactly when its own request is done
```

N concurrent callers each block on **independent futures** — there is no shared lock or sequential bottleneck between requests. Exceptions surface via `future.completeExceptionally(e)` rather than being silently swallowed.

When the queue is full, `submit()` throws `QueueFullException` (with a `retryAfterSeconds` hint), which the REST layer translates to **HTTP 503 + Retry-After**.

## Shard Planner — Fair Layer Distribution

`ShardPlanner` guarantees every eligible node receives **at least one layer**, regardless of VRAM headroom. When assigning layers to a node, the planner caps its allocation to leave at least one layer for each remaining node:

```
maxLayers = min(layersFit, remainingLayers − (remainingNodes − 1))
```

This prevents a large-VRAM node from consuming all remaining layers and leaving later nodes empty-handed.

## API

The coordinator exposes a REST API (OpenAPI 3.0 spec at `api/src/main/resources/openapi.yaml`), implemented by `InferenceApiServer` (Javalin 6):

```
POST   /v1/inference          — blocking inference
POST   /v1/inference/stream   — SSE token streaming
POST   /v1/models             — load model into cluster
GET    /v1/models             — list loaded models
GET    /v1/models/{modelId}   — model status + shard map
DELETE /v1/models/{modelId}   — unload model
GET    /v1/cluster/health     — cluster health overview
GET    /v1/cluster/nodes      — all node statuses
GET    /v1/cluster/shardmap   — current layer assignments
```

**Streaming pipeline — how it works end to end:**

```
Client → POST /v1/inference/stream
         ↓
InferenceApiServer parses body, sets SSE headers
         ↓
scheduler.submit(request, SseTokenConsumer)  ← returns CompletableFuture
         ↓
Generation virtual thread: GenerationLoop.generate()
   each token → TokenConsumer.onToken(piece, tokenId, position)
              → SseTokenConsumer writes:  data: {"token":"Hello","tokenId":9906,"isComplete":false}
              → flushed to HTTP response immediately
         ↓
Generation complete → consumer.sendComplete("stop")
                    → data: {"token":"","tokenId":0,"isComplete":true,"finishReason":"stop"}
         ↓
Client reads the SSE stream token by token
```

`SseTokenConsumer` is decoupled from Javalin via a `SseEmitter` functional interface — in production it wraps `writer::write`; in tests it wraps `list::add`.

**Error codes:**

| Status | Meaning |
|--------|---------|
| 400 | Missing or empty messages |
| 404 | Model not found |
| 429 | Scheduler queue full (`QueueFullException`) |
| 503 | Model not loaded / cluster unavailable |
| 500 | Unexpected inference error |

Internal node-to-node communication uses gRPC (`InferenceService`, `NodeService`, `RegistryService`).

## KV Cache

Two tiers, RAM only — no disk IO:

```
Tier 1  GPU VRAM     JCuda CudaBuffer   — hot active sequences
Tier 2  JVM heap     Caffeine           — warm sequences, bounded by -Xmx
```

Caffeine uses W-TinyLFU eviction. Size is configured via `kv-cache.cpu.max-bytes` in `cluster-config.yaml`.
Prefix caching via a Trie structure — shared system prompts computed once and reused across requests.

## Requirements

- **JDK 21+**
- **Maven 3.9+**
- **CUDA 12.x** (on GPU nodes)
- **10GbE networking** recommended

## Build

```bash
mvn clean install -T 1C
```

## Testing

The project has two test layers:

**Unit / module tests** — run on every build, no network:
```bash
mvn test
```

**Integration tests** — `maven-failsafe-plugin`, enabled with `mvn verify`:
```bash
# Full suite — forks 3 real JVM node processes, exercises live gRPC pipeline
mvn verify -pl integration

# In-process only (fast, zero network, uses LocalInferencePipeline)
mvn verify -pl integration -Dit.test=InProcessClusterIT

# Skip ITs entirely
mvn verify -DskipITs
```

**Integration test architecture** (`integration` module):

```
ThreeNodeClusterIT
  └── ClusterHarness.start()
        ├── ProcessBuilder → NodeMain JVM #1  (port 19092, -Xmx4g -XX:+UseZGC)
        ├── ProcessBuilder → NodeMain JVM #2  (port 19093, -Xmx4g -XX:+UseZGC)
        └── ProcessBuilder → NodeMain JVM #3  (port 19094, -Xmx4g -XX:+UseZGC)
  └── ProcessPipelineClient  (gRPC channels to all 3 nodes)
  └── GenerationLoop + RequestScheduler  (coordinator JVM, -Xmx2g)
```

Memory budget for a 16 GB host: 3 × 4 GB nodes + 2 GB coordinator + 2 GB OS = 16 GB.

**Recommended stub model for local testing:** `TinyLlama-1.1B-Chat-v1.0.Q4_K_M.gguf` (~670 MB), split 8/7/7 layers across 3 nodes.

## Run

```bash
# Start coordinator
java --enable-preview -jar coordinator/target/coordinator.jar \
     --config cluster-config.yaml

# Start inference node (on each GPU machine)
java --enable-preview -jar node/target/node.jar \
     --config cluster-config.yaml \
     --device-id=0

# Load a model
curl -X POST http://coordinator:8080/v1/models \
     -H 'Content-Type: application/json' \
     -d '{"modelId": "llama3-8b", "path": "/models/llama3-8b.Q4_K_M.gguf", "architecture": "llama3"}'

# Chat — REST streaming (SSE)
curl -X POST http://coordinator:8080/v1/inference/stream \
     -H 'Content-Type: application/json' \
     -d '{"messages": [{"role": "user", "content": "Hello!"}]}'

# Chat — gRPC streaming
grpcurl -d '{"messages": [{"role": "user", "content": "Hello!"}]}' \
        coordinator:9090 io.hyperstack4j.api.grpc.InferenceService/InferStream
```

## Technology Stack

| Concern | Technology |
|---------|-----------| 
| Language | Java 21 |
| Build | Maven (multi-module) |
| GPU compute | JCuda / JCublas 12.x |
| Distributed state | Hazelcast 5.x |
| Leader election | Hazelcast CP FencedLock |
| Internal data plane | gRPC + Protocol Buffers |
| REST API server | Javalin 6.x (lightweight, ~1MB, no framework) |
| REST API spec | OpenAPI 3.0 — jaxrs-spec generator |
| Concurrency | Java 21 Virtual Threads + `CompletableFuture` |
| KV Cache L1 | JCuda CudaBuffer (GPU VRAM) |
| KV Cache L2 | Caffeine (JVM heap, W-TinyLFU, bounded by Xmx) |
| Circuit breaker | Resilience4j |
| Metrics | Micrometer + Prometheus (exposed via JDK HttpServer) |
| Tokenizer | DJL SpTokenizer (SentencePiece JNI) |
| Weight format | GGUF |
| Sampler | Pure Java, zero external deps |

## Notable Design Decisions

- **No Spring Boot** — too heavy. Javalin for REST, JDK HttpServer for metrics scrape endpoint.
- **No disk KV cache** — all cache in RAM (GPU VRAM + JVM heap). Caffeine evicts cleanly when Xmx is reached.
- **No OHC / Ehcache / Chronicle Map** — all carry dead transitive dependencies (NetBeans repo). Caffeine is sufficient.
- **Pipeline parallelism** over tensor parallelism — LAN-friendly, no InfiniBand required.
- **Separate data plane (gRPC) from control plane (Hazelcast)** — mirrors Kafka/Kubernetes design.
- **Reactive scheduler** — `submit()` returns `CompletableFuture<GenerationResult>`. N concurrent callers are fully independent; no sequential bottleneck between requests.
- **Fair shard planning** — `ShardPlanner` caps each node's layer allocation to guarantee every eligible node participates, preventing VRAM-rich nodes from monopolising the pipeline.

## License

Apache 2.0