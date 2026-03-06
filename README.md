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
    ├── Scheduler (continuous batching, virtual threads)
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
| `api` | OpenAPI 3.0 spec + generated JAX-RS interfaces + models. gRPC proto (inference.proto) for internal node communication |
| `registry` | Model Registry + Shard Loader (Hazelcast IMap, GGUF via DJL, IMQ seed scoring) |
| `coordinator` | Coordinator + Scheduler (continuous batching, Hazelcast CP leader election, Javalin REST) |
| `node` | Inference Node — runs on each GPU machine (JCuda, JCublas, gRPC server) |
| `kvcache` | KV Cache Manager — GPU tier (JCuda) + JVM heap tier (Caffeine). Prefix Trie for shared prompts |
| `tokenizer` | Tokenizer (DJL SentencePiece, chat template formatter) |
| `sampler` | Sampler — pure Java, zero deps (temperature, top-k, top-p, repetition penalty) |
| `health` | Health Monitor (Hazelcast membership events, JCuda GPU probes, Resilience4j circuit breakers) |

## API

The coordinator exposes a REST API (OpenAPI 3.0 spec at `api/src/main/resources/openapi.yaml`):

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
| Concurrency | Java 21 Virtual Threads |
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

## License

Apache 2.0
