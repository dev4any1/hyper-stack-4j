# 🦙 hyper-stack-4j

> Distributed Java-native LLM inference engine — commodity GPU cluster, no Python, no GIL.

## Vision

Run large language models across a network of **affordable commodity GPUs** — replacing the need for a single expensive high-VRAM card with a cluster of machines you already have.

**16 × 4GB GPUs = 64GB total VRAM at a fraction of the cost.**

## Architecture

```
[Client] gRPC streaming
    ↓
[Load Balancer]
    ↓                    ↓
[Coordinator 1]    [Coordinator 2]
   LEADER             STANDBY
    │
    ├── Tokenizer (DJL SentencePiece)
    ├── Scheduler (continuous batching)
    ├── Sampler (pure Java)
    ├── PrefixCache (Trie)
    └── InferencePipeline
              │
        gRPC (data plane) + Hazelcast (control plane)
              │
    [Node 1] [Node 2] ... [Node 16]
    Layer    Layer         Layer
    0-1      2-3           N
```

## Modules

| Module | Responsibility |
|--------|---------------|
| `api` | gRPC protos + shared DTOs |
| `registry` | Model Registry + Shard Loader (Hazelcast, GGUF, DJL) |
| `coordinator` | Coordinator + Scheduler (continuous batching, leader election) |
| `node` | Inference Node (JCuda, JCublas, gRPC) |
| `kvcache` | KV Cache Manager (GPU/CPU/Disk tiers, prefix Trie) |
| `tokenizer` | Tokenizer (DJL SentencePiece, chat templates) |
| `sampler` | Sampler (pure Java, temperature/top-k/top-p) |
| `health` | Health Monitor (Hazelcast events, Resilience4j, Micrometer) |

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
# Start coordinator (on stable machine)
java --enable-preview -jar coordinator/target/coordinator.jar \
     --spring.config.location=cluster-config.yaml

# Start inference node (on each GPU machine)
java --enable-preview -jar node/target/node.jar \
     --spring.config.location=cluster-config.yaml \
     --node.device-id=0

# Load a model
curl -X POST http://coordinator:8080/models \
     -H 'Content-Type: application/json' \
     -d '{"modelId": "llama3-8b", "path": "/models/llama3-8b.gguf"}'

# Chat (gRPC streaming)
grpcurl -d '{"messages": [{"role": "user", "content": "Hello!"}]}' \
        coordinator:9090 io.hyperstack4j.api.InferenceService/InferStream
```

## Technology Stack

| Concern | Technology |
|---------|-----------|
| Language | Java 21 |
| Build | Maven (multi-module) |
| GPU compute | JCuda / JCublas 12.x |
| Distributed state | Hazelcast 5.x |
| Leader election | Hazelcast CP FencedLock |
| Data plane | gRPC + Protocol Buffers |
| Concurrency | Java 21 Virtual Threads |
| KV Cache L2 | OHC (off-heap) |
| KV Cache L3 | Ehcache 3 |
| Circuit breaker | Resilience4j |
| Metrics | Micrometer + Prometheus |
| Tokenizer | DJL SentencePiece |
| Weight format | GGUF |

## License

Apache 2.0
