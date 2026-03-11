# 🦙 hyper-stack-4j

> Distributed Java-native LLM inference engine — commodity GPU cluster, no Python, no GIL, no Spring.

## Vision

Run large language models across a network of **affordable commodity GPUs** — replacing the need for a single expensive high-VRAM card with a cluster of machines you already have.

**16 × 4GB GPUs = 64GB total VRAM at a fraction of the cost.**

## Status

The full inference stack is working end-to-end with real models:

```
MODEL_PATH=/models/TinyLlama-1.1B-Chat-v1.0.Q4_K_M.gguf ./run-me.sh cluster

  hyper-stack-4j  ·  3-node cluster  ·  TinyLlama-1.1B-Chat-v1.0.Q4_K_M.gguf  ·  interactive console
  dtype=FLOAT16  max_tokens=200  temperature=0.7  nodes=3 (localhost:19092-19094)

✔ Cluster ready  (FLOAT16 activations)

you> awesome. how are you?
bot> i'm doing well. How about you?
     [10 tokens · 3802 ms · FLOAT16]
```

Three real JVM processes, real gRPC, real transformer math — no Ollama, no Python bridge, no external runtime.

**72 production classes · 56 test files · 353 unit tests · 24 integration tests**  
All tests green. Real-model end-to-end verified with TinyLlama-1.1B-Chat-v1.0.Q4_K_M.gguf.

### Performance (session 6, TinyLlama-1.1B, 3-node CPU cluster)

| Version | dtype | ms / 10 tokens | Speedup |
|---------|-------|---------------|---------|
| Session 5 | FLOAT32 | ~34,891 ms | baseline |
| Session 6 | FLOAT16 | ~3,802 ms | **9×** |

Gains come from two independent changes: parallel `matVec` across CPU cores (4–8× on multi-core), and FLOAT16 as the default activation dtype (2× smaller gRPC payloads).

## Architecture

```
[Client] REST (Javalin) or gRPC streaming
    ↓
[Load Balancer]  HAProxy / Nginx
    ↓                         ↓
[Coordinator 1]         [Coordinator 2]
   LEADER                  STANDBY
    │
    ├── GgufTokenizer  (vocab loaded directly from GGUF metadata)
    ├── Scheduler      (CompletableFuture, virtual threads)
    ├── Sampler        (pure Java — temperature, top-k, top-p, rep. penalty)
    ├── PrefixCache    (Trie)
    └── InferencePipeline
              │
              │  gRPC       (data plane  — activations, FLOAT16/INT8/FLOAT32)
              │  Hazelcast  (control plane — state, events)
              │
    =============================================
    ||       10/25GbE RDMA Network            ||
    =============================================
         |          |          |          |
    [Node 1]   [Node 2]   [Node 3]  ... [Node 16]
    Layer 0-7  Layer 8-14 Layer 15-21   Layer N
    + Embed    CpuFPH     CpuFPH        + Output proj
    CpuFPH                              GpuFPH (planned)
```

`CpuFPH` = `CpuForwardPassHandler` — real transformer math, pure Java, parallel matVec.  
`GpuFPH` = `GpuForwardPassHandler` — same interface, JCuda/JCublas (in progress).

## Real Inference Pipeline

### GGUF model loading

`GgufReader` parses GGUF v2/v3 binary files directly — no external tools, no Python, no llama.cpp subprocess. Supported quantisation types:

| Type | Bits/weight | Notes |
|------|------------|-------|
| `F32` | 32 | Lossless |
| `F16` | 16 | IEEE 754 half-precision |
| `BF16` | 16 | bfloat16 |
| `Q8_0` | 8 | Symmetric, block-32 |
| `Q4_0` | 4 | Symmetric, block-32 |
| `Q4_K` | 4 | Per-superblock scale+min, block-256 — used by `Q4_K_M` files |
| `Q6_K` | 6 | Per-superblock scale, block-256 |

`LlamaConfig` extracts model hyperparameters (`hiddenDim`, `numLayers`, `numHeads`, `numKvHeads`, `ropeTheta`, etc.) from GGUF metadata. Works for LLaMA 2/3, TinyLlama, Mistral, Gemma.

### CPU transformer (CpuForwardPassHandler)

Full LLaMA-family transformer forward pass in pure Java. Each node runs only its assigned layer shard:

```
ShardContext:  startLayer=0  endLayer=8  hasEmbeddings=true
  →  embedding lookup (token_embd.weight)
  →  layers 0–7: RMS norm → Q/K/V projection → RoPE → GQA attention → SwiGLU FFN
  →  return activations to next node via gRPC

ShardContext:  startLayer=15  endLayer=22  hasOutputProjection=true
  →  layers 15–21: same
  →  output RMS norm → output projection → return float[vocabSize] logits
```

Math primitives: RMS normalisation, matrix–vector multiply (parallel), rotary position embeddings (RoPE), grouped-query attention (GQA) with in-process KV cache, SwiGLU FFN, softmax. All pure Java.

**Parallel matVec (session 6):** `matVec` uses `IntStream.range(0, rows).parallel()` for matrices with `rows ≥ 256`, distributing row dot-products across all available CPU cores via `ForkJoinPool.commonPool()`. This covers every major weight matrix in the transformer (Q/K/V projection, FFN gate/up/down, output projection) and is the primary source of the 9× speedup. Matrices below the threshold use a plain loop to avoid parallel overhead on small shapes.

**Prefill and decode:**

```
Prefill (prompt, N tokens):
  GenerationLoop: for p in 0..N-1: pipeline.forward(id, [tok_p], p)
  Every node's KV cache fills for positions 0..N-1

Decode (each new token):
  pipeline.forward(id, [lastToken], N + step)
  Last node returns logits → sampler → next token → stream to client
```

### Tokenizer (GgufTokenizer)

SentencePiece BPE tokenizer loaded directly from GGUF metadata — no separate `tokenizer.model` file needed. Reads `tokenizer.ggml.tokens`, `.scores`, and `.token_type` arrays from the same `.gguf` file as the weights. Handles byte fallback tokens (`<0xHH>`) for out-of-vocabulary characters.

`decodeToken()` replaces the SentencePiece space-prefix character `▁` (U+2581) with a real space — essential for correct streaming output.

### Chat Templates

`ChatTemplate` maps a model-family key to a prompt-formatting function. The registry (`BUILT_IN`) currently contains:

| Key | Format | Notes |
|-----|--------|-------|
| `llama3` | `<\|begin_of_text\|>...<\|eot_id\|>` | LLaMA 3 |
| `mistral` | `[INST] ... [/INST]` | Mistral v0.x |
| `gemma` | `<start_of_turn>user\n...<end_of_turn>` | Gemma |
| `chatml` | `<\|im_start\|>role\n...<\|im_end\|>` | Default fallback |
| `tinyllama` | `<\|user\|>\n...\</s>\n<\|assistant\|>\n` | TinyLlama / Zephyr |
| `zephyr` | _(same as tinyllama)_ | Alias — same instance |

`forModelType(String)` is case-insensitive and falls back to `chatml` for unknown keys.

## Quick Start

### Try it now (CPU, no GPU required)

```bash
# Download TinyLlama — 637 MB, runs on any modern CPU
wget https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf

# Build
mvn clean install -T 1C -DskipTests

# Run (FLOAT16 is the default — 2× smaller activations, negligible accuracy loss)
MODEL_PATH=/path/to/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf ./run-me.sh cluster
```

### run-me.sh cluster flags

```
./run-me.sh cluster [flags]

  --dtype FLOAT32|FLOAT16|INT8   activation wire format (default FLOAT16)
  --float16 / --fp16             shorthand for --dtype FLOAT16 (default)
  --float32                      shorthand for --dtype FLOAT32 (debug/reference)
  --int8                         shorthand for --dtype INT8
  --max-tokens N                 max generated tokens    (default 200)
  --temperature F                sampling temperature     (default 0.7)
  --heap SIZE                    JVM heap e.g. 4g 8g     (default 4g)
  --skip-build / -B              skip mvn test-compile (use last build)
  --verbose / -v                 show full gRPC + Maven logs
  --help                         all flags
```

### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | _(none)_ | Path to a GGUF file. When set, loads real weights and tokenizer. When unset, uses stubs (useful for testing cluster plumbing). |
| `DTYPE` | `FLOAT16` | Activation compression between nodes: `FLOAT16`, `FLOAT32`, or `INT8` |
| `MAX_TOKENS` | `200` | Max tokens per response |
| `TEMPERATURE` | `0.7` | Sampling temperature |
| `HEAP` | `4g` | JVM heap size for the coordinator/console process |
| `HYPER_VERBOSE` | _(unset)_ | Set to `true` to show gRPC/node logs |

### Larger models

Any GGUF file with a LLaMA-compatible architecture works:

| Model | Size | RAM needed |
|-------|------|------------|
| `TinyLlama-1.1B-Chat.Q4_K_M.gguf` | 637 MB | ~2 GB |
| `Mistral-7B-Instruct-v0.2.Q4_K_M.gguf` | 4.1 GB | ~6 GB |
| `Llama-3.2-8B-Instruct.Q4_K_M.gguf` | 4.9 GB | ~8 GB |
| `Llama-3.1-70B-Instruct.Q4_K_M.gguf` | 40 GB | 16 × 4GB nodes |

## Modules

| Module | Responsibility |
|--------|---------------|
| `api` | OpenAPI 3.0 spec + JAX-RS interfaces + gRPC proto (`inference.proto`), including `ActivationDtype` |
| `registry` | Model Registry + Shard Planner (Hazelcast IMap, GGUF metadata, IMQ seed scoring) |
| `coordinator` | `GenerationLoop` (prefill + decode + EOS filtering), `RequestScheduler`, Javalin REST, SSE streaming, batch dispatch, `FaultTolerantPipeline` |
| `node` | `CpuForwardPassHandler` (parallel matVec), `GgufReader`, `LlamaConfig`, `StubForwardPassHandler`, `ActivationCodec` |
| `kvcache` | KV Cache Manager — GPU tier (JCuda) + JVM heap tier (Caffeine). Prefix Trie |
| `tokenizer` | `GgufTokenizer` (SentencePiece BPE from GGUF), `DJLTokenizer`, `StubTokenizer`, chat templates |
| `sampler` | Pure Java sampler — temperature, top-k, top-p, repetition penalty |
| `health` | Health Monitor (Hazelcast events, JCuda GPU probes, Resilience4j circuit breakers) |
| `integration` | Multi-JVM cluster tests + `ConsoleMain` REPL. Parallel shard loading via `CompletableFuture.allOf`. |

## Activation Compression

Activation tensors between nodes are the primary network bottleneck. At 70B scale each hop costs ~64 MB over 10GbE. **FLOAT16 is now the default.**

| Dtype | Size/element | Relative error | Transfer (70B, 10GbE) |
|-------|-------------|---------------|-----------------------|
| `FLOAT32` | 4 B | lossless | ~51 ms/hop |
| `FLOAT16` | 2 B | ~0.1% | ~26 ms/hop ← **default** |
| `INT8`    | 1 B + 4 B scale | ~1% | ~13 ms/hop |

```java
ProcessPipelineClient pipeline = new ProcessPipelineClient(nodes, vocabSize, ActivationDtype.FLOAT16);
```

`ActivationCodec` is pure Java. FLOAT16 uses manual IEEE 754 bit manipulation; INT8 uses symmetric quantisation with a float32 scale prefix.

## Scheduler — Reactive Design

```java
// Streaming — returns immediately, tokens delivered via callback
CompletableFuture<GenerationResult> future = scheduler.submit(request, consumer);

// Blocking — caller waits only on its own future
GenerationResult result = scheduler.submitAndWait(request);
```

N concurrent callers block on **independent futures** — no shared lock between requests. Queue full → `QueueFullException` → HTTP 503 + Retry-After.

## Shard Planner — Fair Layer Distribution

```
maxLayers = min(layersFit, remainingLayers − (remainingNodes − 1))
```

Every eligible node is guaranteed at least one layer, regardless of VRAM headroom.

Shard loading is **parallel** — all nodes receive their `LoadShard` RPC concurrently via `CompletableFuture.allOf()`. Startup time is bounded by the slowest single node rather than the sum across all nodes.

## API

REST (Javalin 6, OpenAPI 3.0):

```
POST   /v1/inference          — blocking inference
POST   /v1/inference/stream   — SSE token streaming
POST   /v1/models             — load model
GET    /v1/models/{modelId}   — model status + shard map
DELETE /v1/models/{modelId}   — unload model
GET    /v1/cluster/health     — cluster health
GET    /v1/cluster/shardmap   — current layer assignments
```

## Build & Test

```bash
# Build
mvn clean install -T 1C

# Unit tests only (fast — no model file needed)
mvn test -pl tokenizer,node,coordinator,sampler,kvcache,health,registry

# Unit tests for specific recently-changed modules
mvn test -pl tokenizer,node

# Integration tests — forks 3 real JVM node processes (stub mode, no model)
mvn verify -pl integration

# Integration tests — with real TinyLlama model (runs TinyLlamaLiveIT)
mvn verify -pl integration \
  -Dit.model.path=/home/robocop/dev/space/TinyLlama-1.1B-Chat-v1.0.Q4_K_M.gguf

# Run only the live model IT
mvn verify -pl integration \
  -Dit.model.path=/home/robocop/dev/space/TinyLlama-1.1B-Chat-v1.0.Q4_K_M.gguf \
  -Dit.test=TinyLlamaLiveIT

# Skip ITs
mvn verify -DskipITs
```

## Requirements

- **JDK 25+**
- **Maven 3.9+**
- **CUDA 12.x** (GPU nodes only — not needed for CPU mode)
- **10GbE networking** recommended for multi-machine clusters

## Technology Stack

| Concern | Technology |
|---------|-----------|
| Language | Java 25 |
| Build | Maven (multi-module) |
| GPU compute | JCuda / JCublas 12.x |
| Distributed state | Hazelcast 5.x |
| Leader election | Hazelcast CP FencedLock |
| Data plane | gRPC + Protocol Buffers |
| REST | Javalin 6.x |
| Concurrency | Java 25 Virtual Threads + `CompletableFuture` |
| KV Cache L1 | JCuda CudaBuffer (GPU VRAM) |
| KV Cache L2 | Caffeine (JVM heap, W-TinyLFU) |
| Circuit breaker | Resilience4j |
| Metrics | Micrometer + Prometheus |
| Tokenizer | `GgufTokenizer` (built-in SentencePiece BPE), `DJLTokenizer` (JNI) |
| Weight format | GGUF v2/v3 |
| Sampler | Pure Java |
| CPU matmul | `IntStream.parallel()` over `ForkJoinPool.commonPool()` |

## Bug Fixes (session 5)

Three correctness bugs were found and fixed during real-model verification. All three caused garbage output when running TinyLlama.

### Bug 1 — Wrong chat template for TinyLlama
**File:** `tokenizer/src/main/java/io/hyperstack4j/tokenizer/ChatTemplate.java`

`ChatTemplate.BUILT_IN` was missing `"tinyllama"`, so `ChatTemplateFormatter.forModelType("tinyllama")` silently fell back to ChatML. TinyLlama is fine-tuned on the Zephyr template (`<|user|>`, `</s>`, `<|assistant|>`), not ChatML — the wrong template sends token sequences the model has never seen during training, producing complete garbage.

Fix: added `tinyllama()` static method and registered both `"tinyllama"` and `"zephyr"` as aliases pointing to the same implementation.

### Bug 2 — Raw `▁` leaked in streaming output
**File:** `tokenizer/src/main/java/io/hyperstack4j/tokenizer/GgufTokenizer.java`

`decodeToken()` returned raw SentencePiece pieces with the space-prefix character `▁` (U+2581) intact. The batch `decode()` path replaced it correctly, but the streaming path did not.

Fix: `decodeToken()` now replaces `▁` with a space before returning.

### Bug 3 — Q6_K dequantization wrong for positions ≥ 32
**File:** `node/src/main/java/io/hyperstack4j/node/GgufReader.java`

`loadQ6_K()` used a flat loop with wrong `qh` indexing from position 32 onwards, causing all KV-projection and FFN weights in Q6_K-quantised models to produce incorrect values for ≥ 75% of each block's elements.

Fix: restructured to the two-halves × 32 layout matching the llama.cpp C reference.

## Performance & Correctness Improvements (session 6)

### Parallel matVec — 4–8× faster token generation
**File:** `node/src/main/java/io/hyperstack4j/node/CpuForwardPassHandler.java`

`matVec` now uses `IntStream.range(0, rows).parallel()` for matrices with `rows ≥ 256`. Each row's dot-product is independent, so the operation parallelises perfectly across all CPU cores. The threshold avoids parallel overhead on small matrices (RoPE, attention scores). Combined with FLOAT16 default, delivered a **9× end-to-end speedup** on TinyLlama-1.1B (34,891 ms → 3,802 ms per 10 tokens on the same CPU cluster).

### Parallel shard loading — faster cluster startup
**File:** `integration/src/test/java/io/hyperstack4j/integration/ProcessPipelineClient.java`

`loadShards()` now fires all `LoadShard` RPCs concurrently via `CompletableFuture.allOf()`. Previously nodes loaded sequentially (node-1 finished, then node-2 started, etc.). On a 3-node cluster loading TinyLlama-1.1B this saves ~4 seconds at startup; on larger models or more nodes the saving is proportionally greater.

### EOS piece suppression — clean output, no </s> leaks
**File:** `coordinator/src/main/java/io/hyperstack4j/coordinator/GenerationLoop.java`

Two-layer defence:

1. **Token ID check** (was already present): `nextToken == eosTokenId()` breaks before `decodeToken()` is called — the EOS ID never reaches the consumer.

2. **Piece string filter** (new): `isEosMarker(piece)` catches the GgufTokenizer quirk where a non-EOS token ID decodes to an EOS marker string (`</s>`, `<|endoftext|>`, `<|eot_id|>`, `<end_of_turn>`). Without this, `</s>` appeared visibly in streaming output when the model's regular vocabulary contained the string at a different ID than the special EOS ID.

Applied in both `generate()` and `generateBatch()`.

### FLOAT16 default
**File:** `run-me.sh`

`DTYPE` default changed from `FLOAT32` to `FLOAT16`. On a localhost 3-node cluster the bandwidth saving is minor, but the practice matches production intent and halves gRPC payload size with no measurable accuracy loss on 1–7B models.

### JVM tuning
**File:** `run-me.sh`

| Flag | Old | New | Reason |
|------|-----|-----|--------|
| `-Xms` | `256m` | `512m` | Avoids early GC before GGUF weights are loaded |
| `-Xmx` | `2g` | `4g` (env: `HEAP`) | `2g` was tight for real models; configurable via `--heap` |
| GC | `ZGC` | `G1GC` | Lower startup latency for dev runs; `ZGC` adds ~200ms/JVM |
| `-XX:+AlwaysPreTouch` | absent | added | Pre-commits heap pages; prevents first-request GC stutter |
| `--enable-native-access` | absent | `ALL-UNNAMED` | Silences Netty `loadLibrary` restricted-method warning |
| `--add-opens` | absent | `java.base/java.lang`, `java.base/java.nio` | Silences Guava/Netty `sun.misc.Unsafe` warnings |

New `--skip-build / -B` flag skips `mvn test-compile` when classes are already up to date, saving ~10 seconds per iteration cycle.

### Session 6 test additions

`GenerationLoopEosPieceTest` (coordinator module) — 4 new tests:
- `eos_token_id_stops_immediately_no_piece_streamed`
- `eos_string_piece_from_non_eos_token_suppressed` ← primary regression anchor for `</s>` leak
- `endoftext_string_piece_from_non_eos_token_suppressed`
- `non_eos_angle_bracket_tokens_are_not_filtered` (anti-regression: no over-filtering)

`MatVecParallelTest` (node module) — 9 new tests:  
Correctness anchor for `matVec` across all weight matrix shapes in TinyLlama-1.1B (2×3, 32×32, 256×256, 2048×2048, 5632×2048, 2048×5632, 32000×2048). Scalar reference compared against parallel output with 1e-4 tolerance.

`LoadShardsParallelTest` (integration module) — 2 new tests using lightweight in-process gRPC servers:
- `all_nodes_receive_load_shard` — verifies each of 3 nodes gets exactly one `LoadShard` RPC
- `load_shards_is_parallel_not_serial` — timing test: 3 nodes × 300ms delay must complete in < 600ms total

## Notable Design Decisions

- **No Python.** No Ollama. No llama.cpp subprocess. The JVM reads the GGUF binary directly and runs the transformer end to end.
- **No Spring Boot** — Javalin for REST, JDK HttpServer for metrics scrape.
- **No disk KV cache** — all cache in RAM. Caffeine evicts cleanly when `-Xmx` is reached.
- **Pipeline parallelism** over tensor parallelism — LAN-friendly, no InfiniBand required.
- **Separate data plane (gRPC) from control plane (Hazelcast)** — mirrors Kafka/Kubernetes design.
- **GGUF tokenizer from metadata** — vocab, scores, and special token IDs are all in the `.gguf` file. No separate `tokenizer.model` needed.
- **Stub mode** — without `MODEL_PATH`, the cluster boots in seconds and all integration tests run without any model file.
- **Two `ActivationDtype` enums by design** — `io.hyperstack4j.api.grpc.ActivationDtype` is protobuf-generated and used only for wire serialization. `io.hyperstack4j.node.ActivationDtype` is the domain enum used throughout application code. `ProcessPipelineClient` bridges them with `toProto()` / `fromProto()`. Coupling domain code to protobuf-generated types is an antipattern.

## License

Apache 2.0