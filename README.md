# 🦙 hyper-stack-4j

> Distributed Java-native LLM inference engine — commodity GPU cluster, no Python, no GIL, no Spring.

## Vision

Run large language models across a network of **affordable commodity GPUs** — replacing the need for a single expensive high-VRAM card with a cluster of machines you already have.

**16 × 4GB GPUs = 64GB total VRAM at a fraction of the cost.**

## Status

The full inference stack is working end-to-end with real models:

```
MODEL_PATH=/models/TinyLlama-1.1B-Chat-v1.0.Q4_K_M.gguf ./run-me.sh cluster

  hyper-stack-4j  ·  3-node cluster  ·  interactive console

✔ Cluster ready  (FLOAT32 activations)
✔ Tokenizer loaded from GGUF  (vocabSize=32000)

you> what is the capital of France?
bot> The capital of France is Paris...
     [42 tokens · 8310 ms · FLOAT32]
```

Three real JVM processes, real gRPC, real transformer math — no Ollama, no Python bridge, no external runtime.

**69 production classes · 53 test files · 340 unit tests · 22 integration tests**  
All tests green. Real-model end-to-end verified with TinyLlama-1.1B-Chat-v1.0.Q4_K_M.gguf.

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
              │  gRPC       (data plane  — activations, FLOAT32/16/INT8)
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

`CpuFPH` = `CpuForwardPassHandler` — real transformer math, pure Java.  
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

Math primitives: RMS normalisation, matrix–vector multiply, rotary position embeddings (RoPE), grouped-query attention (GQA) with in-process KV cache, SwiGLU FFN, softmax. All pure Java. `GpuForwardPassHandler` will override `matVec` with a JCublas call; everything above stays identical.

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

`decodeToken()` replaces the SentencePiece space-prefix character `▁` (U+2581) with a real space — essential for correct streaming output. The replacement is applied in the streaming path (`decodeToken`) and the batch path (`decode`) independently so neither can leak the raw SentencePiece marker to the client.

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

`forModelType(String)` is case-insensitive and falls back to `chatml` for unknown keys. Using the wrong template (e.g. ChatML for TinyLlama) sends tokens the model has never seen during fine-tuning and produces garbage output.

## Quick Start

### Try it now (CPU, no GPU required)

```bash
# Download TinyLlama — 637 MB, runs on any modern CPU
wget https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf

# Build
mvn clean install -T 1C -DskipTests

# Run
MODEL_PATH=/path/to/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf ./run-me.sh cluster
```

### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | _(none)_ | Path to a GGUF file. When set, loads real weights and tokenizer. When unset, uses stubs (useful for testing cluster plumbing). |
| `DTYPE` | `FLOAT32` | Activation compression between nodes: `FLOAT32`, `FLOAT16`, or `INT8` |
| `MAX_TOKENS` | `200` | Max tokens per response |
| `TEMPERATURE` | `0.7` | Sampling temperature |
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
| `coordinator` | `GenerationLoop` (prefill + decode), `RequestScheduler`, Javalin REST, SSE streaming, batch dispatch, `FaultTolerantPipeline` |
| `node` | `CpuForwardPassHandler`, `GgufReader`, `LlamaConfig`, `StubForwardPassHandler`, `ActivationCodec` |
| `kvcache` | KV Cache Manager — GPU tier (JCuda) + JVM heap tier (Caffeine). Prefix Trie |
| `tokenizer` | `GgufTokenizer` (SentencePiece BPE from GGUF), `DJLTokenizer`, `StubTokenizer`, chat templates |
| `sampler` | Pure Java sampler — temperature, top-k, top-p, repetition penalty |
| `health` | Health Monitor (Hazelcast events, JCuda GPU probes, Resilience4j circuit breakers) |
| `integration` | Multi-JVM cluster tests + `ConsoleMain` REPL. `GgufTokenizer`, `EmbeddedNodeServer`, `ClusterHarness` |

## Activation Compression

Activation tensors between nodes are the primary network bottleneck. At 70B scale each hop costs ~64 MB over 10GbE.

| Dtype | Size/element | Relative error | Transfer (70B, 10GbE) |
|-------|-------------|---------------|-----------------------|
| `FLOAT32` | 4 B | lossless | ~51 ms/hop |
| `FLOAT16` | 2 B | ~0.1% | ~26 ms/hop |
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

- **JDK 21+**
- **Maven 3.9+**
- **CUDA 12.x** (GPU nodes only — not needed for CPU mode)
- **10GbE networking** recommended for multi-machine clusters

## Technology Stack

| Concern | Technology |
|---------|-----------|
| Language | Java 21 |
| Build | Maven (multi-module) |
| GPU compute | JCuda / JCublas 12.x |
| Distributed state | Hazelcast 5.x |
| Leader election | Hazelcast CP FencedLock |
| Data plane | gRPC + Protocol Buffers |
| REST | Javalin 6.x |
| Concurrency | Java 21 Virtual Threads + `CompletableFuture` |
| KV Cache L1 | JCuda CudaBuffer (GPU VRAM) |
| KV Cache L2 | Caffeine (JVM heap, W-TinyLFU) |
| Circuit breaker | Resilience4j |
| Metrics | Micrometer + Prometheus |
| Tokenizer | `GgufTokenizer` (built-in SentencePiece BPE), `DJLTokenizer` (JNI) |
| Weight format | GGUF v2/v3 |
| Sampler | Pure Java |

## Bug Fixes (session 5)

Three correctness bugs were found and fixed during real-model verification. All three caused garbage output when running TinyLlama.

### Bug 1 — Wrong chat template for TinyLlama
**File:** `tokenizer/src/main/java/io/hyperstack4j/tokenizer/ChatTemplate.java`

`ChatTemplate.BUILT_IN` was missing `"tinyllama"`, so `ChatTemplateFormatter.forModelType("tinyllama")` silently fell back to ChatML. TinyLlama is fine-tuned on the Zephyr template (`<|user|>`, `</s>`, `<|assistant|>`), not ChatML — the wrong template sends token sequences the model has never seen during training, producing complete garbage.

Fix: added `tinyllama()` static method and registered both `"tinyllama"` and `"zephyr"` as aliases pointing to the same implementation.

### Bug 2 — Raw `▁` leaked in streaming output
**File:** `tokenizer/src/main/java/io/hyperstack4j/tokenizer/GgufTokenizer.java`

`decodeToken()` returned raw SentencePiece pieces with the space-prefix character `▁` (U+2581) intact. The batch `decode()` path replaced it correctly, but the streaming path — which builds the full text by accumulating `decodeToken()` pieces — did not. Every word in streaming output appeared with a visible `▁` prefix instead of a space.

Fix: `decodeToken()` now replaces `▁` with a space before returning.

### Bug 3 — Q6_K dequantization wrong for positions ≥ 32
**File:** `node/src/main/java/io/hyperstack4j/node/GgufReader.java`

`loadQ6_K()` used a flat loop `for (int i = 0; i < 256; i++)` and indexed `qh` as `hi = i / 4`. This diverges from the llama.cpp reference (`dequantize_row_q6_K`) from position 32 onwards: the correct structure splits each 256-element block into two halves of 128 elements, and within each half `l` runs 0..31 producing four outputs that share a single `qh` byte. The wrong indexing caused all KV-projection and FFN weights in Q6_K-quantised models to produce incorrect values at position ≥ 32 in every block — a complete correctness failure for the most common superblock quantisation type.

Fix: restructured the inner loop to match the two-halves × 32 layout exactly:
```
for (int half = 0; half < 2; half++)
    for (int l = 0; l < 32; l++)
        // out[l], out[l+32], out[l+64], out[l+96]
        // all share qh[qhBase + l]
```

## Notable Design Decisions

- **No Python.** No Ollama. No llama.cpp subprocess. The JVM reads the GGUF binary directly and runs the transformer end to end.
- **No Spring Boot** — Javalin for REST, JDK HttpServer for metrics scrape.
- **No disk KV cache** — all cache in RAM. Caffeine evicts cleanly when `-Xmx` is reached.
- **Pipeline parallelism** over tensor parallelism — LAN-friendly, no InfiniBand required.
- **Separate data plane (gRPC) from control plane (Hazelcast)** — mirrors Kafka/Kubernetes design.
- **GGUF tokenizer from metadata** — vocab, scores, and special token IDs are all in the `.gguf` file. No separate `tokenizer.model` needed.
- **Stub mode** — without `MODEL_PATH`, the cluster boots in seconds and all 15 integration tests run without any model file.

## License

Apache 2.0