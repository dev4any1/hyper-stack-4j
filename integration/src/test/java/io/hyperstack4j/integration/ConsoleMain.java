package io.hyperstack4j.integration;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.file.Path;
import java.util.List;

import io.hyperstack4j.coordinator.GenerationLoop;
import io.hyperstack4j.coordinator.GenerationResult;
import io.hyperstack4j.coordinator.InferenceRequest;
import io.hyperstack4j.coordinator.RequestPriority;
import io.hyperstack4j.kvcache.CpuKVCache;
import io.hyperstack4j.kvcache.GpuKVCache;
import io.hyperstack4j.kvcache.KVCacheManager;
import io.hyperstack4j.node.ActivationDtype;
import io.hyperstack4j.node.GgufReader;
import io.hyperstack4j.sampler.Sampler;
import io.hyperstack4j.sampler.SamplingParams;
import io.hyperstack4j.tokenizer.ChatMessage;
import io.hyperstack4j.tokenizer.GgufTokenizer;
import io.hyperstack4j.tokenizer.SimpleTokenizer;
import io.hyperstack4j.tokenizer.Tokenizer;

/**
 * Interactive REPL that boots a full 3-node gRPC cluster and accepts prompts
 * from stdin, streaming tokens to stdout as they are generated.
 *
 * Behaviour:
 *   - Forks 3 node JVMs (same as ThreeNodeClusterIT / ClusterHarness)
 *   - Wires GenerationLoop → ProcessPipelineClient → 3 × EmbeddedNodeServer
 *   - Reads stdin line-by-line; each line is a prompt sent to the cluster
 *   - Tokens stream to stdout as they arrive (no buffering)
 *   - Type  exit  or  quit  to shut down cleanly
 *   - Ctrl-C triggers the shutdown hook and tears down all 3 node JVMs
 *
 * Environment variables:
 *   DTYPE         FLOAT32 | FLOAT16 | INT8   (default: FLOAT32)
 *   MAX_TOKENS    integer                    (default: 200)
 *   TEMPERATURE   float                      (default: 0.7)
 *
 *   OLLAMA_MODEL  e.g. tinyllama or llama3   — when set, skips the stub gRPC
 *                 cluster entirely and streams real tokens from a local Ollama
 *                 instance (http://localhost:11434).  No extra Maven dependency
 *                 needed — uses java.net.http.HttpClient (built-in since JDK 11).
 *   OLLAMA_URL    override Ollama base URL   (default: http://localhost:11434)
 *
 *   MODEL_PATH    path to a GGUF model file — when set, uses CpuForwardPassHandler
 *                 on each node to run real transformer inference instead of stubs.
 *                 e.g. MODEL_PATH=/models/TinyLlama-1.1B-Chat-v1.0.Q4_K_M.gguf
 *
 * Launched by run-me.sh cluster via:
 *   mvn exec:java -pl integration \
 *       -Dexec.mainClass=io.hyperstack4j.integration.ConsoleMain \
 *       -Dexec.classpathScope=test
 */
public final class ConsoleMain {

    // Silence gRPC / Netty / hyperstack noise — keep the console readable.
    // Runs before main() so nothing leaks during static init of other classes.
    // In verbose mode (HYPER_VERBOSE=true) SLF4J/logback is re-opened to DEBUG.
    static {
        boolean verbose = "true".equalsIgnoreCase(System.getProperty("HYPER_VERBOSE"));

        if (!verbose) {
            // JUL — covers grpc-java, hyperstack4j
            java.util.logging.LogManager.getLogManager().reset();
            java.util.logging.Logger root = java.util.logging.Logger.getLogger("");
            root.setLevel(java.util.logging.Level.OFF);
            // Also explicitly silence common noisy namespaces in case something
            // resets the root logger after this (e.g. a library calling getLogger())
            for (String ns : new String[]{
                    "io.grpc", "io.netty", "io.hyperstack4j",
                    "com.google", "org.slf4j", ""}) {
                java.util.logging.Logger.getLogger(ns)
                        .setLevel(java.util.logging.Level.OFF);
            }
            // logback-test.xml already defaults root level to ERROR — no action needed.

        } else {
            // Verbose mode: logback-test.xml defaults root to ERROR, so raise it back
            // to DEBUG so gRPC / Netty wire-level logs become visible again.
            try {
                ch.qos.logback.classic.Logger logbackRoot =
                        (ch.qos.logback.classic.Logger)
                        org.slf4j.LoggerFactory.getLogger(org.slf4j.Logger.ROOT_LOGGER_NAME);
                logbackRoot.setLevel(ch.qos.logback.classic.Level.DEBUG);
            } catch (Exception ignored) { /* logback not on classpath — safe to skip */ }
        }
    }

    // ANSI colours
    private static final String CYAN    = "\033[0;36m";
    private static final String GREEN   = "\033[0;32m";
    private static final String YELLOW  = "\033[1;33m";
    private static final String DIM     = "\033[2m";
    private static final String RESET   = "\033[0m";
    private static final String BOLD    = "\033[1m";

    public static void main(String[] args) throws Exception {

        // ── Config — reads -DDTYPE=... etc (exec:exec) or env vars
        ActivationDtype dtype       = parseDtype(syspropOrEnv("DTYPE",        "FLOAT32"));
        int             maxTokens   = parseIntVal(syspropOrEnv("MAX_TOKENS",   "200"));
        float           temperature = parseFloatVal(syspropOrEnv("TEMPERATURE", "0.7"));
        String          ollamaModel = syspropOrEnv("OLLAMA_MODEL", "");
        String          ollamaUrl   = syspropOrEnv("OLLAMA_URL",   "http://localhost:11434");
        String          modelPath   = syspropOrEnv("MODEL_PATH",   "");

        banner(dtype, maxTokens, temperature, ollamaModel.isBlank() && !modelPath.isBlank() ? "(gguf: " + java.nio.file.Path.of(modelPath).getFileName() + ")" : ollamaModel);

        if (!ollamaModel.isBlank()) {
            runOllamaRepl(ollamaModel, ollamaUrl, maxTokens, temperature);
        } else {
            runCpuClusterRepl(dtype, maxTokens, temperature, modelPath);
        }
    }

    // ── Ollama REPL ────────────────────────────────────────────────────────────────

    /**
     * Real-model mode: bypass the stub cluster entirely.
     * Streams tokens from a local Ollama instance using java.net.http.HttpClient
     * (JDK built-in — zero extra Maven dependencies).
     *
     * Get started:
     *   curl -fsSL https://ollama.com/install.sh | sh
     *   ollama pull tinyllama       # 637 MB — fast on CPU, good for dev
     *   ollama pull llama3.2        # 2 GB  — much better quality
     *   ollama pull mistral         # 4 GB  — excellent quality
     *
     * Then run:
     *   OLLAMA_MODEL=tinyllama ./run-me.sh cluster
     *   OLLAMA_MODEL=llama3.2  MAX_TOKENS=500 ./run-me.sh cluster
     */
    private static void runOllamaRepl(String model, String baseUrl,
                                      int maxTokens, float temperature) throws Exception {
        print(CYAN + "▶ Ollama mode — model: " + BOLD + model + RESET + CYAN
                + "  url: " + baseUrl + RESET);

        HttpClient http = HttpClient.newHttpClient();

        // Verify Ollama is reachable before entering the REPL
        try {
            HttpRequest ping = HttpRequest.newBuilder()
                    .uri(URI.create(baseUrl + "/api/tags"))
                    .GET().build();
            HttpResponse<String> resp = http.send(ping, HttpResponse.BodyHandlers.ofString());
            if (resp.statusCode() != 200) throw new RuntimeException("HTTP " + resp.statusCode());
            print(GREEN + "✔ Ollama reachable.  Streaming with " + model + RESET + "\n");
        } catch (Exception e) {
            print("\033[0;31m✖ Cannot reach Ollama at " + baseUrl + ": " + e.getMessage() + "\033[0m");
            print(DIM + "  Is Ollama running?  Try:  ollama serve" + RESET);
            print(DIM + "  Model installed?    Try:  ollama pull " + model + RESET);
            System.exit(1);
        }

        print(DIM + "Type your prompt and press Enter. Type 'exit' or Ctrl-C to quit." + RESET);
        print("");

        // Conversation history — Ollama /api/chat keeps context across turns
        java.util.List<String> history = new java.util.ArrayList<>();

        BufferedReader stdin = new BufferedReader(new InputStreamReader(System.in));
        String line;

        while (true) {
            System.out.print(BOLD + CYAN + "you> " + RESET);
            System.out.flush();

            line = stdin.readLine();
            if (line == null) break;
            line = line.strip();
            if (line.isEmpty()) continue;
            if (line.equalsIgnoreCase("exit") || line.equalsIgnoreCase("quit")) break;

            history.add("{\"role\":\"user\",\"content\":\"" + escapeJson(line) + "\"}");

            String body = "{\"model\":\"" + model + "\",\"stream\":true,"
                    + "\"options\":{\"num_predict\":" + maxTokens
                    + ",\"temperature\":" + temperature + "},"
                    + "\"messages\":[" + String.join(",", history) + "]}";

            HttpRequest req = HttpRequest.newBuilder()
                    .uri(URI.create(baseUrl + "/api/chat"))
                    .header("Content-Type", "application/json")
                    .POST(HttpRequest.BodyPublishers.ofString(body))
                    .build();

            System.out.print(BOLD + GREEN + "bot> " + RESET);
            System.out.flush();

            long         start   = System.currentTimeMillis();
            StringBuilder reply  = new StringBuilder();
            int          tokens  = 0;

            HttpResponse<java.io.InputStream> resp =
                    http.send(req, HttpResponse.BodyHandlers.ofInputStream());

            if (resp.statusCode() != 200) {
                print("\033[0;31m\n✖ Ollama returned HTTP " + resp.statusCode() + "\033[0m");
                continue;
            }

            try (BufferedReader reader = new BufferedReader(
                    new InputStreamReader(resp.body()))) {
                String jsonLine;
                while ((jsonLine = reader.readLine()) != null && !jsonLine.isBlank()) {
                    // Each streaming line: { "message": { "content": "token" }, "done": false }
                    String piece = extractJsonString(jsonLine, "content");
                    if (piece != null && !piece.isEmpty()) {
                        System.out.print(piece);
                        System.out.flush();
                        reply.append(piece);
                    }
                    if (jsonLine.contains("\"done\":true")) {
                        // Final line carries eval_count (actual token count)
                        String ec = extractJsonField(jsonLine, "eval_count");
                        if (ec != null) {
                            try { tokens = Integer.parseInt(ec); } catch (NumberFormatException ignored) {}
                        }
                        break;
                    }
                    tokens++;
                }
            }

            long elapsed = System.currentTimeMillis() - start;
            System.out.println();
            System.out.printf(DIM + "     [%d tokens · %d ms · %s]" + RESET + "%n",
                    tokens, elapsed, model);
            System.out.println();

            // Append assistant reply to history for multi-turn context
            history.add("{\"role\":\"assistant\",\"content\":\"" + escapeJson(reply.toString()) + "\"}");
        }

        print(YELLOW + "\nbye." + RESET);
        System.exit(0);
    }

    // ── Stub cluster REPL ──────────────────────────────────────────────────────────

    private static void runCpuClusterRepl(ActivationDtype dtype, int maxTokens,
                                           float temperature, String modelPath) throws Exception {
        print(CYAN + "▶ Starting 3-node cluster..." + RESET);

        ClusterHarness harness = ClusterHarness.threeNodes(modelPath.isBlank() ? null : modelPath);

        Runtime.getRuntime().addShutdownHook(Thread.ofVirtual().unstarted(() -> {
            print("\n" + YELLOW + "⏹ Shutting down cluster..." + RESET);
            try { harness.stop(); } catch (Exception e) { /* best effort */ }
            print(YELLOW + "✔ Cluster stopped." + RESET);
        }));

        harness.start();
        print(GREEN + "✔ Cluster ready  (" + dtype + " activations)" + RESET + "\n");

        ProcessPipelineClient pipeline = new ProcessPipelineClient(
                harness.nodeAddresses(),
                EmbeddedNodeServer.VOCAB_SIZE,
                dtype
        );
        Tokenizer tokenizer;
        if (modelPath != null && !modelPath.isBlank()) {
            try (GgufReader reader = GgufReader.open(Path.of(modelPath))) {
                tokenizer = GgufTokenizer.load(reader);
            } catch (IOException e) {
                System.err.println("Failed to load tokenizer from GGUF, falling back to SimpleTokenizer: " + e);
                tokenizer = new SimpleTokenizer();
            }
        } else {
            tokenizer = new SimpleTokenizer();
        }

        GenerationLoop loop = new GenerationLoop(
                tokenizer,
        		Sampler.create(),
                pipeline,
                new KVCacheManager(
                        new GpuKVCache(512L * 1024 * 1024),
                        new CpuKVCache(4096)
                )
        );

        SamplingParams params = SamplingParams.defaults()
                .withMaxTokens(maxTokens)
                .withTemperature(temperature);

        print(DIM + "Type your prompt and press Enter. Type 'exit' or Ctrl-C to quit." + RESET);
        print("");

        BufferedReader stdin = new BufferedReader(new InputStreamReader(System.in));
        String line;

        while (true) {
            System.out.print(BOLD + CYAN + "you> " + RESET);
            System.out.flush();

            line = stdin.readLine();
            if (line == null) break;
            line = line.strip();
            if (line.isEmpty()) continue;
            if (line.equalsIgnoreCase("exit") || line.equalsIgnoreCase("quit")) break;

            InferenceRequest request = InferenceRequest.of(
                    "tinyllama",
                    List.of(ChatMessage.user(line)),
                    params,
                    RequestPriority.NORMAL
            );

            System.out.print(BOLD + GREEN + "bot> " + RESET);
            System.out.flush();

            long start = System.currentTimeMillis();

            // Anonymous consumer so we can hook onPrefillStart / onPrefillComplete
            // and show a visible indicator during the (potentially slow) prefill phase.
            io.hyperstack4j.coordinator.TokenConsumer consumer =
                    new io.hyperstack4j.coordinator.TokenConsumer() {
                @Override
                public void onToken(String piece, int tokenId, int step) {
                    System.out.println("[" + step + ":" + tokenId + "]" + piece);
                    System.out.flush();
                }
                @Override
                public void onPrefillStart(int promptLen) {
                    System.out.print(DIM + "(prefilling " + promptLen + " tokens…) " + RESET);
                    System.out.flush();
                }
                @Override
                public void onPrefillComplete() {
                    // Clear the "(prefilling…)" hint by overwriting with spaces,
                    // then reprint the bot> marker so tokens follow it cleanly.
                    System.out.print("\r" + BOLD + GREEN + "bot> " + RESET);
                    System.out.flush();
                }
            };

            GenerationResult result = loop.generate(request, consumer);

            long elapsed = System.currentTimeMillis() - start;

            System.out.println();
            System.out.printf(DIM + "     [%d tokens · %d ms · %s]" + RESET + "%n",
                    result.generatedTokens(), elapsed, dtype);
            System.out.println();
        }

        print(YELLOW + "\nbye." + RESET);
        System.exit(0);
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    private static void banner(ActivationDtype dtype, int maxTokens, float temperature, String ollamaModel) {
        System.out.println();
        System.out.println(BOLD + CYAN
                + "  ██╗  ██╗██╗   ██╗██████╗ ███████╗██████╗ ");
        System.out.println(
                "  ██║  ██║╚██╗ ██╔╝██╔══██╗██╔════╝██╔══██╗");
        System.out.println(
                "  ███████║ ╚████╔╝ ██████╔╝█████╗  ██████╔╝");
        System.out.println(
                "  ██╔══██║  ╚██╔╝  ██╔═══╝ ██╔══╝  ██╔══██╗");
        System.out.println(
                "  ██║  ██║   ██║   ██║     ███████╗██║  ██║");
        System.out.println(
                "  ╚═╝  ╚═╝   ╚═╝   ╚═╝     ╚══════╝╚═╝  ╚═╝"
                + RESET);
        System.out.println();
        String mode = ollamaModel.isBlank() ? "3-node stub cluster" : "ollama · " + ollamaModel;
        System.out.println(CYAN + "  hyper-stack-4j  ·  " + mode + "  ·  interactive console" + RESET);
        System.out.println(DIM
                + (ollamaModel.isBlank()
                    ? "  dtype=" + dtype + "  max_tokens=" + maxTokens
                        + "  temperature=" + temperature + "  nodes=3 (localhost:19092-19094)"
                    : "  model=" + ollamaModel + "  max_tokens=" + maxTokens
                        + "  temperature=" + temperature)
                + RESET);
        System.out.println();
    }

    private static void print(String msg) {
        System.out.println(msg);
        System.out.flush();
    }

    private static String syspropOrEnv(String key, String defaultVal) {
        String v = System.getProperty(key);
        if (v != null && !v.isBlank()) return v.strip();
        v = System.getenv(key);
        if (v != null && !v.isBlank()) return v.strip();
        return defaultVal;
    }

    private static ActivationDtype parseDtype(String val) {
        if (val == null || val.isBlank()) return ActivationDtype.FLOAT32;
        return switch (val.toUpperCase().strip()) {
            case "FLOAT16", "F16", "FP16" -> ActivationDtype.FLOAT16;
            case "INT8",    "I8"           -> ActivationDtype.INT8;
            default                        -> ActivationDtype.FLOAT32;
        };
    }

    private static int parseIntVal(String val) {
        try { return Integer.parseInt(val.strip()); } catch (NumberFormatException e) { return 200; }
    }

    private static float parseFloatVal(String val) {
        try { return Float.parseFloat(val.strip()); } catch (NumberFormatException e) { return 0.7f; }
    }

    // ── Tiny JSON helpers (no external library needed) ───────────────────────────

    /**
     * Extract the string value of a JSON key from a flat JSON line.
     * Handles: "key":"value" and "key": "value" and nested one level deep.
     * Good enough for Ollama’s streaming protocol; not a general parser.
     */
    private static String extractJsonString(String json, String key) {
        String needle = "\"" + key + "\"";
        int ki = json.indexOf(needle);
        if (ki < 0) return null;
        int colon = json.indexOf(':', ki + needle.length());
        if (colon < 0) return null;
        int q1 = json.indexOf('"', colon + 1);
        if (q1 < 0) return null;
        StringBuilder sb = new StringBuilder();
        for (int i = q1 + 1; i < json.length(); i++) {
            char c = json.charAt(i);
            if (c == '\\') {
                if (i + 1 < json.length()) {
                    char next = json.charAt(++i);
                    switch (next) {
                        case 'n' -> sb.append('\n');
                        case 't' -> sb.append('\t');
                        case 'r' -> sb.append('\r');
                        default  -> sb.append(next);
                    }
                }
            } else if (c == '"') {
                break;
            } else {
                sb.append(c);
            }
        }
        return sb.toString();
    }

    /** Extract a raw (non-string) JSON field value, e.g. a number. */
    private static String extractJsonField(String json, String key) {
        String needle = "\"" + key + "\"";
        int ki = json.indexOf(needle);
        if (ki < 0) return null;
        int colon = json.indexOf(':', ki + needle.length());
        if (colon < 0) return null;
        int start = colon + 1;
        while (start < json.length() && json.charAt(start) == ' ') start++;
        int end = start;
        while (end < json.length() && ",}]".indexOf(json.charAt(end)) < 0) end++;
        return json.substring(start, end).strip();
    }

    /** Escape a plain string for embedding inside a JSON string value. */
    private static String escapeJson(String s) {
        return s.replace("\\", "\\\\")
                .replace("\"",  "\\\"")
                .replace("\n",  "\\n")
                .replace("\r",  "\\r")
                .replace("\t",  "\\t");
    }

}