package io.hyperstack4j.integration;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.util.List;

import io.hyperstack4j.coordinator.GenerationLoop;
import io.hyperstack4j.coordinator.GenerationResult;
import io.hyperstack4j.coordinator.InferenceRequest;
import io.hyperstack4j.coordinator.RequestPriority;
import io.hyperstack4j.kvcache.CpuKVCache;
import io.hyperstack4j.kvcache.GpuKVCache;
import io.hyperstack4j.kvcache.KVCacheManager;
import io.hyperstack4j.node.ActivationDtype;
import io.hyperstack4j.sampler.Sampler;
import io.hyperstack4j.sampler.SamplingParams;
import io.hyperstack4j.tokenizer.ChatMessage;
import io.hyperstack4j.tokenizer.StubTokenizer;

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
 *   DTYPE        FLOAT32 | FLOAT16 | INT8   (default: FLOAT32)
 *   MAX_TOKENS   integer                    (default: 200)
 *   TEMPERATURE  float                      (default: 0.7)
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

    // ── Config — reads -DDTYPE=... / -DMAX_TOKENS=... / -DTEMPERATURE=... (exec:exec)
    //            falls back to DTYPE= / MAX_TOKENS= / TEMPERATURE= env vars
    ActivationDtype dtype       = parseDtype(syspropOrEnv("DTYPE",       "FLOAT32"));
    int             maxTokens   = parseIntVal(syspropOrEnv("MAX_TOKENS",  "200"));
    float           temperature = parseFloatVal(syspropOrEnv("TEMPERATURE", "0.7"));

        banner(dtype, maxTokens, temperature);

        // ── Boot cluster ──────────────────────────────────────────────────────
        print(CYAN + "▶ Starting 3-node cluster..." + RESET);

        ClusterHarness harness = ClusterHarness.threeNodes();

        // Shutdown hook — fires on Ctrl-C and on normal exit
        Runtime.getRuntime().addShutdownHook(Thread.ofVirtual().unstarted(() -> {
            print("\n" + YELLOW + "⏹ Shutting down cluster..." + RESET);
            try { harness.stop(); } catch (Exception e) { /* best effort */ }
            print(YELLOW + "✔ Cluster stopped." + RESET);
        }));

        harness.start();
        print(GREEN + "✔ Cluster ready  (" + dtype + " activations)" + RESET + "\n");

        // ── Wire coordinator ──────────────────────────────────────────────────
        ProcessPipelineClient pipeline = new ProcessPipelineClient(
                harness.nodeAddresses(),
                EmbeddedNodeServer.VOCAB_SIZE,
                dtype
        );

        GenerationLoop loop = new GenerationLoop(
                new StubTokenizer(),
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

        // ── REPL ──────────────────────────────────────────────────────────────
        print(DIM + "Type your prompt and press Enter. Type 'exit' or Ctrl-C to quit." + RESET);
        print("");

        BufferedReader stdin = new BufferedReader(new InputStreamReader(System.in));
        String line;

        while (true) {
            System.out.print(BOLD + CYAN + "you> " + RESET);
            System.out.flush();

            line = stdin.readLine();

            if (line == null) break;                          // EOF (pipe closed)
            line = line.strip();
            if (line.isEmpty()) continue;
            if (line.equalsIgnoreCase("exit") ||
                line.equalsIgnoreCase("quit")) break;

            InferenceRequest request = InferenceRequest.of(
                    "tinyllama",
                    List.of(ChatMessage.user(line)),
                    params,
                    RequestPriority.NORMAL
            );

            System.out.print(BOLD + GREEN + "bot> " + RESET);
            System.out.flush();

            long start = System.currentTimeMillis();

            GenerationResult result = loop.generate(request, (piece, tokenId, step) -> {
                System.out.print(piece);
                System.out.flush();
            });

            long elapsed = System.currentTimeMillis() - start;

            System.out.println();   // newline after last token
            System.out.printf(DIM + "     [%d tokens · %d ms · %s]" + RESET + "%n",
                    result.generatedTokens(), elapsed, dtype);
            System.out.println();
        }

        print(YELLOW + "\nbye." + RESET);
        System.exit(0);   // trigger shutdown hook
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    private static void banner(ActivationDtype dtype, int maxTokens, float temperature) {
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
        System.out.println(CYAN + "  hyper-stack-4j  ·  3-node stub cluster  ·  interactive console" + RESET);
        System.out.println(DIM
                + "  dtype=" + dtype
                + "  max_tokens=" + maxTokens
                + "  temperature=" + temperature
                + "  nodes=3 (localhost:19092-19094)"
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
}