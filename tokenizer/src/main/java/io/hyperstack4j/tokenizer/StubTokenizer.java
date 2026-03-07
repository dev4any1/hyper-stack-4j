package io.hyperstack4j.tokenizer;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Lightweight test-double tokenizer for unit and integration testing.
 *
 * Does NOT require DJL or any model file.
 * Splits text on whitespace and assigns deterministic integer IDs.
 * Sufficient for testing everything that uses the Tokenizer interface
 * (coordinator, pipeline, sampler integration) without touching real model files.
 *
 * NOT for production use.
 */
public final class StubTokenizer implements Tokenizer {

    private static final int BOS = 1;
    private static final int EOS = 2;
    private static final int PAD = 0;

    private final Map<String, Integer> vocab   = new ConcurrentHashMap<>();
    private final Map<Integer, String> reverse = new ConcurrentHashMap<>();
    private final AtomicInteger        nextId  = new AtomicInteger(10);

    public StubTokenizer() {
        // pre-register special tokens
        register("<pad>", PAD);
        register("<bos>", BOS);
        register("<eos>", EOS);
    }

    private void register(String token, int id) {
        vocab.put(token, id);
        reverse.put(id, token);
    }

    private int getOrCreate(String token) {
        return vocab.computeIfAbsent(token, t -> {
            int id = nextId.getAndIncrement();
            reverse.put(id, t);
            return id;
        });
    }

    @Override
    public int[] encode(String text) {
        if (text == null || text.isBlank()) return new int[0];
        String[] words = text.strip().split("\\s+");
        int[] ids = new int[words.length];
        for (int i = 0; i < words.length; i++) {
            ids[i] = getOrCreate(words[i]);
        }
        return ids;
    }

    @Override
    public String decode(int[] tokenIds) {
        if (tokenIds == null || tokenIds.length == 0) return "";
        List<String> parts = new ArrayList<>();
        for (int id : tokenIds) {
            String token = reverse.getOrDefault(id, "<unk>");
            if (!token.startsWith("<") || !token.endsWith(">")) {
                parts.add(token);
            }
        }
        return String.join(" ", parts);
    }

    @Override
    public String decodeToken(int tokenId) {
        String token = reverse.getOrDefault(tokenId, "");
        // suppress special tokens for streaming
        if (token.startsWith("<") && token.endsWith(">")) return "";
        return token + " ";
    }

    @Override public int     bosTokenId() { return BOS; }
    @Override public int     eosTokenId() { return EOS; }
    @Override public int     padTokenId() { return PAD; }
    @Override public int     vocabSize()  { return vocab.size(); }
    @Override public String  modelType()  { return "stub"; }
    @Override public boolean isReady()    { return true; }
}