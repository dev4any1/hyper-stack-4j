package io.hyperstack4j.node;

public interface InferencePipeline {
    float[] forward(String requestId, int[] tokens, int startPos);
    int vocabSize();
}