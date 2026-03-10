package io.hyperstack4j.tokenizer;

/**
 * Core tokenizer contract.
 *
 * Called twice per generated token: 1. encode() — once at request start,
 * converts full prompt to token IDs 2. decodeToken() — once per output token,
 * for real-time streaming to client
 *
 * Implementations must be thread-safe — one instance shared across all
 * requests.
 */
public interface Tokenizer {

	/**
	 * Encode text into token IDs. Prepends BOS token if the model requires it.
	 */
	int[] encode(String text);

	/**
	 * Decode a full sequence of token IDs back to text.
	 */
	String decode(int[] tokenIds);

	/**
	 * Decode a single token ID to its text piece. Used for real-time streaming —
	 * called once per generated token. May return empty string for special tokens
	 * (BOS, EOS, padding).
	 */
	String decodeToken(int tokenId);

	/** Beginning-of-sequence token ID. */
	int bosTokenId();

	/** End-of-sequence token ID. */
	int eosTokenId();

	/** Padding token ID. */
	int padTokenId();

	/** Full vocabulary size. */
	int vocabSize();

	/** Model family identifier e.g. "llama3", "mistral", "gemma". */
	String modelType();

	/** Whether this tokenizer has been loaded and is ready for use. */
	boolean isReady();
}
