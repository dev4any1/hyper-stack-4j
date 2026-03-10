package io.hyperstack4j.tokenizer;

import java.util.List;

/**
 * Applies a ChatTemplate to a list of messages and returns the formatted
 * prompt.
 *
 * Sits between the coordinator's InferenceRequest and the Tokenizer.encode()
 * call:
 *
 * InferenceRequest.messages() → ChatTemplateFormatter.format() → prompt String
 * → Tokenizer.encode() → int[] tokens → InferencePipeline.forward()
 *
 * Thread-safe — stateless, template is immutable.
 */
public final class ChatTemplateFormatter {

	private final ChatTemplate template;

	public ChatTemplateFormatter(ChatTemplate template) {
		if (template == null)
			throw new IllegalArgumentException("template must not be null");
		this.template = template;
	}

	/**
	 * Convenience factory — resolve by model type string. Falls back to ChatML for
	 * unknown model types.
	 */
	public static ChatTemplateFormatter forModelType(String modelType) {
		return new ChatTemplateFormatter(ChatTemplate.forModelType(modelType));
	}

	/**
	 * Format a list of chat messages into a prompt string.
	 *
	 * @param messages ordered list of conversation turns (system, user,
	 *                 assistant...)
	 * @return formatted prompt ready for tokenizer.encode()
	 * @throws IllegalArgumentException if messages is null or empty
	 */
	public String format(List<ChatMessage> messages) {
		if (messages == null || messages.isEmpty())
			throw new IllegalArgumentException("messages must not be null or empty");
		return template.format(messages);
	}

	public String modelType() {
		return template.modelType();
	}
}
