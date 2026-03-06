package io.hyperstack4j.tokenizer;

/**
 * A single turn in a chat conversation.
 * Role is one of: "system", "user", "assistant".
 */
public record ChatMessage(String role, String content) {

    public ChatMessage {
        if (role == null || role.isBlank())
            throw new IllegalArgumentException("role must not be blank");
        if (content == null)
            throw new IllegalArgumentException("content must not be null");
        role = role.strip().toLowerCase();
    }

    public static ChatMessage system(String content)    { return new ChatMessage("system",    content); }
    public static ChatMessage user(String content)      { return new ChatMessage("user",      content); }
    public static ChatMessage assistant(String content) { return new ChatMessage("assistant", content); }

    public boolean isSystem()    { return "system".equals(role); }
    public boolean isUser()      { return "user".equals(role); }
    public boolean isAssistant() { return "assistant".equals(role); }
}
