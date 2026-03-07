package io.hyperstack4j.tokenizer;

import static org.assertj.core.api.Assertions.assertThat;

import java.util.List;

import org.junit.jupiter.api.Test;

class ChatTemplateTest {

    private final List<ChatMessage> messages = List.of(
            ChatMessage.system("You are helpful."),
            ChatMessage.user("Hello!"),
            ChatMessage.assistant("Hi there!")
    );

    @Test
    void llama3_contains_correct_special_tokens() {
        String prompt = ChatTemplate.llama3().format(messages);

        assertThat(prompt).startsWith("<|begin_of_text|>");
        assertThat(prompt).contains("<|start_header_id|>system<|end_header_id|>");
        assertThat(prompt).contains("<|start_header_id|>user<|end_header_id|>");
        assertThat(prompt).contains("<|eot_id|>");
        assertThat(prompt).endsWith("<|start_header_id|>assistant<|end_header_id|>\n\n");
    }

    @Test
    void mistral_wraps_user_in_inst_tags() {
        String prompt = ChatTemplate.mistral().format(messages);

        assertThat(prompt).contains("[INST]");
        assertThat(prompt).contains("[/INST]");
        // System message should be prepended into first user turn
        assertThat(prompt).contains("You are helpful.");
    }

    @Test
    void gemma_uses_start_of_turn_tokens() {
        String prompt = ChatTemplate.gemma().format(messages);

        assertThat(prompt).contains("<start_of_turn>user");
        assertThat(prompt).contains("<start_of_turn>model");
        assertThat(prompt).contains("<end_of_turn>");
        assertThat(prompt).endsWith("<start_of_turn>model\n");
    }

    @Test
    void chatml_uses_im_start_end_tokens() {
        String prompt = ChatTemplate.chatml().format(messages);

        assertThat(prompt).contains("<|im_start|>system");
        assertThat(prompt).contains("<|im_start|>user");
        assertThat(prompt).contains("<|im_end|>");
        assertThat(prompt).endsWith("<|im_start|>assistant\n");
    }

    @Test
    void all_templates_include_message_content() {
        for (ChatTemplate t : ChatTemplate.BUILT_IN.values()) {
            String prompt = t.format(messages);
            assertThat(prompt)
                    .as("Template %s should contain user message", t.modelType())
                    .contains("Hello!");
        }
    }

    @Test
    void forModelType_falls_back_to_chatml_for_unknown() {
        ChatTemplate t = ChatTemplate.forModelType("some-unknown-model");
        assertThat(t.modelType()).isEqualTo("chatml");
    }

    @Test
    void forModelType_is_case_insensitive() {
        assertThat(ChatTemplate.forModelType("LLaMA3").modelType()).isEqualTo("llama3");
        assertThat(ChatTemplate.forModelType("MISTRAL").modelType()).isEqualTo("mistral");
    }
}
