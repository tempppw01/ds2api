package promptcompat

import (
	"ds2api/internal/prompt"
	"strings"
)

func buildOpenAIFinalPrompt(messagesRaw []any, toolsRaw any, traceID string, thinkingEnabled bool) (string, []string) {
	return BuildOpenAIPrompt(messagesRaw, toolsRaw, traceID, DefaultToolChoicePolicy(), thinkingEnabled)
}

func BuildOpenAIPrompt(messagesRaw []any, toolsRaw any, traceID string, toolPolicy ToolChoicePolicy, thinkingEnabled bool) (string, []string) {
	return buildOpenAIPrompt(messagesRaw, toolsRaw, traceID, toolPolicy, thinkingEnabled, true)
}

func BuildOpenAIPromptWithToolInstructionsOnly(messagesRaw []any, toolsRaw any, traceID string, toolPolicy ToolChoicePolicy, thinkingEnabled bool) (string, []string) {
	messages := NormalizeOpenAIMessagesForPrompt(messagesRaw, traceID)
	toolNames := []string{}
	if tools, ok := toolsRaw.([]any); ok && len(tools) > 0 {
		messages, toolNames = injectToolPromptInstructionsOnly(messages, tools, toolPolicy)
	}
	return prompt.MessagesPrepareWithThinking(messages, thinkingEnabled), toolNames
}

func buildOpenAIPrompt(messagesRaw []any, toolsRaw any, traceID string, toolPolicy ToolChoicePolicy, thinkingEnabled bool, includeToolDescriptions bool) (string, []string) {
	messages := NormalizeOpenAIMessagesForPrompt(messagesRaw, traceID)
	_ = includeToolDescriptions
	_ = thinkingEnabled
	return latestUserPrompt(messages), collectAllowedToolNames(toolsRaw, toolPolicy)
}

// BuildOpenAIPromptForAdapter exposes the OpenAI-compatible prompt building flow so
// other protocol adapters (for example Gemini) can reuse the same tool/history
// normalization logic and remain behavior-compatible with chat/completions.
func BuildOpenAIPromptForAdapter(messagesRaw []any, toolsRaw any, traceID string, thinkingEnabled bool) (string, []string) {
	return buildOpenAIFinalPrompt(messagesRaw, toolsRaw, traceID, thinkingEnabled)
}

func latestUserPrompt(messages []map[string]any) string {
	for i := len(messages) - 1; i >= 0; i-- {
		role := strings.ToLower(strings.TrimSpace(asString(messages[i]["role"])))
		if role != "user" {
			continue
		}
		text := strings.TrimSpace(NormalizeOpenAIContentForPrompt(messages[i]["content"]))
		if text != "" {
			return text
		}
	}
	for i := len(messages) - 1; i >= 0; i-- {
		text := strings.TrimSpace(NormalizeOpenAIContentForPrompt(messages[i]["content"]))
		if text != "" {
			return text
		}
	}
	return ""
}

func collectAllowedToolNames(toolsRaw any, policy ToolChoicePolicy) []string {
	declared := extractDeclaredToolNames(toolsRaw)
	if len(declared) == 0 {
		return nil
	}
	if len(policy.Allowed) == 0 {
		return declared
	}
	out := make([]string, 0, len(declared))
	for _, name := range declared {
		if _, ok := policy.Allowed[name]; ok {
			out = append(out, name)
		}
	}
	return out
}
