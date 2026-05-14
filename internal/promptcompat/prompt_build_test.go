package promptcompat

import "testing"

func TestBuildOpenAIFinalPromptUsesLatestUserTextOnly(t *testing.T) {
	messages := []any{
		map[string]any{"role": "system", "content": "You are helpful"},
		map[string]any{"role": "user", "content": "first user turn"},
		map[string]any{"role": "assistant", "content": "assistant reply"},
		map[string]any{"role": "user", "content": "latest user turn"},
	}
	tools := []any{
		map[string]any{
			"type": "function",
			"function": map[string]any{
				"name":        "search",
				"description": "search docs",
				"parameters":  map[string]any{"type": "object"},
			},
		},
	}

	finalPrompt, toolNames := buildOpenAIFinalPrompt(messages, tools, "", false)
	if finalPrompt != "latest user turn" {
		t.Fatalf("expected only latest user text, got %q", finalPrompt)
	}
	if len(toolNames) != 1 || toolNames[0] != "search" {
		t.Fatalf("unexpected tool names: %#v", toolNames)
	}
}

func TestBuildOpenAIPromptWithToolInstructionsOnlyRetainsCurrentInputToolInstructions(t *testing.T) {
	messages := []any{
		map[string]any{"role": "system", "content": "You are helpful"},
		map[string]any{"role": "user", "content": "call the tool please"},
	}
	tools := []any{
		map[string]any{
			"type": "function",
			"function": map[string]any{
				"name":        "search",
				"description": "search docs",
				"parameters":  map[string]any{"type": "object"},
			},
		},
	}

	finalPrompt, toolNames := BuildOpenAIPromptWithToolInstructionsOnly(messages, tools, "", DefaultToolChoicePolicy(), false)
	if !containsString(finalPrompt, "call the tool please") {
		t.Fatalf("expected original user text in prompt, got %q", finalPrompt)
	}
	if !containsString(finalPrompt, "DS2API_TOOLS.txt") || !containsString(finalPrompt, "TOOL CALL FORMAT") {
		t.Fatalf("expected tool instructions-only prompt to keep current-input tool guidance, got %q", finalPrompt)
	}
	if len(toolNames) != 1 || toolNames[0] != "search" {
		t.Fatalf("unexpected tool names: %#v", toolNames)
	}
}

func TestBuildOpenAIToolsContextTranscriptContainsDescriptions(t *testing.T) {
	tools := []any{
		map[string]any{
			"type": "function",
			"function": map[string]any{
				"name":        "search",
				"description": "search docs",
				"parameters":  map[string]any{"type": "object"},
			},
		},
	}

	transcript, toolNames := BuildOpenAIToolsContextTranscript(tools, DefaultToolChoicePolicy())
	if len(toolNames) != 1 || toolNames[0] != "search" {
		t.Fatalf("unexpected tool names: %#v", toolNames)
	}
	for _, want := range []string{"# DS2API_TOOLS.txt", "You have access to these tools", "Tool: search", "Description: search docs"} {
		if transcript == "" || !containsString(transcript, want) {
			t.Fatalf("expected transcript to contain %q, got %q", want, transcript)
		}
	}
}

func TestBuildOpenAIFinalPromptFallsBackToLastNonEmptyContent(t *testing.T) {
	messages := []any{
		map[string]any{"role": "system", "content": "system only"},
	}

	finalPrompt, _ := buildOpenAIFinalPrompt(messages, nil, "", false)
	if finalPrompt != "system only" {
		t.Fatalf("expected fallback to last non-empty content, got %q", finalPrompt)
	}
}

func containsString(s, sub string) bool {
	return len(sub) == 0 || len(s) >= len(sub) && (s == sub || indexString(s, sub) >= 0)
}

func indexString(s, sub string) int {
	for i := 0; i+len(sub) <= len(s); i++ {
		if s[i:i+len(sub)] == sub {
			return i
		}
	}
	return -1
}
