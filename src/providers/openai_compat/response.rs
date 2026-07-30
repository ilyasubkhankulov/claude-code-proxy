use anyhow::{Context, Result, bail};
use base64::Engine;
use serde::Deserialize;
use serde_json::{Value, json};

/// Deserialize a field that may be present-but-`null` into its `Default`.
///
/// `#[serde(default)]` only fills a field when it is *absent*; a field that is
/// explicitly `null` still fails to deserialize into a non-`Option` type such as
/// `Vec<T>`. Some OpenAI-compatible upstreams (e.g. Arcee) send
/// `"tool_calls": null` rather than omitting it, so pair this with `default` to
/// accept both the absent and the explicit-`null` forms.
pub(crate) fn null_to_default<'de, D, T>(deserializer: D) -> std::result::Result<T, D::Error>
where
    D: serde::Deserializer<'de>,
    T: Default + Deserialize<'de>,
{
    Ok(Option::<T>::deserialize(deserializer)?.unwrap_or_default())
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct Usage {
    #[serde(default)]
    pub prompt_tokens: u64,
    #[serde(default)]
    pub completion_tokens: u64,
    #[serde(default)]
    pub prompt_tokens_details: Option<PromptTokenDetails>,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct PromptTokenDetails {
    #[serde(default)]
    pub cached_tokens: u64,
}

#[derive(Debug, Deserialize)]
struct ChatResponse {
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    choices: Vec<Choice>,
    #[serde(default)]
    usage: Option<Usage>,
}

#[derive(Debug, Deserialize)]
struct Choice {
    message: ResponseMessage,
    #[serde(default)]
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ResponseMessage {
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    reasoning_content: Option<String>,
    #[serde(default)]
    reasoning: Option<String>,
    #[serde(default, deserialize_with = "null_to_default")]
    tool_calls: Vec<ToolCall>,
}

#[derive(Debug, Deserialize)]
struct ToolCall {
    id: String,
    function: ToolFunction,
}

#[derive(Debug, Deserialize)]
struct ToolFunction {
    name: String,
    arguments: String,
}

pub fn translate_response(
    bytes: &[u8],
    fallback_message_id: &str,
    model: &str,
    provider: &str,
) -> Result<Value> {
    let response: ChatResponse =
        serde_json::from_slice(bytes).context("invalid Chat Completions response")?;
    let choice = response
        .choices
        .into_iter()
        .next()
        .context("Chat Completions response has no choices")?;
    let mut content = Vec::new();
    let reasoning = choice
        .message
        .reasoning_content
        .or(choice.message.reasoning)
        .unwrap_or_default();
    if !reasoning.is_empty() {
        content.push(json!({
            "type": "thinking",
            "thinking": reasoning,
            "signature": thinking_signature(provider, fallback_message_id, 0)
        }));
    }
    if let Some(text) = choice.message.content.filter(|text| !text.is_empty()) {
        content.push(json!({"type": "text", "text": text}));
    }
    for tool in choice.message.tool_calls {
        let input: Value = serde_json::from_str(&tool.function.arguments).with_context(|| {
            format!(
                "tool {} returned invalid JSON arguments",
                tool.function.name
            )
        })?;
        content.push(json!({
            "type": "tool_use",
            "id": tool.id,
            "name": tool.function.name,
            "input": input
        }));
    }

    let has_tools = content
        .iter()
        .any(|block| block.get("type").and_then(Value::as_str) == Some("tool_use"));
    let stop_reason = match choice.finish_reason.as_deref() {
        Some("length") => "max_tokens",
        Some("tool_calls") => "tool_use",
        Some("stop") | None => {
            if has_tools {
                "tool_use"
            } else {
                "end_turn"
            }
        }
        Some(_) => "end_turn",
    };
    let usage = anthropic_usage(response.usage.as_ref());
    let id = response
        .id
        .unwrap_or_else(|| fallback_message_id.to_string());
    if content.is_empty() && stop_reason != "tool_use" {
        bail!("Chat Completions response contains no assistant content");
    }

    Ok(json!({
        "id": id,
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": null,
        "usage": usage
    }))
}

pub fn anthropic_usage(usage: Option<&Usage>) -> Value {
    let usage = usage.cloned().unwrap_or_default();
    let cached = usage
        .prompt_tokens_details
        .as_ref()
        .map(|details| details.cached_tokens)
        .unwrap_or(0);
    json!({
        "input_tokens": usage.prompt_tokens.saturating_sub(cached),
        "output_tokens": usage.completion_tokens,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": cached
    })
}

pub fn thinking_signature(provider: &str, message_id: &str, index: usize) -> String {
    let input = format!("ccp:{provider}:v1:{message_id}:{index}");
    base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(input.as_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn translates_reasoning_tools_and_cached_usage() {
        let response = br#"{
          "id":"chat_1",
          "choices":[{"message":{"content":"done","reasoning_content":"think","tool_calls":[{"id":"call_1","function":{"name":"search","arguments":"{\"q\":\"rust\"}"}}]},"finish_reason":"tool_calls"}],
          "usage":{"prompt_tokens":10,"completion_tokens":4,"prompt_tokens_details":{"cached_tokens":3}}
        }"#;
        let translated = translate_response(response, "msg_1", "model", "custom").unwrap();
        assert_eq!(translated["stop_reason"], "tool_use");
        assert_eq!(translated["usage"]["input_tokens"], 7);
        assert_eq!(translated["content"][0]["type"], "thinking");
        assert_eq!(translated["content"][2]["input"]["q"], "rust");
    }

    #[test]
    fn tolerates_explicit_null_tool_calls() {
        // Arcee and some other OpenAI-compatible upstreams send
        // "tool_calls": null rather than omitting it. That must not fail
        // deserialization (previously a 502 "invalid Chat Completions response").
        let response = br#"{
          "id":"chat_1",
          "choices":[{"message":{"content":"OK","role":"assistant","tool_calls":null,"function_call":null,"reasoning_content":"brief"},"finish_reason":"stop"}],
          "usage":{"prompt_tokens":19,"completion_tokens":62,"completion_tokens_details":{"reasoning_tokens":59},"prompt_tokens_details":{"cached_tokens":0,"audio_tokens":null}}
        }"#;
        let translated = translate_response(response, "msg_1", "model", "custom").unwrap();
        assert_eq!(translated["stop_reason"], "end_turn");
        assert_eq!(translated["content"][0]["type"], "thinking");
        assert_eq!(translated["content"][1]["type"], "text");
        assert_eq!(translated["content"][1]["text"], "OK");
    }
}
