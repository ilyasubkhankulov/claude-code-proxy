use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessagesRequest {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(default)]
    pub messages: Vec<Message>,
    #[serde(default, skip_serializing_if = "is_false")]
    pub stream: bool,
    #[serde(default, flatten)]
    pub extra: serde_json::Map<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: String,
    pub content: serde_json::Value,
    #[serde(default, flatten)]
    pub extra: serde_json::Map<String, serde_json::Value>,
}

fn is_false(value: &bool) -> bool {
    !*value
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn messages_request_round_trip_preserves_native_fields() {
        let original = json!({
            "model": "anthropic/claude-sonnet-5",
            "max_tokens": 256,
            "stream": true,
            "system": [{"type": "text", "text": "system", "cache_control": {"type": "ephemeral"}}],
            "tools": [{"name": "lookup", "input_schema": {"type": "object"}}],
            "thinking": {"type": "enabled", "budget_tokens": 128},
            "future_top_level": {"enabled": true},
            "messages": [{
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": "tool_1",
                    "content": [{"type": "text", "text": "result", "future_block": 1}]
                }],
                "future_message": "preserved"
            }]
        });

        let request: MessagesRequest = serde_json::from_value(original.clone()).unwrap();
        assert_eq!(serde_json::to_value(request).unwrap(), original);
    }

    #[test]
    fn messages_request_omits_absent_optional_defaults() {
        let request: MessagesRequest = serde_json::from_value(json!({
            "messages": [{"role": "user", "content": "hello"}]
        }))
        .unwrap();
        let serialized = serde_json::to_value(request).unwrap();
        assert!(serialized.get("model").is_none());
        assert!(serialized.get("max_tokens").is_none());
        assert!(serialized.get("stream").is_none());
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CountTokensResponse {
    pub input_tokens: u64,
}
