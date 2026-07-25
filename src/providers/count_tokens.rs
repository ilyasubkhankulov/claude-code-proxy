use crate::anthropic::schema::MessagesRequest;

pub const IMAGE_TOKEN_ESTIMATE: u64 = 2000;
pub const MESSAGE_OVERHEAD_TOKENS: u64 = 4;

pub fn count_tokens(req: &MessagesRequest) -> u64 {
    let mut total = 0u64;
    if let Some(system) = req.extra.get("system") {
        total += count_system_tokens(system);
    }
    for message in &req.messages {
        total += count_message_tokens(&message.content);
    }
    total += req.messages.len() as u64 * MESSAGE_OVERHEAD_TOKENS;
    if let Some(tools) = req.extra.get("tools").and_then(|value| value.as_array()) {
        total += count_tool_tokens(tools);
    }
    total
}

fn count_system_tokens(system: &serde_json::Value) -> u64 {
    match system {
        serde_json::Value::String(text) => approx_token_count(text),
        serde_json::Value::Array(blocks) => blocks
            .iter()
            .filter_map(|block| block.get("text").and_then(|value| value.as_str()))
            .filter(|text| !text.starts_with("x-anthropic-billing-header:"))
            .map(approx_token_count)
            .sum(),
        _ => 0,
    }
}

fn count_message_tokens(content: &serde_json::Value) -> u64 {
    match content {
        serde_json::Value::String(text) => approx_token_count(text),
        serde_json::Value::Array(blocks) => blocks.iter().map(count_content_block_tokens).sum(),
        _ => 0,
    }
}

fn count_content_block_tokens(block: &serde_json::Value) -> u64 {
    match block.get("type").and_then(|value| value.as_str()) {
        Some("text") => block
            .get("text")
            .and_then(|value| value.as_str())
            .map(approx_token_count)
            .unwrap_or(0),
        Some("image") => IMAGE_TOKEN_ESTIMATE,
        Some("thinking") => block
            .get("thinking")
            .and_then(|value| value.as_str())
            .map(approx_token_count)
            .unwrap_or(0),
        Some("tool_use") => {
            let name = block
                .get("name")
                .and_then(|value| value.as_str())
                .map(approx_token_count)
                .unwrap_or(0);
            let input = block
                .get("input")
                .map(|value| approx_token_count(&serde_json::to_string(value).unwrap_or_default()))
                .unwrap_or(0);
            name + input
        }
        Some("tool_result") => block.get("content").map(count_message_tokens).unwrap_or(0),
        _ => 0,
    }
}

fn count_tool_tokens(tools: &[serde_json::Value]) -> u64 {
    tools
        .iter()
        .map(|tool| {
            let name = tool
                .get("name")
                .and_then(|value| value.as_str())
                .map(approx_token_count)
                .unwrap_or(0);
            let description = tool
                .get("description")
                .and_then(|value| value.as_str())
                .map(approx_token_count)
                .unwrap_or(0);
            let schema = tool
                .get("input_schema")
                .map(|value| approx_token_count(&serde_json::to_string(value).unwrap_or_default()))
                .unwrap_or(0);
            name + description + schema
        })
        .sum()
}

fn approx_token_count(text: &str) -> u64 {
    if text.is_empty() {
        return 0;
    }
    let mut count = 0u64;
    let mut in_word = false;
    for character in text.chars() {
        if character.is_alphanumeric() || matches!(character, '-' | '_') {
            if !in_word {
                count += 1;
                in_word = true;
            }
        } else {
            in_word = false;
            if !character.is_whitespace() {
                count += 1;
            }
        }
    }
    count.max(1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn estimate_is_monotonic_and_counts_images() {
        let text: MessagesRequest = serde_json::from_value(json!({
            "messages": [{"role":"user","content":"hello world"}]
        }))
        .unwrap();
        let image: MessagesRequest = serde_json::from_value(json!({
            "messages": [{"role":"user","content":[
                {"type":"text","text":"hello world"},
                {"type":"image","source":{"type":"base64","media_type":"image/png","data":"abc"}}
            ]}]
        }))
        .unwrap();
        assert!(count_tokens(&image) >= count_tokens(&text) + IMAGE_TOKEN_ESTIMATE);
    }
}
