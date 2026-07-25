use anyhow::{Result, bail};
use serde::Serialize;
use serde_json::{Value, json};

use crate::anthropic::schema::MessagesRequest;
use crate::providers::translate_shared::{
    ContentBlock, flatten_system_text, image_block_to_url, image_source_to_url, normalize_content,
};

#[derive(Debug, Clone, Serialize)]
pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Value>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<Value>,
    pub stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream_options: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<Value>,
}

pub fn translate_request(req: &MessagesRequest, model: &str) -> Result<ChatRequest> {
    let mut messages = Vec::new();
    if let Some(system) = flatten_system_text(req.extra.get("system")) {
        messages.push(json!({"role": "system", "content": system}));
    }

    for message in &req.messages {
        let blocks = normalize_content(&message.content, json!({}));
        match message.role.as_str() {
            "user" => push_user_messages(&mut messages, &blocks),
            "assistant" => push_assistant_message(&mut messages, &blocks),
            "system" | "developer" => {
                let text = blocks
                    .iter()
                    .filter_map(|block| match block {
                        ContentBlock::Text { text } => Some(text.as_str()),
                        _ => None,
                    })
                    .collect::<Vec<_>>()
                    .join("\n");
                if !text.is_empty() {
                    messages.push(json!({"role": "system", "content": text}));
                }
            }
            role => bail!("unexpected message role: {role}"),
        }
    }

    let tools = translate_tools(req.extra.get("tools"))?;

    Ok(ChatRequest {
        model: model.to_string(),
        messages,
        tools,
        tool_choice: map_tool_choice(req.extra.get("tool_choice")),
        stream: req.stream,
        stream_options: req.stream.then(|| json!({"include_usage": true})),
        max_tokens: req.max_tokens,
        temperature: req.extra.get("temperature").and_then(Value::as_f64),
        top_p: req.extra.get("top_p").and_then(Value::as_f64),
        stop: req.extra.get("stop_sequences").cloned(),
    })
}

fn translate_tools(value: Option<&Value>) -> Result<Option<Vec<Value>>> {
    let Some(value) = value else {
        return Ok(None);
    };
    let tools = value
        .as_array()
        .ok_or_else(|| anyhow::anyhow!("tools must be an array"))?;
    let mut translated = Vec::with_capacity(tools.len());
    for tool in tools {
        let name = tool
            .get("name")
            .and_then(Value::as_str)
            .filter(|name| !name.is_empty())
            .ok_or_else(|| anyhow::anyhow!("each tool must have a non-empty name"))?;
        let mut function = serde_json::Map::new();
        function.insert("name".into(), json!(name));
        if let Some(description) = tool.get("description").and_then(Value::as_str) {
            function.insert("description".into(), json!(description));
        }
        function.insert(
            "parameters".into(),
            tool.get("input_schema")
                .cloned()
                .unwrap_or_else(|| json!({"type": "object"})),
        );
        translated.push(json!({"type": "function", "function": function}));
    }
    Ok((!translated.is_empty()).then_some(translated))
}

fn map_tool_choice(choice: Option<&Value>) -> Option<Value> {
    match choice {
        Some(Value::String(value)) => match value.as_str() {
            "auto" => None,
            "none" => Some(json!("none")),
            "any" | "required" => Some(json!("required")),
            _ => None,
        },
        Some(Value::Object(choice)) => match choice.get("type").and_then(Value::as_str) {
            Some("auto") => None,
            Some("none") => Some(json!("none")),
            Some("any") => Some(json!("required")),
            Some("tool") => choice
                .get("name")
                .and_then(Value::as_str)
                .map(|name| json!({"type": "function", "function": {"name": name}})),
            _ => None,
        },
        _ => None,
    }
}

fn push_user_messages(out: &mut Vec<Value>, blocks: &[ContentBlock]) {
    let mut parts = Vec::new();
    let flush = |out: &mut Vec<Value>, parts: &mut Vec<Value>| {
        if parts.is_empty() {
            return;
        }
        let all_text = parts
            .iter()
            .all(|part| part.get("type").and_then(Value::as_str) == Some("text"));
        if all_text {
            let text = parts
                .iter()
                .filter_map(|part| part.get("text").and_then(Value::as_str))
                .collect::<String>();
            out.push(json!({"role": "user", "content": text}));
        } else {
            out.push(json!({"role": "user", "content": std::mem::take(parts)}));
        }
        parts.clear();
    };

    for block in blocks {
        match block {
            ContentBlock::Text { text } => parts.push(json!({"type": "text", "text": text})),
            ContentBlock::Image { source } => parts.push(json!({
                "type": "image_url",
                "image_url": {"url": image_source_to_url(source)}
            })),
            ContentBlock::ToolResult {
                tool_use_id,
                content,
                is_error,
            } => {
                flush(out, &mut parts);
                out.push(json!({
                    "role": "tool",
                    "tool_call_id": tool_use_id,
                    "content": tool_result_content(content, is_error.unwrap_or(false))
                }));
            }
            _ => {}
        }
    }
    flush(out, &mut parts);
}

fn tool_result_content(content: &Value, is_error: bool) -> Value {
    let prefix = if is_error {
        "[tool execution error]\n"
    } else {
        ""
    };
    match content {
        Value::String(text) => json!(format!("{prefix}{text}")),
        Value::Array(blocks) => {
            let mut parts = Vec::new();
            if !prefix.is_empty() {
                parts.push(json!({"type": "text", "text": prefix}));
            }
            for block in blocks {
                match block.get("type").and_then(Value::as_str) {
                    Some("text") => parts.push(json!({
                        "type": "text",
                        "text": block.get("text").and_then(Value::as_str).unwrap_or("")
                    })),
                    Some("image") => parts.push(json!({
                        "type": "image_url",
                        "image_url": {"url": image_block_to_url(block)}
                    })),
                    Some(kind) => parts.push(json!({
                        "type": "text",
                        "text": format!("[unsupported content block omitted: {kind}]")
                    })),
                    None => {}
                }
            }
            if parts.len() == 1 && parts[0].get("type").and_then(Value::as_str) == Some("text") {
                return parts[0].get("text").cloned().unwrap_or(Value::Null);
            }
            Value::Array(parts)
        }
        _ => json!(prefix),
    }
}

fn push_assistant_message(out: &mut Vec<Value>, blocks: &[ContentBlock]) {
    let mut text = String::new();
    let mut reasoning = String::new();
    let mut tool_calls = Vec::new();
    for block in blocks {
        match block {
            ContentBlock::Text { text: value } => text.push_str(value),
            ContentBlock::Thinking {
                thinking: value, ..
            } => {
                if !reasoning.is_empty() {
                    reasoning.push_str("\n\n");
                }
                reasoning.push_str(value);
            }
            ContentBlock::ToolUse { id, name, input } => tool_calls.push(json!({
                "id": id,
                "type": "function",
                "function": {"name": name, "arguments": serde_json::to_string(input).unwrap_or_else(|_| "{}".into())}
            })),
            _ => {}
        }
    }
    if text.is_empty() && reasoning.is_empty() && tool_calls.is_empty() {
        return;
    }
    let mut message = serde_json::Map::new();
    message.insert("role".into(), json!("assistant"));
    message.insert("content".into(), json!(text));
    if !reasoning.is_empty() {
        message.insert("reasoning_content".into(), json!(reasoning));
    }
    if !tool_calls.is_empty() {
        message.insert("tool_calls".into(), Value::Array(tool_calls));
    }
    out.push(Value::Object(message));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn translates_tools_and_slash_model() {
        let request: MessagesRequest = serde_json::from_value(json!({
            "model": "org/model",
            "max_tokens": 100,
            "system": "help",
            "messages": [{"role":"user","content":"hello"}],
            "tools": [{"name":"search","input_schema":{"type":"object"}}],
            "tool_choice": {"type":"tool","name":"search"}
        }))
        .unwrap();
        let translated = translate_request(&request, "org/model").unwrap();
        assert_eq!(translated.model, "org/model");
        assert_eq!(translated.messages[0]["role"], "system");
        assert_eq!(
            translated.tool_choice.unwrap()["function"]["name"],
            "search"
        );
    }
}
