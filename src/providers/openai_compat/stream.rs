use std::collections::BTreeMap;

use anyhow::{Context, Result, bail};
use serde::Deserialize;
use serde_json::{Value, json};

use crate::anthropic::sse::encode_sse_event;

use super::response::{Usage, anthropic_usage, null_to_default, thinking_signature};

const MAX_SSE_FRAME_BYTES: usize = 1024 * 1024;

#[derive(Default)]
pub struct SseDecoder {
    frame: Vec<u8>,
    line_start: usize,
    skip_lf: bool,
}

impl SseDecoder {
    fn push(&mut self, input: &[u8]) -> Result<Vec<String>> {
        let mut events = Vec::new();
        for &byte in input {
            if self.skip_lf {
                self.skip_lf = false;
                if byte == b'\n' {
                    continue;
                }
            }
            match byte {
                b'\n' => self.end_line(&mut events)?,
                b'\r' => {
                    self.end_line(&mut events)?;
                    self.skip_lf = true;
                }
                _ => {
                    if self.frame.len() >= MAX_SSE_FRAME_BYTES {
                        bail!("OpenAI-compatible SSE frame exceeds size limit");
                    }
                    self.frame.push(byte);
                }
            }
        }
        Ok(events)
    }

    fn end_line(&mut self, events: &mut Vec<String>) -> Result<()> {
        if self.frame.len() == self.line_start {
            if !self.frame.is_empty() {
                let frame = std::str::from_utf8(&self.frame).context("SSE frame is not UTF-8")?;
                let data = frame
                    .lines()
                    .filter(|line| !line.starts_with(':'))
                    .filter_map(|line| line.strip_prefix("data:").map(|value| value.trim_start()))
                    .collect::<Vec<_>>()
                    .join("\n");
                if !data.is_empty() {
                    events.push(data);
                }
            }
            self.frame.clear();
            self.line_start = 0;
        } else {
            self.frame.push(b'\n');
            self.line_start = self.frame.len();
        }
        Ok(())
    }

    fn finish(&self) -> Result<()> {
        if self.frame.is_empty() {
            Ok(())
        } else {
            bail!("OpenAI-compatible SSE stream ended with an incomplete frame")
        }
    }
}

#[derive(Debug, Deserialize)]
struct Chunk {
    #[serde(default)]
    choices: Vec<Choice>,
    #[serde(default)]
    usage: Option<Usage>,
    #[serde(default)]
    error: Option<ErrorEnvelope>,
}

#[derive(Debug, Deserialize)]
struct ErrorEnvelope {
    #[serde(default)]
    message: Option<String>,
}

#[derive(Debug, Deserialize)]
struct Choice {
    #[serde(default)]
    delta: Option<Delta>,
    #[serde(default)]
    finish_reason: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
struct Delta {
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    reasoning_content: Option<String>,
    #[serde(default)]
    reasoning: Option<String>,
    #[serde(default, deserialize_with = "null_to_default")]
    tool_calls: Vec<ToolCallDelta>,
}

#[derive(Debug, Deserialize)]
struct ToolCallDelta {
    index: usize,
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    function: Option<FunctionDelta>,
}

#[derive(Debug, Deserialize)]
struct FunctionDelta {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    arguments: Option<String>,
}

#[derive(Default)]
struct ToolState {
    block_index: Option<usize>,
    id: String,
    name: String,
    pending_arguments: String,
}

pub struct StreamTranslator {
    decoder: SseDecoder,
    provider: String,
    message_id: String,
    model: String,
    started: bool,
    finished: bool,
    next_block_index: usize,
    thinking_index: Option<usize>,
    text_index: Option<usize>,
    tools: BTreeMap<usize, ToolState>,
    finish_reason: Option<String>,
    usage: Option<Usage>,
}

impl StreamTranslator {
    pub fn new(provider: String, message_id: String, model: String) -> Self {
        Self {
            decoder: SseDecoder::default(),
            provider,
            message_id,
            model,
            started: false,
            finished: false,
            next_block_index: 0,
            thinking_index: None,
            text_index: None,
            tools: BTreeMap::new(),
            finish_reason: None,
            usage: None,
        }
    }

    pub fn push(&mut self, bytes: &[u8]) -> Result<Vec<u8>> {
        let mut out = Vec::new();
        for data in self.decoder.push(bytes)? {
            if data.trim() == "[DONE]" {
                out.extend(self.finalize()?);
                continue;
            }
            let chunk: Chunk =
                serde_json::from_str(&data).context("malformed Chat Completions SSE event")?;
            if let Some(error) = chunk.error {
                bail!(
                    "upstream stream error: {}",
                    error.message.unwrap_or_else(|| "unknown error".into())
                );
            }
            if let Some(usage) = chunk.usage {
                self.usage = Some(usage);
            }
            for choice in chunk.choices {
                if let Some(reason) = choice.finish_reason {
                    self.finish_reason = Some(reason);
                }
                if let Some(delta) = choice.delta {
                    self.render_delta(delta, &mut out)?;
                }
            }
        }
        Ok(out)
    }

    pub fn finish(&mut self) -> Result<Vec<u8>> {
        self.decoder.finish()?;
        if self.finished {
            return Ok(Vec::new());
        }
        if self.finish_reason.is_none() {
            bail!("OpenAI-compatible SSE stream ended before a finish reason");
        }
        self.finalize()
    }

    pub fn is_finished(&self) -> bool {
        self.finished
    }

    fn render_delta(&mut self, delta: Delta, out: &mut Vec<u8>) -> Result<()> {
        let reasoning = delta
            .reasoning_content
            .or(delta.reasoning)
            .unwrap_or_default();
        if !reasoning.is_empty() {
            self.ensure_message_start(out);
            if self.thinking_index.is_none() {
                let index = self.allocate_block();
                self.thinking_index = Some(index);
                emit(
                    out,
                    "content_block_start",
                    json!({
                        "type":"content_block_start","index":index,
                        "content_block":{"type":"thinking","thinking":""}
                    }),
                );
            }
            let index = self.thinking_index.unwrap_or(0);
            emit(
                out,
                "content_block_delta",
                json!({
                    "type":"content_block_delta","index":index,
                    "delta":{"type":"thinking_delta","thinking":reasoning}
                }),
            );
        }

        if let Some(text) = delta.content.filter(|text| !text.is_empty()) {
            self.close_thinking(out);
            self.ensure_message_start(out);
            if self.text_index.is_none() {
                let index = self.allocate_block();
                self.text_index = Some(index);
                emit(
                    out,
                    "content_block_start",
                    json!({
                        "type":"content_block_start","index":index,
                        "content_block":{"type":"text","text":""}
                    }),
                );
            }
            let index = self.text_index.unwrap_or(0);
            emit(
                out,
                "content_block_delta",
                json!({
                    "type":"content_block_delta","index":index,
                    "delta":{"type":"text_delta","text":text}
                }),
            );
        }

        if !delta.tool_calls.is_empty() {
            self.close_thinking(out);
            self.close_text(out);
        }
        for call in delta.tool_calls {
            let mut start = None;
            let mut argument_delta = None;
            {
                let state = self.tools.entry(call.index).or_default();
                if let Some(id) = call.id {
                    state.id.push_str(&id);
                }
                if let Some(function) = call.function {
                    if let Some(name) = function.name {
                        state.name.push_str(&name);
                    }
                    if let Some(arguments) = function.arguments.filter(|args| !args.is_empty()) {
                        if state.block_index.is_some() {
                            argument_delta = Some(arguments);
                        } else {
                            state.pending_arguments.push_str(&arguments);
                        }
                    }
                }
                if state.block_index.is_none()
                    && !state.id.is_empty()
                    && !state.name.is_empty()
                    && !state.pending_arguments.is_empty()
                {
                    start = Some((
                        state.id.clone(),
                        state.name.clone(),
                        std::mem::take(&mut state.pending_arguments),
                    ));
                }
            }
            if let Some((id, name, pending)) = start {
                self.ensure_message_start(out);
                let index = self.allocate_block();
                self.tools
                    .get_mut(&call.index)
                    .expect("tool exists")
                    .block_index = Some(index);
                emit(
                    out,
                    "content_block_start",
                    json!({
                        "type":"content_block_start","index":index,
                        "content_block":{"type":"tool_use","id":id,"name":name,"input":{}}
                    }),
                );
                if !pending.is_empty() {
                    emit_tool_delta(out, index, pending);
                }
            }
            if let Some(arguments) = argument_delta
                && let Some(index) = self
                    .tools
                    .get(&call.index)
                    .and_then(|tool| tool.block_index)
            {
                emit_tool_delta(out, index, arguments);
            }
        }
        Ok(())
    }

    fn finalize(&mut self) -> Result<Vec<u8>> {
        if self.finished {
            return Ok(Vec::new());
        }
        let mut out = Vec::new();
        self.close_thinking(&mut out);
        self.close_text(&mut out);
        let unopened = self
            .tools
            .iter()
            .filter(|(_, tool)| tool.block_index.is_none())
            .map(|(tool_index, tool)| {
                (
                    *tool_index,
                    tool.id.clone(),
                    tool.name.clone(),
                    tool.pending_arguments.clone(),
                )
            })
            .collect::<Vec<_>>();
        for (tool_index, id, name, arguments) in unopened {
            if id.is_empty() || name.is_empty() {
                bail!("tool call ended before id and function name were received");
            }
            self.ensure_message_start(&mut out);
            let index = self.allocate_block();
            self.tools
                .get_mut(&tool_index)
                .expect("tool exists")
                .block_index = Some(index);
            emit(
                &mut out,
                "content_block_start",
                json!({
                    "type":"content_block_start","index":index,
                    "content_block":{"type":"tool_use","id":id,"name":name,"input":{}}
                }),
            );
            if !arguments.is_empty() {
                emit_tool_delta(&mut out, index, arguments);
            }
        }
        for tool in self.tools.values() {
            let index = tool.block_index.expect("all tools opened above");
            emit(
                &mut out,
                "content_block_stop",
                json!({"type":"content_block_stop","index":index}),
            );
        }
        self.ensure_message_start(&mut out);
        let has_tools = self.tools.values().any(|tool| tool.block_index.is_some());
        let stop_reason = match self.finish_reason.as_deref() {
            Some("length") => "max_tokens",
            Some("tool_calls") => "tool_use",
            _ if has_tools => "tool_use",
            _ => "end_turn",
        };
        emit(
            &mut out,
            "message_delta",
            json!({
                "type":"message_delta",
                "delta":{"stop_reason":stop_reason,"stop_sequence":null},
                "usage":anthropic_usage(self.usage.as_ref())
            }),
        );
        emit(&mut out, "message_stop", json!({"type":"message_stop"}));
        self.finished = true;
        Ok(out)
    }

    fn ensure_message_start(&mut self, out: &mut Vec<u8>) {
        if self.started {
            return;
        }
        emit(
            out,
            "message_start",
            json!({
                "type":"message_start",
                "message":{
                    "id":self.message_id,"type":"message","role":"assistant","model":self.model,
                    "content":[],"stop_reason":null,"stop_sequence":null,
                    "usage":{"input_tokens":0,"output_tokens":0}
                }
            }),
        );
        self.started = true;
    }

    fn allocate_block(&mut self) -> usize {
        let index = self.next_block_index;
        self.next_block_index += 1;
        index
    }

    fn close_thinking(&mut self, out: &mut Vec<u8>) {
        if let Some(index) = self.thinking_index.take() {
            emit(
                out,
                "content_block_delta",
                json!({
                    "type":"content_block_delta","index":index,
                    "delta":{"type":"signature_delta","signature":thinking_signature(&self.provider, &self.message_id, index)}
                }),
            );
            emit(
                out,
                "content_block_stop",
                json!({"type":"content_block_stop","index":index}),
            );
        }
    }

    fn close_text(&mut self, out: &mut Vec<u8>) {
        if let Some(index) = self.text_index.take() {
            emit(
                out,
                "content_block_stop",
                json!({"type":"content_block_stop","index":index}),
            );
        }
    }
}

fn emit(out: &mut Vec<u8>, event: &str, data: Value) {
    out.extend(encode_sse_event(Some(event), &data.to_string()));
}

fn emit_tool_delta(out: &mut Vec<u8>, index: usize, partial_json: String) {
    emit(
        out,
        "content_block_delta",
        json!({
            "type":"content_block_delta","index":index,
            "delta":{"type":"input_json_delta","partial_json":partial_json}
        }),
    );
}

pub fn stream_error(message: &str) -> Vec<u8> {
    encode_sse_event(
        Some("error"),
        &json!({"type":"error","error":{"type":"api_error","message":message}}).to_string(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn streams_fragmented_reasoning_text_and_tools() {
        let upstream = concat!(
            "data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"think\"}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"content\":\"answer\"}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",\"function\":{\"name\":\"search\",\"arguments\":\"{\\\"q\\\":\"}}]}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"\\\"rust\\\"}\"}}]},\"finish_reason\":\"tool_calls\"}]}\n\n",
            "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":5,\"completion_tokens\":2}}\n\n",
            "data: [DONE]\n\n"
        );
        let mut translator = StreamTranslator::new("custom".into(), "msg_1".into(), "model".into());
        let split = upstream.len() / 2;
        let mut output = translator.push(&upstream.as_bytes()[..split]).unwrap();
        output.extend(translator.push(&upstream.as_bytes()[split..]).unwrap());
        let output = String::from_utf8(output).unwrap();
        assert!(output.contains("thinking_delta"));
        assert!(output.contains("text_delta"));
        assert!(output.contains("input_json_delta"));
        assert!(output.contains("message_stop"));
    }

    #[test]
    fn tolerates_explicit_null_tool_calls_in_delta() {
        // Some OpenAI-compatible upstreams (e.g. Arcee) put "tool_calls": null
        // in the delta. That must parse rather than aborting the stream.
        let upstream = concat!(
            "data: {\"choices\":[{\"delta\":{\"role\":\"assistant\",\"content\":\"\",\"tool_calls\":null}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"content\":\"OK\",\"tool_calls\":null},\"finish_reason\":\"stop\"}]}\n\n",
            "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":5,\"completion_tokens\":2}}\n\n",
            "data: [DONE]\n\n"
        );
        let mut translator = StreamTranslator::new("custom".into(), "msg_1".into(), "model".into());
        let output = translator.push(upstream.as_bytes()).unwrap();
        let output = String::from_utf8(output).unwrap();
        assert!(output.contains("text_delta"));
        assert!(output.contains("OK"));
        assert!(output.contains("message_stop"));
    }
}
