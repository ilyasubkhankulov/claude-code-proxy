pub mod client;
pub mod request;
pub mod response;
pub mod stream;

use std::convert::Infallible;
use std::pin::Pin;
use std::sync::Arc;

use anyhow::{Result, anyhow, bail};
use async_trait::async_trait;
use axum::{
    Json,
    body::Body,
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
};
use bytes::Bytes;
use futures_util::{Stream, StreamExt};

use crate::anthropic::{
    error::json_error,
    schema::{CountTokensResponse, MessagesRequest},
};
use crate::config::CompatibleProtocol;
use crate::monitor::MonitorHandle;
use crate::provider::{CliHandlers, Provider, RequestContext};
use crate::registry::normalize_incoming_model;

pub struct OpenAiCompatibleProvider {
    name: String,
    api_key_env: String,
    models: Vec<String>,
    protocol: CompatibleProtocol,
    model_rewrites: std::collections::BTreeMap<String, String>,
    client: Arc<client::OpenAiClient>,
}

impl OpenAiCompatibleProvider {
    pub fn new(
        name: String,
        base_url: String,
        api_key_env: String,
        models: Vec<String>,
        protocol: CompatibleProtocol,
        headers: HeaderMap,
        model_rewrites: std::collections::BTreeMap<String, String>,
    ) -> Self {
        Self {
            name,
            api_key_env,
            models,
            protocol,
            model_rewrites,
            client: Arc::new(client::OpenAiClient::new(base_url, headers)),
        }
    }

    /// True when the incoming (normalized) model ID is routable to this
    /// provider: either a configured `models` entry or a `modelRewrites` key.
    fn accepts_model(&self, model: &str) -> bool {
        self.models.iter().any(|candidate| candidate == model)
            || self.model_rewrites.contains_key(model)
    }

    /// The model ID actually sent upstream: the rewrite target if the incoming
    /// ID is a rewrite key, otherwise the incoming ID unchanged.
    fn upstream_model(&self, model: &str) -> String {
        self.model_rewrites
            .get(model)
            .cloned()
            .unwrap_or_else(|| model.to_string())
    }
}

#[async_trait]
impl Provider for OpenAiCompatibleProvider {
    fn name(&self) -> &str {
        &self.name
    }

    fn supported_models(&self) -> Vec<String> {
        let mut models = self.models.clone();
        models.extend(self.model_rewrites.keys().cloned());
        models.sort_unstable();
        models.dedup();
        models
    }

    fn cli(&self) -> &'static dyn CliHandlers {
        &OPENAI_COMPATIBLE_CLI
    }

    async fn handle_messages(&self, body: MessagesRequest, ctx: RequestContext) -> Response {
        let raw_model = body.model.as_deref().unwrap_or_default();
        let model = normalize_incoming_model(raw_model);
        if !self.accepts_model(&model) {
            return json_error(
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
                format!(
                    "Model {raw_model:?} is not configured for provider {:?}",
                    self.name
                ),
            );
        }
        // The client keeps using a recognized/incoming ID (`model`); the wire
        // request carries the rewritten upstream ID when one is configured.
        let upstream_model = self.upstream_model(&model);
        if let Some(monitor) = ctx.monitor.as_ref() {
            monitor.model_resolved(&ctx.req_id, &model);
            monitor.upstream_started(&ctx.req_id);
        }
        let api_key = match std::env::var(&self.api_key_env) {
            Ok(key) if !key.trim().is_empty() => key,
            _ => {
                return json_error(
                    StatusCode::UNAUTHORIZED,
                    "authentication_error",
                    format!(
                        "Provider {:?} requires API key environment variable {}",
                        self.name, self.api_key_env
                    ),
                );
            }
        };
        if self.protocol == CompatibleProtocol::AnthropicMessages {
            let mut native_body = match normalize_anthropic_messages(body) {
                Ok(body) => body,
                Err(error) => {
                    return json_error(
                        StatusCode::BAD_REQUEST,
                        "invalid_request_error",
                        error.to_string(),
                    );
                }
            };
            native_body.model = Some(upstream_model);
            let upstream = match self
                .client
                .send_anthropic(&api_key, &native_body, ctx.anthropic_headers.clone())
                .await
            {
                Ok(response) => response,
                Err(error) => return map_client_error(error),
            };
            return native_response(upstream, native_body.stream, ctx.monitor, ctx.req_id).await;
        }
        let translated = match request::translate_request(&body, &upstream_model) {
            Ok(request) => request,
            Err(error) => {
                return json_error(
                    StatusCode::BAD_REQUEST,
                    "invalid_request_error",
                    error.to_string(),
                );
            }
        };
        let upstream = match self.client.send(&api_key, &translated).await {
            Ok(response) => response,
            Err(error) => return map_client_error(error),
        };
        let message_id = format!("msg_{}", uuid::Uuid::new_v4().simple());

        if body.stream {
            stream_response(
                upstream,
                self.name.clone(),
                message_id,
                upstream_model,
                ctx.monitor,
                ctx.req_id,
            )
        } else {
            let bytes = match read_limited(upstream).await {
                Ok(bytes) => bytes,
                Err(error) => {
                    return json_error(
                        StatusCode::BAD_GATEWAY,
                        "api_error",
                        format!("Failed reading upstream response: {error}"),
                    );
                }
            };
            match response::translate_response(&bytes, &message_id, &upstream_model, &self.name) {
                Ok(value) => {
                    if let Some(monitor) = ctx.monitor.as_ref() {
                        monitor.usage_updated(
                            &ctx.req_id,
                            value
                                .pointer("/usage/input_tokens")
                                .and_then(|v| v.as_u64()),
                            value
                                .pointer("/usage/output_tokens")
                                .and_then(|v| v.as_u64()),
                        );
                    }
                    (StatusCode::OK, Json(value)).into_response()
                }
                Err(error) => json_error(
                    StatusCode::BAD_GATEWAY,
                    "api_error",
                    format!("OpenAI-compatible response translation failed: {error}"),
                ),
            }
        }
    }

    async fn handle_count_tokens(&self, body: MessagesRequest, ctx: RequestContext) -> Response {
        let tokens = crate::providers::count_tokens::count_tokens(&body);
        if let Some(monitor) = ctx.monitor.as_ref() {
            monitor.usage_updated(&ctx.req_id, Some(tokens), None);
        }
        (
            StatusCode::OK,
            Json(CountTokensResponse {
                input_tokens: tokens,
            }),
        )
            .into_response()
    }
}

/// Prepare a request for a native Anthropic Messages gateway.
///
/// The Cloudflare AI Gateway's `/ai/v1/messages` validator is stricter than
/// `api.anthropic.com`. Two constraints were confirmed live against the
/// gateway and are the reason this normalizer is lossy:
///
/// * `system` must be a plain string. A structured content-block array is
///   rejected with `Invalid value at system: expected string, received
///   array`, so system blocks are flattened into a single string here.
/// * the `context_management` beta body field is rejected with `Extra inputs
///   are not permitted` (even though the paired `anthropic-beta` header is
///   forwarded), so it is dropped.
///
/// Because `system` must be a plain string, a `cache_control` breakpoint placed
/// *on a system block* cannot survive the flatten. Anthropic prompt caching is
/// prefix-based, though, and the gateway does accept a top-level `cache_control`
/// field (confirmed live: it drops the cached prefix out of `input_tokens`). So
/// when the client marked the system prompt for caching, we re-express that as a
/// top-level `cache_control`, which auto-applies a breakpoint to the last
/// cacheable block and thereby caches the whole prefix — the stringified system
/// included. This is only done when the client actually asked for system
/// caching, so one-shot clients never pay cache-write costs they did not request.
/// `cache_control` markers on tool and message content blocks are forwarded
/// unchanged and continue to work through the gateway on their own.
///
/// Note: the gateway does not report `cache_creation_input_tokens` /
/// `cache_read_input_tokens`, so cache activity is invisible in usage even
/// though the savings are real (cached tokens leave `input_tokens`).
fn normalize_anthropic_messages(mut body: MessagesRequest) -> Result<MessagesRequest> {
    body.extra.remove("context_management");

    let mut system_text = Vec::new();
    let mut system_cache_control: Option<serde_json::Value> = None;
    if let Some(system) = body.extra.remove("system") {
        append_system_text(&mut system_text, &mut system_cache_control, system)?;
    }

    let mut messages = Vec::with_capacity(body.messages.len());
    for message in body.messages {
        match message.role.as_str() {
            "user" | "assistant" => messages.push(message),
            "system" | "developer" => {
                append_system_text(&mut system_text, &mut system_cache_control, message.content)?;
            }
            role => bail!("unexpected message role for Anthropic Messages: {role}"),
        }
    }
    body.messages = messages;
    if !system_text.is_empty() {
        body.extra.insert(
            "system".to_string(),
            serde_json::Value::String(system_text.join("\n\n")),
        );
    }
    // Recover system-prompt caching that the string-collapse above would
    // otherwise destroy. Reuse the client's own cache_control value so its TTL
    // (5m/1h) is preserved, and never override a top-level marker the client
    // already set.
    if let Some(cache_control) = system_cache_control {
        body.extra
            .entry("cache_control".to_string())
            .or_insert(cache_control);
    }
    Ok(body)
}

fn append_system_text(
    out: &mut Vec<String>,
    cache_control: &mut Option<serde_json::Value>,
    content: serde_json::Value,
) -> Result<()> {
    match content {
        serde_json::Value::String(text) => out.push(text),
        serde_json::Value::Array(blocks) => {
            for block in blocks {
                let text = block
                    .get("text")
                    .and_then(serde_json::Value::as_str)
                    .ok_or_else(|| anyhow!("Anthropic system blocks must contain text"))?;
                out.push(text.to_string());
                // Keep the last system-block breakpoint so it can be re-expressed
                // as a top-level cache_control after the flatten (see caller).
                if let Some(cc) = block.get("cache_control") {
                    *cache_control = Some(cc.clone());
                }
            }
        }
        _ => bail!("Anthropic system content must be a string or content block array"),
    }
    Ok(())
}

const MAX_NON_STREAM_RESPONSE_BYTES: usize = 8 * 1024 * 1024;

async fn native_response(
    upstream: reqwest::Response,
    stream: bool,
    monitor: Option<MonitorHandle>,
    req_id: String,
) -> Response {
    let status = upstream.status();
    let headers = passthrough_headers(upstream.headers());
    if stream {
        let mut saw_bytes = false;
        let body_stream = upstream.bytes_stream().map(move |chunk| {
            if let Ok(bytes) = chunk.as_ref() {
                if !saw_bytes {
                    saw_bytes = true;
                    if let Some(monitor) = monitor.as_ref() {
                        monitor.generation_started(&req_id);
                    }
                }
                if let Some(monitor) = monitor.as_ref() {
                    monitor.stream_progress(&req_id, bytes.len() as u64, 1, None, None);
                }
            }
            chunk
        });
        let mut response = Response::new(Body::from_stream(body_stream));
        *response.status_mut() = status;
        *response.headers_mut() = headers;
        return response;
    }

    let bytes = match read_limited(upstream).await {
        Ok(bytes) => bytes,
        Err(error) => {
            return json_error(
                StatusCode::BAD_GATEWAY,
                "api_error",
                format!("Failed reading upstream response: {error}"),
            );
        }
    };
    if !bytes.is_empty()
        && let Some(monitor) = monitor.as_ref()
    {
        monitor.generation_started(&req_id);
        monitor.stream_progress(&req_id, bytes.len() as u64, 1, None, None);
    }
    let mut response = Response::new(Body::from(bytes));
    *response.status_mut() = status;
    *response.headers_mut() = headers;
    response
}

fn passthrough_headers(upstream: &reqwest::header::HeaderMap) -> HeaderMap {
    let mut headers = HeaderMap::new();
    for (name, value) in upstream {
        let lower = name.as_str();
        if matches!(
            lower,
            "content-type"
                | "cache-control"
                | "retry-after"
                | "request-id"
                | "x-request-id"
                | "cf-ray"
        ) || lower.starts_with("x-ratelimit-")
            || lower.starts_with("anthropic-ratelimit-")
        {
            headers.append(name.clone(), value.clone());
        }
    }
    headers
}

async fn read_limited(response: reqwest::Response) -> Result<Vec<u8>> {
    let mut stream = response.bytes_stream();
    let mut bytes = Vec::new();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        if bytes.len().saturating_add(chunk.len()) > MAX_NON_STREAM_RESPONSE_BYTES {
            bail!("upstream response exceeds 8 MiB limit");
        }
        bytes.extend_from_slice(&chunk);
    }
    Ok(bytes)
}

fn map_client_error(error: client::ClientError) -> Response {
    let (status, kind) = match error.status.map(|status| status.as_u16()) {
        Some(400 | 404 | 422) => (StatusCode::BAD_REQUEST, "invalid_request_error"),
        Some(401) => (StatusCode::UNAUTHORIZED, "authentication_error"),
        Some(402) => (StatusCode::PAYMENT_REQUIRED, "permission_error"),
        Some(403) => (StatusCode::FORBIDDEN, "permission_error"),
        Some(429) => (StatusCode::TOO_MANY_REQUESTS, "rate_limit_error"),
        _ => (StatusCode::BAD_GATEWAY, "api_error"),
    };
    let mut response = json_error(status, kind, error.message);
    if let Some(retry_after) = error.retry_after
        && let Ok(value) = retry_after.parse()
    {
        response
            .headers_mut()
            .insert(http::header::RETRY_AFTER, value);
    }
    response
}

fn stream_response(
    upstream: reqwest::Response,
    provider: String,
    message_id: String,
    model: String,
    monitor: Option<MonitorHandle>,
    req_id: String,
) -> Response {
    let state = LiveState {
        upstream: Box::pin(upstream.bytes_stream()),
        translator: stream::StreamTranslator::new(provider, message_id, model),
        monitor,
        req_id,
        saw_bytes: false,
        terminal: false,
        error_sent: false,
    };
    let body_stream = futures_util::stream::unfold(state, |mut state| async move {
        state
            .next_output()
            .await
            .map(|bytes| (Ok::<Bytes, Infallible>(Bytes::from(bytes)), state))
    });
    (
        [
            (http::header::CONTENT_TYPE, "text/event-stream"),
            (http::header::CACHE_CONTROL, "no-cache"),
            (http::header::CONNECTION, "keep-alive"),
        ],
        Body::from_stream(body_stream),
    )
        .into_response()
}

type UpstreamStream = Pin<Box<dyn Stream<Item = Result<Bytes, reqwest::Error>> + Send>>;

struct LiveState {
    upstream: UpstreamStream,
    translator: stream::StreamTranslator,
    monitor: Option<MonitorHandle>,
    req_id: String,
    saw_bytes: bool,
    terminal: bool,
    error_sent: bool,
}

impl LiveState {
    async fn next_output(&mut self) -> Option<Vec<u8>> {
        if self.terminal || self.error_sent {
            return None;
        }
        loop {
            match self.upstream.next().await {
                Some(Ok(chunk)) => {
                    if !self.saw_bytes {
                        self.saw_bytes = true;
                        if let Some(monitor) = self.monitor.as_ref() {
                            monitor.generation_started(&self.req_id);
                        }
                    }
                    if let Some(monitor) = self.monitor.as_ref() {
                        monitor.stream_progress(&self.req_id, chunk.len() as u64, 1, None, None);
                    }
                    match self.translator.push(&chunk) {
                        Ok(bytes) if self.translator.is_finished() => {
                            self.terminal = true;
                            return (!bytes.is_empty()).then_some(bytes);
                        }
                        Ok(bytes) if !bytes.is_empty() => return Some(bytes),
                        Ok(_) => continue,
                        Err(_) => return Some(self.fail()),
                    }
                }
                Some(Err(_)) => return Some(self.fail()),
                None => match self.translator.finish() {
                    Ok(bytes) => {
                        self.terminal = true;
                        return (!bytes.is_empty()).then_some(bytes);
                    }
                    Err(_) => return Some(self.fail()),
                },
            }
        }
    }

    fn fail(&mut self) -> Vec<u8> {
        self.error_sent = true;
        stream::stream_error("OpenAI-compatible upstream stream is invalid")
    }
}

struct OpenAiCompatibleCli;

impl CliHandlers for OpenAiCompatibleCli {
    fn login(&self) -> Result<()> {
        bail!(
            "OpenAI-compatible providers use the API key environment variable configured in config.json"
        )
    }

    fn device(&self) -> Result<()> {
        self.login()
    }

    fn status(&self) -> Result<()> {
        Err(anyhow!(
            "OpenAI-compatible authentication is configured through environment variables"
        ))
    }

    fn logout(&self) -> Result<()> {
        bail!("Remove the API key from its environment source to log out")
    }
}

static OPENAI_COMPATIBLE_CLI: OpenAiCompatibleCli = OpenAiCompatibleCli;

#[cfg(test)]
mod normalize_tests {
    use super::*;
    use serde_json::json;

    fn normalize(value: serde_json::Value) -> serde_json::Value {
        let body: MessagesRequest = serde_json::from_value(value).unwrap();
        let normalized = normalize_anthropic_messages(body).unwrap();
        serde_json::to_value(normalized).unwrap()
    }

    #[test]
    fn system_cache_control_is_recovered_as_top_level_marker() {
        // A cache_control breakpoint on a system block cannot survive the
        // flatten to a plain string, so it is re-expressed as a top-level
        // cache_control (preserving the client's TTL) to keep the prefix cached.
        let out = normalize(json!({
            "model": "anthropic/claude-sonnet-5",
            "system": [
                {"type": "text", "text": "be concise"},
                {"type": "text", "text": "cite files", "cache_control": {"type": "ephemeral", "ttl": "1h"}}
            ],
            "messages": [{"role": "user", "content": "hi"}]
        }));
        assert_eq!(out["system"], "be concise\n\ncite files");
        assert_eq!(
            out["cache_control"],
            json!({"type": "ephemeral", "ttl": "1h"})
        );
    }

    #[test]
    fn no_system_cache_control_means_no_forced_caching() {
        // Clients that did not ask for caching must not be charged cache-write
        // costs: no top-level cache_control is synthesized.
        let out = normalize(json!({
            "model": "anthropic/claude-sonnet-5",
            "system": [{"type": "text", "text": "be concise"}],
            "messages": [{"role": "user", "content": "hi"}]
        }));
        assert_eq!(out["system"], "be concise");
        assert!(out.get("cache_control").is_none());
    }

    #[test]
    fn existing_top_level_cache_control_is_not_overridden() {
        let out = normalize(json!({
            "model": "anthropic/claude-sonnet-5",
            "cache_control": {"type": "ephemeral", "ttl": "5m"},
            "system": [{"type": "text", "text": "s", "cache_control": {"type": "ephemeral", "ttl": "1h"}}],
            "messages": [{"role": "user", "content": "hi"}]
        }));
        assert_eq!(
            out["cache_control"],
            json!({"type": "ephemeral", "ttl": "5m"})
        );
    }

    #[test]
    fn context_management_is_dropped() {
        let out = normalize(json!({
            "model": "anthropic/claude-sonnet-5",
            "context_management": {"edits": []},
            "messages": [{"role": "user", "content": "hi"}]
        }));
        assert!(out.get("context_management").is_none());
    }
}
