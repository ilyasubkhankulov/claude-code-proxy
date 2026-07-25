use std::collections::BTreeMap;
use std::sync::Arc;

use axum::{
    Json, Router,
    body::Body,
    extract::State,
    http::{HeaderMap, Request, StatusCode},
    routing::post,
};
use claude_code_proxy::{
    config::{CompatibleProtocol, OpenAiCompatibleProviderConfig},
    registry::Registry,
    server::app,
};
use serde_json::{Value, json};
use tokio::sync::oneshot;
use tower::util::ServiceExt;

#[derive(Clone)]
struct UpstreamState {
    captured: Arc<tokio::sync::Mutex<Option<(HeaderMap, Value)>>>,
}

#[derive(Clone)]
struct CloudflareUpstreamState {
    captured: Arc<tokio::sync::Mutex<Vec<(HeaderMap, Value)>>>,
}

#[tokio::test]
async fn configured_provider_translates_non_stream_request_and_response() {
    let captured = Arc::new(tokio::sync::Mutex::new(None));
    let upstream_state = UpstreamState {
        captured: captured.clone(),
    };
    let upstream = Router::new()
        .route(
            "/api/v1/chat/completions",
            post(
                |State(state): State<UpstreamState>,
                 headers: HeaderMap,
                 Json(body): Json<Value>| async move {
                    *state.captured.lock().await = Some((headers, body));
                    Json(json!({
                        "id": "chat_1",
                        "choices": [{
                            "message": {"content": "hello from upstream"},
                            "finish_reason": "stop"
                        }],
                        "usage": {"prompt_tokens": 7, "completion_tokens": 3}
                    }))
                },
            ),
        )
        .with_state(upstream_state);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let (shutdown_tx, shutdown_rx) = oneshot::channel::<()>();
    let task = tokio::spawn(async move {
        axum::serve(listener, upstream)
            .with_graceful_shutdown(async {
                let _ = shutdown_rx.await;
            })
            .await
            .unwrap();
    });

    let key_name = "CCP_TEST_OPENAI_COMPAT_KEY";
    unsafe { std::env::set_var(key_name, "secret-test-key") };
    let registry = Registry::new_with_openai_compatible(vec![OpenAiCompatibleProviderConfig {
        name: "arcee".into(),
        base_url: format!("http://{address}/api/v1"),
        api_key_env: key_name.into(),
        models: vec!["moonshotai/kimi-k2.7-code".into()],
        protocol: Default::default(),
        headers: BTreeMap::new(),
        model_rewrites: BTreeMap::new(),
    }])
    .unwrap();
    let response = app(Arc::new(registry))
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/messages")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "model": "moonshotai/kimi-k2.7-code",
                        "max_tokens": 128,
                        "stream": false,
                        "messages": [{"role":"user","content":"hello"}]
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    unsafe { std::env::remove_var(key_name) };

    assert_eq!(response.status(), StatusCode::OK);
    let body: Value = serde_json::from_slice(
        &axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap(),
    )
    .unwrap();
    assert_eq!(body["content"][0]["text"], "hello from upstream");
    assert_eq!(body["usage"]["input_tokens"], 7);

    let (headers, request) = captured.lock().await.take().unwrap();
    assert_eq!(request["model"], "moonshotai/kimi-k2.7-code");
    assert_eq!(request["stream"], false);
    assert_eq!(
        headers.get("authorization").unwrap().to_str().unwrap(),
        "Bearer secret-test-key"
    );

    let _ = shutdown_tx.send(());
    task.await.unwrap();
}

#[tokio::test]
async fn cloudflare_gateway_routes_anthropic_and_openai_models() {
    let captured = Arc::new(tokio::sync::Mutex::new(Vec::new()));
    let upstream_state = CloudflareUpstreamState {
        captured: captured.clone(),
    };
    let upstream = Router::new()
        .route(
            "/client/v4/accounts/test-account/ai/v1/chat/completions",
            post(
                |State(state): State<CloudflareUpstreamState>,
                 headers: HeaderMap,
                 Json(body): Json<Value>| async move {
                    state.captured.lock().await.push((headers, body));
                    Json(json!({
                        "id": "chat_cloudflare",
                        "choices": [{
                            "message": {"content": "hello through Cloudflare"},
                            "finish_reason": "stop"
                        }],
                        "usage": {"prompt_tokens": 5, "completion_tokens": 2}
                    }))
                },
            ),
        )
        .with_state(upstream_state);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let (shutdown_tx, shutdown_rx) = oneshot::channel::<()>();
    let task = tokio::spawn(async move {
        axum::serve(listener, upstream)
            .with_graceful_shutdown(async {
                let _ = shutdown_rx.await;
            })
            .await
            .unwrap();
    });

    let key_name = "CCP_TEST_CLOUDFLARE_AIG_TOKEN";
    unsafe { std::env::set_var(key_name, "cloudflare-test-token") };
    let registry = Arc::new(
        Registry::new_with_openai_compatible(vec![OpenAiCompatibleProviderConfig {
            name: "cloudflare".into(),
            base_url: format!("http://{address}/client/v4/accounts/test-account/ai/v1"),
            api_key_env: key_name.into(),
            models: vec!["anthropic/claude-sonnet-5".into(), "openai/gpt-5.5".into()],
            protocol: Default::default(),
            headers: BTreeMap::from([("cf-aig-gateway-id".into(), "test-gateway".into())]),
            model_rewrites: BTreeMap::new(),
        }])
        .unwrap(),
    );

    for model in ["anthropic/claude-sonnet-5", "openai/gpt-5.5"] {
        let response = app(registry.clone())
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/messages")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        json!({
                            "model": model,
                            "max_tokens": 128,
                            "stream": false,
                            "messages": [{"role":"user","content":"hello"}]
                        })
                        .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body: Value = serde_json::from_slice(
            &axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .unwrap(),
        )
        .unwrap();
        assert_eq!(body["content"][0]["text"], "hello through Cloudflare");
    }
    unsafe { std::env::remove_var(key_name) };

    let requests = captured.lock().await;
    assert_eq!(requests.len(), 2);
    for ((headers, request), model) in requests
        .iter()
        .zip(["anthropic/claude-sonnet-5", "openai/gpt-5.5"])
    {
        assert_eq!(request["model"], model);
        assert_eq!(
            headers.get("authorization").unwrap().to_str().unwrap(),
            "Bearer cloudflare-test-token"
        );
        assert_eq!(
            headers.get("cf-aig-gateway-id").unwrap().to_str().unwrap(),
            "test-gateway"
        );
    }
    drop(requests);

    let _ = shutdown_tx.send(());
    task.await.unwrap();
}

#[tokio::test]
async fn cloudflare_anthropic_protocol_preserves_native_messages() {
    let captured = Arc::new(tokio::sync::Mutex::new(None));
    let upstream_state = UpstreamState {
        captured: captured.clone(),
    };
    let upstream = Router::new()
        .route(
            "/client/v4/accounts/test-account/ai/v1/messages",
            post(
                |State(state): State<UpstreamState>,
                 headers: HeaderMap,
                 Json(body): Json<Value>| async move {
                    *state.captured.lock().await = Some((headers, body));
                    (
                        [
                            ("x-request-id", "cf-request-1"),
                            ("set-cookie", "must-not-pass=1"),
                        ],
                        Json(json!({
                            "id": "msg_cloudflare",
                            "type": "message",
                            "role": "assistant",
                            "model": "anthropic/claude-sonnet-5",
                            "content": [{"type": "text", "text": "native response"}],
                            "stop_reason": "end_turn",
                            "usage": {"input_tokens": 8, "output_tokens": 2}
                        })),
                    )
                },
            ),
        )
        .with_state(upstream_state);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let (shutdown_tx, shutdown_rx) = oneshot::channel::<()>();
    let task = tokio::spawn(async move {
        axum::serve(listener, upstream)
            .with_graceful_shutdown(async {
                let _ = shutdown_rx.await;
            })
            .await
            .unwrap();
    });

    let key_name = "CCP_TEST_CLOUDFLARE_ANTHROPIC_TOKEN";
    unsafe { std::env::set_var(key_name, "cloudflare-anthropic-token") };
    let registry = Registry::new_with_openai_compatible(vec![OpenAiCompatibleProviderConfig {
        name: "cloudflare-anthropic".into(),
        base_url: format!("http://{address}/client/v4/accounts/test-account/ai/v1"),
        api_key_env: key_name.into(),
        models: vec!["anthropic/claude-sonnet-5".into()],
        protocol: CompatibleProtocol::AnthropicMessages,
        headers: BTreeMap::from([("cf-aig-gateway-id".into(), "test-gateway".into())]),
        model_rewrites: BTreeMap::from([(
            "claude-opus-4-8".into(),
            "anthropic/claude-opus-4.8".into(),
        )]),
    }])
    .unwrap();
    let response = app(Arc::new(registry))
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/messages")
                .header("content-type", "application/json")
                .header("anthropic-version", "2023-06-01")
                .header("anthropic-beta", "prompt-caching-2024-07-31")
                .header("cookie", "must-not-forward=1")
                .body(Body::from(
                    json!({
                        "model": "anthropic/claude-sonnet-5[1m]",
                        "max_tokens": 128,
                        "system": [{"type": "text", "text": "be concise", "cache_control": {"type": "ephemeral"}}],
                        "tools": [{"name": "lookup", "description": "look up", "input_schema": {"type": "object"}}],
                        "thinking": {"type": "enabled", "budget_tokens": 64},
                        "context_management": {"edits": [{"type": "clear_tool_uses_20250919"}]},
                        "messages": [
                            {
                                "role": "system",
                                "content": [{"type": "text", "text": "late system reminder"}]
                            },
                            {
                                "role": "user",
                                "content": [{"type": "tool_result", "tool_use_id": "tool_1", "content": "done"}],
                                "context": {"future_field": true}
                            }
                        ]
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    unsafe { std::env::remove_var(key_name) };

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(response.headers()["x-request-id"], "cf-request-1");
    assert!(response.headers().get("set-cookie").is_none());
    let response_body: Value = serde_json::from_slice(
        &axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap(),
    )
    .unwrap();
    assert_eq!(response_body["content"][0]["text"], "native response");

    let (headers, request) = captured.lock().await.take().unwrap();
    assert_eq!(request["model"], "anthropic/claude-sonnet-5");
    assert_eq!(request["system"], "be concise\n\nlate system reminder");
    assert_eq!(request["messages"].as_array().unwrap().len(), 1);
    assert_eq!(request["tools"][0]["name"], "lookup");
    assert_eq!(request["thinking"]["type"], "enabled");
    assert!(request.get("context_management").is_none());
    assert_eq!(request["messages"][0]["content"][0]["type"], "tool_result");
    assert_eq!(request["messages"][0]["context"]["future_field"], true);
    assert_eq!(
        headers["authorization"],
        "Bearer cloudflare-anthropic-token"
    );
    assert_eq!(headers["cf-aig-gateway-id"], "test-gateway");
    assert_eq!(headers["anthropic-version"], "2023-06-01");
    assert_eq!(headers["anthropic-beta"], "prompt-caching-2024-07-31");
    assert!(headers.get("cookie").is_none());

    let _ = shutdown_tx.send(());
    task.await.unwrap();
}

#[tokio::test]
async fn model_rewrite_sends_upstream_id_but_keeps_client_id_recognized() {
    // The client selects a Claude-Code-recognized id (`claude-opus-4-8`, which
    // keeps auto mode + [1m] context on the client). The proxy must route it to
    // the gateway and rewrite the wire model to the gateway's id
    // (`anthropic/claude-opus-4.8`) that Cloudflare actually accepts.
    let captured = Arc::new(tokio::sync::Mutex::new(None));
    let upstream_state = UpstreamState {
        captured: captured.clone(),
    };
    let upstream = Router::new()
        .route(
            "/client/v4/accounts/test-account/ai/v1/messages",
            post(
                |State(state): State<UpstreamState>,
                 headers: HeaderMap,
                 Json(body): Json<Value>| async move {
                    *state.captured.lock().await = Some((headers, body));
                    Json(json!({
                        "id": "msg_rewrite",
                        "type": "message",
                        "role": "assistant",
                        "model": "claude-opus-4-8",
                        "content": [{"type": "text", "text": "rewritten"}],
                        "stop_reason": "end_turn",
                        "usage": {"input_tokens": 4, "output_tokens": 1}
                    }))
                },
            ),
        )
        .with_state(upstream_state);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let (shutdown_tx, shutdown_rx) = oneshot::channel::<()>();
    let task = tokio::spawn(async move {
        axum::serve(listener, upstream)
            .with_graceful_shutdown(async {
                let _ = shutdown_rx.await;
            })
            .await
            .unwrap();
    });

    let key_name = "CCP_TEST_CLOUDFLARE_REWRITE_TOKEN";
    unsafe { std::env::set_var(key_name, "cloudflare-rewrite-token") };
    let registry = Registry::new_with_openai_compatible(vec![OpenAiCompatibleProviderConfig {
        name: "cloudflare-anthropic".into(),
        base_url: format!("http://{address}/client/v4/accounts/test-account/ai/v1"),
        api_key_env: key_name.into(),
        models: vec!["anthropic/claude-sonnet-5".into()],
        protocol: CompatibleProtocol::AnthropicMessages,
        headers: BTreeMap::from([("cf-aig-gateway-id".into(), "test-gateway".into())]),
        model_rewrites: BTreeMap::from([(
            "claude-opus-4-8".into(),
            "anthropic/claude-opus-4.8".into(),
        )]),
    }])
    .unwrap();
    let app = app(Arc::new(registry));

    // Discovery must advertise the recognized (client-facing) id, not the
    // gateway id, so Claude Code sees an auto-mode-eligible model.
    let models_response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/v1/models?limit=1000")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    let models_body: Value = serde_json::from_slice(
        &axum::body::to_bytes(models_response.into_body(), usize::MAX)
            .await
            .unwrap(),
    )
    .unwrap();
    let ids: Vec<&str> = models_body["data"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|entry| entry["id"].as_str())
        .collect();
    assert!(
        ids.contains(&"claude-opus-4-8"),
        "discovery advertises {ids:?}"
    );

    // Client selects the recognized id (with the [1m] suffix, as Claude Code
    // does); the proxy rewrites the wire model to the gateway id.
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/messages")
                .header("content-type", "application/json")
                .header("anthropic-version", "2023-06-01")
                .body(Body::from(
                    json!({
                        "model": "claude-opus-4-8[1m]",
                        "max_tokens": 16,
                        "messages": [{"role":"user","content":"hi"}]
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    unsafe { std::env::remove_var(key_name) };

    assert_eq!(response.status(), StatusCode::OK);
    let (_headers, request) = captured.lock().await.take().unwrap();
    assert_eq!(request["model"], "anthropic/claude-opus-4.8");

    let _ = shutdown_tx.send(());
    task.await.unwrap();
}

#[tokio::test]
async fn configured_provider_reports_missing_api_key() {
    let key_name = "CCP_TEST_MISSING_OPENAI_COMPAT_KEY";
    unsafe { std::env::remove_var(key_name) };
    let registry = Registry::new_with_openai_compatible(vec![OpenAiCompatibleProviderConfig {
        name: "custom".into(),
        base_url: "https://example.invalid/v1".into(),
        api_key_env: key_name.into(),
        models: vec!["org/model".into()],
        protocol: Default::default(),
        headers: BTreeMap::new(),
        model_rewrites: BTreeMap::new(),
    }])
    .unwrap();
    let response = app(Arc::new(registry))
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/messages")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "model": "org/model",
                        "messages": [{"role":"user","content":"hello"}]
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    let body: Value = serde_json::from_slice(
        &axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap(),
    )
    .unwrap();
    assert_eq!(body["error"]["type"], "authentication_error");
    assert!(
        body["error"]["message"]
            .as_str()
            .unwrap()
            .contains(key_name)
    );
}
