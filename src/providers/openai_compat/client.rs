use std::time::Duration;

use reqwest::StatusCode;

use super::request::ChatRequest;
use crate::anthropic::schema::MessagesRequest;

pub struct OpenAiClient {
    client: reqwest::Client,
    base_url: String,
    headers: reqwest::header::HeaderMap,
}

#[derive(Debug)]
pub struct ClientError {
    pub status: Option<StatusCode>,
    pub message: String,
    pub retry_after: Option<String>,
}

impl OpenAiClient {
    pub fn new(base_url: String, headers: reqwest::header::HeaderMap) -> Self {
        Self {
            client: reqwest::Client::builder()
                .connect_timeout(Duration::from_secs(15))
                .timeout(Duration::from_secs(120))
                .redirect(reqwest::redirect::Policy::none())
                .build()
                .expect("failed to construct OpenAI-compatible HTTP client"),
            base_url: base_url.trim_end_matches('/').to_string(),
            headers,
        }
    }

    pub async fn send_anthropic(
        &self,
        api_key: &str,
        request: &MessagesRequest,
        anthropic_headers: reqwest::header::HeaderMap,
    ) -> Result<reqwest::Response, ClientError> {
        let url = format!("{}/messages", self.base_url);
        self.client
            .post(url)
            .headers(self.headers.clone())
            .headers(anthropic_headers)
            .bearer_auth(api_key)
            .header(
                reqwest::header::ACCEPT,
                if request.stream {
                    "text/event-stream"
                } else {
                    "application/json"
                },
            )
            .header(
                reqwest::header::USER_AGENT,
                concat!("claude-code-proxy/", env!("CARGO_PKG_VERSION")),
            )
            .json(request)
            .send()
            .await
            .map_err(|error| ClientError {
                status: None,
                message: format!("Anthropic-compatible upstream request failed: {error}"),
                retry_after: None,
            })
    }

    pub async fn send(
        &self,
        api_key: &str,
        request: &ChatRequest,
    ) -> Result<reqwest::Response, ClientError> {
        let url = format!("{}/chat/completions", self.base_url);
        let response = self
            .client
            .post(url)
            .headers(self.headers.clone())
            .bearer_auth(api_key)
            .header(
                reqwest::header::ACCEPT,
                if request.stream {
                    "text/event-stream"
                } else {
                    "application/json"
                },
            )
            .header(
                reqwest::header::USER_AGENT,
                concat!("claude-code-proxy/", env!("CARGO_PKG_VERSION")),
            )
            .json(request)
            .send()
            .await
            .map_err(|error| ClientError {
                status: None,
                message: format!("OpenAI-compatible upstream request failed: {error}"),
                retry_after: None,
            })?;

        if response.status().is_success() {
            return Ok(response);
        }

        let status = response.status();
        let retry_after = response
            .headers()
            .get(reqwest::header::RETRY_AFTER)
            .and_then(|value| value.to_str().ok())
            .map(str::to_string);
        let body = response.text().await.unwrap_or_default();
        let detail = serde_json::from_str::<serde_json::Value>(&body)
            .ok()
            .and_then(|value| {
                value
                    .pointer("/error/message")
                    .or_else(|| value.get("message"))
                    .and_then(|value| value.as_str())
                    .map(str::to_string)
            })
            .unwrap_or_else(|| {
                let mut bounded = body.chars().take(500).collect::<String>();
                if bounded.is_empty() {
                    bounded = status
                        .canonical_reason()
                        .unwrap_or("upstream error")
                        .to_string();
                }
                bounded
            });
        Err(ClientError {
            status: Some(status),
            message: detail,
            retry_after,
        })
    }
}
