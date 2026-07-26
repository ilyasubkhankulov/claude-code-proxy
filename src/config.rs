use anyhow::{Context, Result, anyhow, bail};
use http::{HeaderMap, HeaderName, HeaderValue};
use serde::Deserialize;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};

use crate::paths;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AliasProvider {
    Codex,
    Kimi,
}

impl AliasProvider {
    pub fn as_str(&self) -> &str {
        match self {
            AliasProvider::Codex => "codex",
            AliasProvider::Kimi => "kimi",
        }
    }
}

#[derive(Debug, Clone)]
pub struct LoadedConfig {
    pub bind_address: String,
    pub port: u16,
    pub alias_provider: AliasProvider,
    pub log_verbose: bool,
    pub log_stderr: bool,
    pub config_dir: PathBuf,
}

#[derive(Deserialize)]
struct FileConfig {
    #[serde(rename = "bindAddress")]
    pub bind_address: Option<String>,
    pub port: Option<u16>,
    #[serde(rename = "aliasProvider")]
    pub alias_provider: Option<String>,
    pub log: Option<FileLog>,
    pub kimi: Option<KimiConfig>,
    pub codex: Option<CodexConfig>,
    pub cursor: Option<CursorConfig>,
    pub grok: Option<GrokConfig>,
    #[serde(rename = "openaiCompatible")]
    pub openai_compatible: Option<BTreeMap<String, OpenAiCompatibleFileConfig>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Default)]
pub enum CompatibleProtocol {
    #[default]
    #[serde(rename = "openai-chat")]
    OpenAiChat,
    #[serde(rename = "anthropic-messages")]
    AnthropicMessages,
}

impl CompatibleProtocol {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::OpenAiChat => "openai-chat",
            Self::AnthropicMessages => "anthropic-messages",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OpenAiCompatibleProviderConfig {
    pub name: String,
    pub base_url: String,
    pub api_key_env: String,
    pub models: Vec<String>,
    pub headers: BTreeMap<String, String>,
    pub protocol: CompatibleProtocol,
    /// Maps an incoming (client-facing) model ID to the model ID actually sent
    /// upstream. A rewrite key is implicitly routable to this provider, so the
    /// client can keep using a Claude-Code-recognized ID (e.g. `claude-opus-4-8`)
    /// while the proxy forwards the gateway's ID (e.g. `anthropic/claude-opus-4.8`).
    pub model_rewrites: BTreeMap<String, String>,
    /// Optional cache-TTL override for the `anthropic-messages` protocol. When
    /// set to `"5m"` or `"1h"`, every ephemeral `cache_control` breakpoint the
    /// proxy forwards to the gateway is rewritten to this TTL. Claude Code only
    /// ever emits 5-minute breakpoints; forcing `"1h"` keeps Anthropic's prompt
    /// cache alive across idle gaps longer than five minutes, avoiding a full
    /// uncached prefix re-read on the next turn.
    pub cache_ttl: Option<String>,
}

#[derive(Deserialize, Clone)]
struct OpenAiCompatibleFileConfig {
    #[serde(rename = "baseUrl")]
    base_url: String,
    #[serde(rename = "apiKeyEnv")]
    api_key_env: String,
    models: Vec<String>,
    #[serde(default)]
    headers: BTreeMap<String, String>,
    #[serde(default)]
    protocol: CompatibleProtocol,
    #[serde(rename = "modelRewrites", default)]
    model_rewrites: BTreeMap<String, String>,
    #[serde(rename = "cacheTtl", default)]
    cache_ttl: Option<String>,
}

#[derive(Deserialize, Clone)]
struct CodexConfig {
    #[serde(rename = "baseUrl")]
    pub base_url: Option<String>,
    #[serde(rename = "originator")]
    pub originator: Option<String>,
    #[serde(rename = "userAgent")]
    pub user_agent: Option<String>,
    #[serde(rename = "previousResponseId")]
    pub previous_response_id: Option<bool>,
    #[serde(rename = "serverCompaction")]
    pub server_compaction: Option<bool>,
    #[serde(rename = "responsesApi")]
    pub responses_api: Option<bool>,
    #[serde(rename = "serviceTier")]
    pub service_tier: Option<String>,
    #[serde(rename = "reasoningSummary")]
    pub reasoning_summary: Option<String>,
    #[serde(rename = "effort")]
    pub effort: Option<String>,
    #[serde(rename = "model")]
    pub model: Option<String>,
    pub transport: Option<String>,
}

#[derive(Deserialize, Clone)]
struct CursorConfig {
    #[serde(rename = "baseUrl")]
    pub base_url: Option<String>,
    #[serde(rename = "clientVersion")]
    pub client_version: Option<String>,
    #[serde(rename = "agentBundle")]
    pub agent_bundle: Option<String>,
}

#[derive(Deserialize, Clone)]
struct KimiConfig {
    #[serde(rename = "userAgent")]
    pub user_agent: Option<String>,
    #[serde(rename = "oauthHost")]
    pub oauth_host: Option<String>,
    #[serde(rename = "baseUrl")]
    pub base_url: Option<String>,
}

#[derive(Deserialize, Clone)]
struct GrokConfig {
    #[serde(rename = "baseUrl")]
    pub base_url: Option<String>,
    #[serde(rename = "clientVersion")]
    pub client_version: Option<String>,
}

#[derive(Deserialize)]
struct FileLog {
    pub verbose: Option<bool>,
    pub stderr: Option<bool>,
}

fn parse_alias(raw: &str) -> Option<AliasProvider> {
    match raw {
        "codex" => Some(AliasProvider::Codex),
        "kimi" => Some(AliasProvider::Kimi),
        _ => None,
    }
}

fn read_file_config(config_dir: &Path) -> Option<FileConfig> {
    let path = config_dir.join("config.json");
    let raw = fs::read_to_string(path).ok()?;
    serde_json::from_str(&raw).ok()
}

pub fn openai_compatible_providers() -> Result<Vec<OpenAiCompatibleProviderConfig>> {
    let path = paths::config_dir().join("config.json");
    let raw = match fs::read_to_string(&path) {
        Ok(raw) => raw,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(err) => return Err(err).with_context(|| format!("read {}", path.display())),
    };
    let file: FileConfig =
        serde_json::from_str(&raw).with_context(|| format!("parse {}", path.display()))?;
    let providers = file.openai_compatible.unwrap_or_default();
    let mut out = Vec::with_capacity(providers.len());

    for (name, provider) in providers {
        validate_openai_compatible_provider(&name, &provider)?;
        out.push(OpenAiCompatibleProviderConfig {
            name,
            base_url: provider.base_url.trim_end_matches('/').to_string(),
            api_key_env: provider.api_key_env,
            models: provider.models,
            headers: provider.headers,
            protocol: provider.protocol,
            model_rewrites: provider.model_rewrites,
            cache_ttl: provider.cache_ttl,
        });
    }

    Ok(out)
}

fn validate_openai_compatible_provider(
    name: &str,
    provider: &OpenAiCompatibleFileConfig,
) -> Result<()> {
    if name.is_empty()
        || !name
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_'))
    {
        bail!(
            "openaiCompatible provider name {name:?} must contain only letters, numbers, '-' or '_'"
        );
    }
    if matches!(name, "codex" | "kimi" | "cursor" | "grok") {
        bail!("openaiCompatible provider name {name:?} is reserved");
    }

    let url = reqwest::Url::parse(&provider.base_url)
        .map_err(|err| anyhow!("openaiCompatible.{name}.baseUrl is invalid: {err}"))?;
    if !matches!(url.scheme(), "http" | "https") || url.host_str().is_none() {
        bail!("openaiCompatible.{name}.baseUrl must be an absolute HTTP(S) URL");
    }
    if provider.api_key_env.trim().is_empty() {
        bail!("openaiCompatible.{name}.apiKeyEnv must not be empty");
    }
    parse_openai_compatible_headers(name, &provider.headers)?;
    if provider.models.is_empty() {
        bail!("openaiCompatible.{name}.models must not be empty");
    }

    let mut seen = HashSet::new();
    for model in &provider.models {
        if model.trim().is_empty() {
            bail!("openaiCompatible.{name}.models must not contain an empty model ID");
        }
        if !seen.insert(model) {
            bail!("openaiCompatible.{name}.models contains duplicate model {model:?}");
        }
    }

    for (from, to) in &provider.model_rewrites {
        if from.trim().is_empty() {
            bail!(
                "openaiCompatible.{name}.modelRewrites must not contain an empty source model ID"
            );
        }
        if to.trim().is_empty() {
            bail!("openaiCompatible.{name}.modelRewrites has an empty target for model {from:?}");
        }
        if seen.contains(from) {
            bail!(
                "openaiCompatible.{name}.modelRewrites key {from:?} also appears in models; list it in only one place"
            );
        }
    }

    if let Some(ttl) = &provider.cache_ttl {
        if !matches!(ttl.as_str(), "5m" | "1h") {
            bail!("openaiCompatible.{name}.cacheTtl must be \"5m\" or \"1h\"");
        }
        if provider.protocol != CompatibleProtocol::AnthropicMessages {
            bail!(
                "openaiCompatible.{name}.cacheTtl is only supported with protocol \"anthropic-messages\""
            );
        }
    }
    Ok(())
}

pub(crate) fn parse_openai_compatible_headers(
    provider_name: &str,
    headers: &BTreeMap<String, String>,
) -> Result<HeaderMap> {
    let mut parsed = HeaderMap::new();
    let mut seen = HashSet::new();

    for (name, value) in headers {
        let header_name = HeaderName::from_bytes(name.as_bytes()).map_err(|_| {
            anyhow!(
                "openaiCompatible.{provider_name}.headers contains invalid header name {name:?}"
            )
        })?;
        let normalized = header_name.as_str();
        if !seen.insert(normalized.to_string()) {
            bail!(
                "openaiCompatible.{provider_name}.headers contains duplicate header name {name:?}"
            );
        }
        if is_reserved_openai_compatible_header(normalized) {
            bail!(
                "openaiCompatible.{provider_name}.headers cannot override reserved header {name:?}"
            );
        }
        let header_value = value.parse::<HeaderValue>().map_err(|_| {
            anyhow!(
                "openaiCompatible.{provider_name}.headers contains invalid value for header {name:?}"
            )
        })?;
        parsed.insert(header_name, header_value);
    }

    Ok(parsed)
}

fn is_reserved_openai_compatible_header(name: &str) -> bool {
    matches!(
        name,
        "authorization"
            | "accept"
            | "content-type"
            | "content-length"
            | "host"
            | "user-agent"
            | "connection"
            | "keep-alive"
            | "proxy-authenticate"
            | "proxy-authorization"
            | "proxy-connection"
            | "te"
            | "trailer"
            | "transfer-encoding"
            | "upgrade"
    )
}

pub fn load_config() -> LoadedConfig {
    let config_dir = paths::config_dir();
    let file = read_file_config(&config_dir);
    let env: HashMap<_, _> = std::env::vars().collect();

    let mut out = LoadedConfig {
        bind_address: "127.0.0.1".to_string(),
        port: 18765,
        alias_provider: AliasProvider::Codex,
        log_verbose: false,
        log_stderr: false,
        config_dir: config_dir.clone(),
    };

    if let Some(raw) = env.get("CCP_BIND_ADDRESS") {
        out.bind_address = raw.clone();
    } else if let Some(bind_address) = file.as_ref().and_then(|f| f.bind_address.clone()) {
        out.bind_address = bind_address;
    }

    if let Some(raw) = env.get("CCP_ALIAS_PROVIDER") {
        if let Some(alias) = parse_alias(raw) {
            out.alias_provider = alias;
        }
    } else if let Some(alias_provider) = file
        .as_ref()
        .and_then(|f| f.alias_provider.as_deref())
        .and_then(parse_alias)
    {
        out.alias_provider = alias_provider;
    }

    if let Some(raw) = env.get("PORT") {
        if let Ok(port) = raw.parse::<u16>() {
            out.port = port;
        }
    } else if let Some(port) = file.as_ref().and_then(|f| f.port) {
        out.port = port;
    }

    if env.contains_key("CCP_LOG_VERBOSE") {
        out.log_verbose = true;
    } else if let Some(value) = file
        .as_ref()
        .and_then(|f| f.log.as_ref().and_then(|v| v.verbose))
    {
        out.log_verbose = value;
    }

    if env.contains_key("CCP_LOG_STDERR") {
        out.log_stderr = true;
    } else if let Some(value) = file
        .as_ref()
        .and_then(|f| f.log.as_ref().and_then(|v| v.stderr))
    {
        out.log_stderr = value;
    }

    out
}

pub fn config_path() -> PathBuf {
    paths::config_dir().join("config.json")
}

pub fn port() -> u16 {
    load_config().port
}

pub fn bind_address() -> String {
    load_config().bind_address
}

pub fn alias_provider() -> AliasProvider {
    load_config().alias_provider
}

pub fn log_verbose() -> bool {
    load_config().log_verbose
}

pub fn log_stderr() -> bool {
    load_config().log_stderr
}

pub fn config_override_summary_lines(cfg: &LoadedConfig) -> Vec<String> {
    let file = read_file_config(&cfg.config_dir);
    let env: HashMap<_, _> = std::env::vars().collect();
    let mut out = Vec::new();
    if env.contains_key("CCP_BIND_ADDRESS") {
        out.push("bindAddress (env)".to_string());
    }
    if env.contains_key("PORT") {
        out.push("port (env)".to_string());
    }
    if env.contains_key("CCP_ALIAS_PROVIDER") {
        out.push("aliasProvider (env)".to_string());
    }
    if env.contains_key("CCP_LOG_VERBOSE") {
        out.push("log.verbose (env)".to_string());
    }
    if env.contains_key("CCP_LOG_STDERR") {
        out.push("log.stderr (env)".to_string());
    }
    if env.contains_key("CCP_CODEX_RESPONSES_API") {
        out.push("codex.responsesApi (env)".to_string());
    }
    if env.contains_key("CCP_KIMI_OAUTH_HOST") {
        out.push("kimi.oauthHost (env)".to_string());
    }
    if env.contains_key("CCP_KIMI_BASE_URL") {
        out.push("kimi.baseUrl (env)".to_string());
    }
    if env.contains_key("CCP_CURSOR_BASE_URL") {
        out.push("cursor.baseUrl (env)".to_string());
    }
    if env.contains_key("CCP_CURSOR_CLIENT_VERSION") {
        out.push("cursor.clientVersion (env)".to_string());
    }
    if env.contains_key("CCP_KIMI_USER_AGENT") {
        out.push("kimi.userAgent (env)".to_string());
    }
    if env.contains_key("CCP_GROK_BASE_URL") {
        out.push("grok.baseUrl (env)".to_string());
    }
    if env.contains_key("CCP_GROK_CLIENT_VERSION") {
        out.push("grok.clientVersion (env)".to_string());
    }
    if env
        .get("CCP_CODEX_REASONING_SUMMARY")
        .is_some_and(|raw| !raw.is_empty())
    {
        out.push("CCP_CODEX_REASONING_SUMMARY (env)".to_string());
    }
    if env.contains_key("CCP_CODEX_SERVER_COMPACTION") {
        out.push("CCP_CODEX_SERVER_COMPACTION (env)".to_string());
    }
    if let Some(file_cfg) = file {
        if let Some(bind_address) = file_cfg.bind_address {
            out.push(format!("bindAddress: {bind_address}"));
        }
        if let Some(p) = file_cfg.port {
            out.push(format!("port: {p}"));
        }
        if let Some(alias) = file_cfg.alias_provider {
            out.push(format!("aliasProvider: {alias}"));
        }
        if let Some(log) = file_cfg.log {
            if let Some(v) = log.verbose {
                out.push(format!("log.verbose: {v}"));
            }
            if let Some(v) = log.stderr {
                out.push(format!("log.stderr: {v}"));
            }
        }
        if let Some(codex) = file_cfg.codex {
            if codex
                .reasoning_summary
                .is_some_and(|value| !value.is_empty())
            {
                out.push("codex.reasoningSummary (config)".to_string());
            }
            if let Some(enabled) = codex.server_compaction {
                out.push(format!("codex.serverCompaction: {enabled}"));
            }
            if codex.responses_api == Some(true) {
                out.push("codex.responsesApi: true".to_string());
            }
        }
        if let Some(providers) = file_cfg.openai_compatible {
            for (name, provider) in providers {
                let key_state = if env.contains_key(&provider.api_key_env) {
                    "set"
                } else {
                    "unset"
                };
                out.push(format!(
                    "openaiCompatible.{name}: {} protocol, {} models, key env {} ({key_state}), {} custom headers, {} model rewrites",
                    provider.protocol.as_str(),
                    provider.models.len(),
                    provider.api_key_env,
                    provider.headers.len(),
                    provider.model_rewrites.len()
                ));
            }
        }
    }
    out
}

pub fn grok_base_url() -> String {
    let env: HashMap<_, _> = std::env::vars().collect();
    if let Some(raw) = env.get("CCP_GROK_BASE_URL") {
        return raw.clone();
    }
    if let Some(grok) = read_file_config(&paths::config_dir()).and_then(|f| f.grok)
        && let Some(url) = grok.base_url
    {
        return url;
    }
    "https://cli-chat-proxy.grok.com/v1".to_string()
}

pub fn grok_client_version() -> String {
    let env: HashMap<_, _> = std::env::vars().collect();
    if let Some(raw) = env.get("CCP_GROK_CLIENT_VERSION") {
        return raw.clone();
    }
    if let Some(grok) = read_file_config(&paths::config_dir()).and_then(|f| f.grok)
        && let Some(version) = grok.client_version
    {
        return version;
    }
    "0.2.93".to_string()
}

pub fn is_verbose() -> bool {
    log_verbose()
}

pub fn kimi_oauth_host() -> String {
    let env: HashMap<_, _> = std::env::vars().collect();
    if let Some(raw) = env.get("CCP_KIMI_OAUTH_HOST") {
        return raw.clone();
    }
    let config_dir = paths::config_dir();
    if let Some(file) = read_file_config(&config_dir)
        && let Some(kimi) = file.kimi
        && let Some(host) = kimi.oauth_host
    {
        return host;
    }
    "https://auth.kimi.com".to_string()
}

pub fn kimi_base_url() -> String {
    let env: HashMap<_, _> = std::env::vars().collect();
    if let Some(raw) = env.get("CCP_KIMI_BASE_URL") {
        return raw.clone();
    }
    let config_dir = paths::config_dir();
    if let Some(file) = read_file_config(&config_dir)
        && let Some(kimi) = file.kimi
        && let Some(url) = kimi.base_url
    {
        return url;
    }
    "https://api.kimi.com/coding/v1".to_string()
}

pub fn kimi_user_agent(default: &str) -> String {
    let env: HashMap<_, _> = std::env::vars().collect();
    if let Some(raw) = env.get("CCP_KIMI_USER_AGENT") {
        return raw.clone();
    }
    if let Some(raw) = env.get("CCP_USER_AGENT") {
        return raw.clone();
    }
    let config_dir = paths::config_dir();
    if let Some(file) = read_file_config(&config_dir)
        && let Some(kimi) = file.kimi
        && let Some(ua) = kimi.user_agent
    {
        return ua;
    }
    default.to_string()
}

// ---------------------------------------------------------------------------
// Codex config
// ---------------------------------------------------------------------------

pub fn codex_base_url(default: &str) -> String {
    let env: HashMap<_, _> = std::env::vars().collect();
    if let Some(raw) = env.get("CCP_CODEX_BASE_URL") {
        return raw.clone();
    }
    if let Some(raw) = env.get("CLAUDE_CODE_PROXY_CODEX_BASE_URL") {
        return raw.clone();
    }
    let config_dir = paths::config_dir();
    if let Some(file) = read_file_config(&config_dir)
        && let Some(codex) = file.codex
        && let Some(url) = codex.base_url
    {
        return url;
    }
    default.to_string()
}

pub fn codex_originator(default: &str) -> String {
    let env: HashMap<_, _> = std::env::vars().collect();
    if let Some(raw) = env.get("CCP_CODEX_ORIGINATOR") {
        return raw.clone();
    }
    let config_dir = paths::config_dir();
    if let Some(file) = read_file_config(&config_dir)
        && let Some(codex) = file.codex
        && let Some(val) = codex.originator
    {
        return val;
    }
    default.to_string()
}

pub fn codex_user_agent(default: &str) -> String {
    let env: HashMap<_, _> = std::env::vars().collect();
    if let Some(raw) = env.get("CCP_CODEX_USER_AGENT") {
        return raw.clone();
    }
    if let Some(raw) = env.get("CCP_USER_AGENT") {
        return raw.clone();
    }
    let config_dir = paths::config_dir();
    if let Some(file) = read_file_config(&config_dir)
        && let Some(codex) = file.codex
        && let Some(ua) = codex.user_agent
    {
        return ua;
    }
    default.to_string()
}

pub fn codex_previous_response_id() -> bool {
    let env: HashMap<_, _> = std::env::vars().collect();
    if let Some(raw) = env.get("CCP_CODEX_PREVIOUS_RESPONSE_ID") {
        return matches!(raw.to_ascii_lowercase().as_str(), "1" | "true" | "yes");
    }
    let config_dir = paths::config_dir();
    if let Some(file) = read_file_config(&config_dir)
        && let Some(codex) = file.codex
        && let Some(val) = codex.previous_response_id
    {
        return val;
    }
    false
}

pub fn codex_server_compaction() -> bool {
    let env: HashMap<_, _> = std::env::vars().collect();
    if let Some(raw) = env.get("CCP_CODEX_SERVER_COMPACTION") {
        match raw.trim().to_ascii_lowercase().as_str() {
            "1" | "true" | "yes" | "on" => return true,
            "0" | "false" | "no" | "off" => return false,
            _ => {}
        }
    }
    let config_dir = paths::config_dir();
    if let Some(file) = read_file_config(&config_dir)
        && let Some(codex) = file.codex
        && let Some(enabled) = codex.server_compaction
    {
        return enabled;
    }
    false
}

pub fn codex_responses_api() -> bool {
    let env: HashMap<_, _> = std::env::vars().collect();
    if let Some(raw) = env.get("CCP_CODEX_RESPONSES_API") {
        return matches!(raw.to_ascii_lowercase().as_str(), "1" | "true" | "yes");
    }
    let config_dir = paths::config_dir();
    if let Some(file) = read_file_config(&config_dir)
        && let Some(codex) = file.codex
        && let Some(enabled) = codex.responses_api
    {
        return enabled;
    }
    false
}

pub fn codex_service_tier() -> Option<String> {
    let env: HashMap<_, _> = std::env::vars().collect();
    if let Some(raw) = env.get("CCP_CODEX_SERVICE_TIER") {
        return Some(raw.clone());
    }
    let config_dir = paths::config_dir();
    if let Some(file) = read_file_config(&config_dir)
        && let Some(codex) = file.codex
    {
        return codex.service_tier;
    }
    None
}

pub fn codex_effort() -> Option<String> {
    let env: HashMap<_, _> = std::env::vars().collect();
    if let Some(raw) = env.get("CCP_CODEX_EFFORT") {
        return Some(raw.clone());
    }
    let config_dir = paths::config_dir();
    if let Some(file) = read_file_config(&config_dir)
        && let Some(codex) = file.codex
    {
        return codex.effort;
    }
    None
}

pub fn codex_reasoning_summary() -> Option<String> {
    let env: HashMap<_, _> = std::env::vars().collect();
    if let Some(raw) = env
        .get("CCP_CODEX_REASONING_SUMMARY")
        .filter(|raw| !raw.is_empty())
    {
        return Some(raw.clone());
    }
    let config_dir = paths::config_dir();
    if let Some(file) = read_file_config(&config_dir)
        && let Some(codex) = file.codex
        && let Some(summary) = codex.reasoning_summary.filter(|raw| !raw.is_empty())
    {
        return Some(summary);
    }
    None
}

pub fn codex_model() -> Option<String> {
    let env: HashMap<_, _> = std::env::vars().collect();
    if let Some(raw) = env.get("CCP_CODEX_MODEL") {
        return Some(raw.clone());
    }
    let config_dir = paths::config_dir();
    if let Some(file) = read_file_config(&config_dir)
        && let Some(codex) = file.codex
    {
        return codex.model;
    }
    None
}

// ---------------------------------------------------------------------------
// Codex transport config
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CodexTransport {
    Http,
    WebSocket,
    Auto,
}

impl CodexTransport {
    pub fn as_str(self) -> &'static str {
        match self {
            CodexTransport::Http => "http",
            CodexTransport::WebSocket => "websocket",
            CodexTransport::Auto => "auto",
        }
    }
}

fn parse_codex_transport(raw: &str) -> Option<CodexTransport> {
    match raw {
        "http" => Some(CodexTransport::Http),
        "websocket" => Some(CodexTransport::WebSocket),
        "auto" => Some(CodexTransport::Auto),
        _ => None,
    }
}

pub fn codex_transport() -> CodexTransport {
    let env: HashMap<_, _> = std::env::vars().collect();
    if let Some(raw) = env.get("CCP_CODEX_TRANSPORT")
        && let Some(transport) = parse_codex_transport(raw)
    {
        return transport;
    }
    let config_dir = paths::config_dir();
    if let Some(file) = read_file_config(&config_dir)
        && let Some(codex) = file.codex
        && let Some(transport) = codex.transport.as_deref().and_then(parse_codex_transport)
    {
        return transport;
    }
    CodexTransport::WebSocket
}

// ---------------------------------------------------------------------------
// Cursor config
// ---------------------------------------------------------------------------

pub fn cursor_base_url() -> String {
    let env: HashMap<_, _> = std::env::vars().collect();
    if let Some(raw) = env.get("CCP_CURSOR_BASE_URL") {
        return raw.clone();
    }
    let config_dir = paths::config_dir();
    if let Some(file) = read_file_config(&config_dir)
        && let Some(cursor) = file.cursor
        && let Some(url) = cursor.base_url
    {
        return url;
    }
    "https://api2.cursor.sh".to_string()
}

pub fn cursor_client_version() -> String {
    let env: HashMap<_, _> = std::env::vars().collect();
    if let Some(raw) = env.get("CCP_CURSOR_CLIENT_VERSION") {
        return raw.clone();
    }
    let config_dir = paths::config_dir();
    if let Some(file) = read_file_config(&config_dir)
        && let Some(cursor) = file.cursor
        && let Some(version) = cursor.client_version
    {
        return version;
    }
    "0.48.5".to_string()
}

pub fn cursor_agent_bundle() -> Option<String> {
    let env: HashMap<_, _> = std::env::vars().collect();
    if let Some(raw) = env.get("CCP_CURSOR_AGENT_BUNDLE") {
        return Some(raw.clone());
    }
    let config_dir = paths::config_dir();
    if let Some(file) = read_file_config(&config_dir)
        && let Some(cursor) = file.cursor
        && let Some(bundle) = cursor.agent_bundle
    {
        return Some(bundle);
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use once_cell::sync::Lazy;
    use std::sync::Mutex;

    static ENV_LOCK: Lazy<Mutex<()>> = Lazy::new(|| Mutex::new(()));

    fn clear_env() {
        unsafe {
            std::env::remove_var("CCP_BIND_ADDRESS");
            std::env::remove_var("CCP_CODEX_TRANSPORT");
            std::env::remove_var("CCP_CONFIG_DIR");
            std::env::remove_var("CCP_LOG_VERBOSE");
            std::env::remove_var("CCP_LOG_STDERR");
            std::env::remove_var("CCP_CODEX_REASONING_SUMMARY");
            std::env::remove_var("CCP_CODEX_SERVER_COMPACTION");
            std::env::remove_var("CCP_CODEX_RESPONSES_API");
        }
    }

    #[test]
    fn bind_address_defaults_to_loopback() {
        let _guard = ENV_LOCK.lock().unwrap();
        clear_env();
        let config = tempfile::TempDir::new().unwrap();
        let _config_env = EnvGuard::set("CCP_CONFIG_DIR", config.path());

        assert_eq!(load_config().bind_address, "127.0.0.1");
    }

    #[test]
    fn bind_address_reads_config_and_env_takes_precedence() {
        let _guard = ENV_LOCK.lock().unwrap();
        clear_env();
        let config = tempfile::TempDir::new().unwrap();
        std::fs::write(
            config.path().join("config.json"),
            r#"{"bindAddress":"192.0.2.10"}"#,
        )
        .unwrap();
        let _config_env = EnvGuard::set("CCP_CONFIG_DIR", config.path());

        assert_eq!(load_config().bind_address, "192.0.2.10");
        let _bind_env = EnvGuard::set("CCP_BIND_ADDRESS", "0.0.0.0");
        assert_eq!(load_config().bind_address, "0.0.0.0");
    }

    struct EnvGuard {
        key: &'static str,
        previous: Option<std::ffi::OsString>,
    }

    impl EnvGuard {
        fn set(key: &'static str, value: impl AsRef<std::ffi::OsStr>) -> Self {
            let previous = std::env::var_os(key);
            unsafe {
                std::env::set_var(key, value);
            }
            Self { key, previous }
        }
    }

    impl Drop for EnvGuard {
        fn drop(&mut self) {
            unsafe {
                match self.previous.take() {
                    Some(value) => std::env::set_var(self.key, value),
                    None => std::env::remove_var(self.key),
                }
            }
        }
    }

    #[test]
    fn openai_compatible_config_loads_without_reading_secret() {
        let _guard = ENV_LOCK.lock().unwrap();
        clear_env();
        let config = tempfile::TempDir::new().unwrap();
        std::fs::write(
            config.path().join("config.json"),
            r#"{"openaiCompatible":{"arcee":{"baseUrl":"https://api.arcee.ai/api/v1/","apiKeyEnv":"ARCEE_API_KEY","models":["org/model"]}}}"#,
        )
        .unwrap();
        let _config_env = EnvGuard::set("CCP_CONFIG_DIR", config.path());

        let providers = openai_compatible_providers().unwrap();
        assert_eq!(providers.len(), 1);
        assert_eq!(providers[0].name, "arcee");
        assert_eq!(providers[0].base_url, "https://api.arcee.ai/api/v1");
        assert_eq!(providers[0].api_key_env, "ARCEE_API_KEY");
        assert_eq!(providers[0].models, vec!["org/model"]);
        assert_eq!(providers[0].protocol, CompatibleProtocol::OpenAiChat);
        assert!(providers[0].headers.is_empty());
    }

    #[test]
    fn openai_compatible_config_loads_protocol_and_rejects_unknown_value() {
        let _guard = ENV_LOCK.lock().unwrap();
        clear_env();
        let config = tempfile::TempDir::new().unwrap();
        let _config_env = EnvGuard::set("CCP_CONFIG_DIR", config.path());
        std::fs::write(
            config.path().join("config.json"),
            r#"{"openaiCompatible":{"cloudflare":{"baseUrl":"https://api.cloudflare.com/client/v4/accounts/test/ai/v1","apiKeyEnv":"CF_AIG_TOKEN","models":["anthropic/claude-sonnet-5"],"protocol":"anthropic-messages"}}}"#,
        )
        .unwrap();

        let providers = openai_compatible_providers().unwrap();
        assert_eq!(providers[0].protocol, CompatibleProtocol::AnthropicMessages);

        std::fs::write(
            config.path().join("config.json"),
            r#"{"openaiCompatible":{"custom":{"baseUrl":"https://example.com/v1","apiKeyEnv":"KEY","models":["x"],"protocol":"unknown"}}}"#,
        )
        .unwrap();
        assert!(openai_compatible_providers().is_err());
    }

    #[test]
    fn openai_compatible_config_loads_literal_headers() {
        let _guard = ENV_LOCK.lock().unwrap();
        clear_env();
        let config = tempfile::TempDir::new().unwrap();
        std::fs::write(
            config.path().join("config.json"),
            r#"{"openaiCompatible":{"cloudflare":{"baseUrl":"https://api.cloudflare.com/client/v4/accounts/test/ai/v1","apiKeyEnv":"CF_AIG_TOKEN","models":["openai/gpt-5.5"],"headers":{"cf-aig-gateway-id":"gateway"}}}}"#,
        )
        .unwrap();
        let _config_env = EnvGuard::set("CCP_CONFIG_DIR", config.path());

        let providers = openai_compatible_providers().unwrap();
        assert_eq!(
            providers[0].headers.get("cf-aig-gateway-id"),
            Some(&"gateway".to_string())
        );
    }

    #[test]
    fn openai_compatible_headers_reject_invalid_duplicates_and_reserved_names() {
        let invalid_name = BTreeMap::from([("bad header".to_string(), "value".to_string())]);
        assert!(parse_openai_compatible_headers("custom", &invalid_name).is_err());

        let duplicates = BTreeMap::from([
            ("X-Custom".to_string(), "one".to_string()),
            ("x-custom".to_string(), "two".to_string()),
        ]);
        assert!(parse_openai_compatible_headers("custom", &duplicates).is_err());

        for name in [
            "Authorization",
            "ACCEPT",
            "Content-Type",
            "content-length",
            "Host",
            "user-agent",
            "Connection",
            "keep-alive",
            "proxy-authenticate",
            "Proxy-Authorization",
            "proxy-connection",
            "TE",
            "trailer",
            "Transfer-Encoding",
            "upgrade",
        ] {
            let headers = BTreeMap::from([(name.to_string(), "value".to_string())]);
            let error = parse_openai_compatible_headers("custom", &headers)
                .expect_err("reserved header should fail")
                .to_string();
            assert!(error.contains("reserved header"), "{name}: {error}");
            assert!(!error.contains("value"));
        }
    }

    #[test]
    fn openai_compatible_headers_reject_injection_without_exposing_value() {
        let secret = "secret-value\r\nx-injected: true";
        let headers = BTreeMap::from([("x-custom".to_string(), secret.to_string())]);
        let error = parse_openai_compatible_headers("custom", &headers)
            .expect_err("invalid header value should fail")
            .to_string();
        assert!(error.contains("x-custom"));
        assert!(!error.contains(secret));
        assert!(!error.contains("secret-value"));
    }

    #[test]
    fn openai_compatible_summary_reports_only_header_count() {
        let _guard = ENV_LOCK.lock().unwrap();
        clear_env();
        let config = tempfile::TempDir::new().unwrap();
        std::fs::write(
            config.path().join("config.json"),
            r#"{"openaiCompatible":{"cloudflare":{"baseUrl":"https://api.cloudflare.com/client/v4/accounts/test/ai/v1","apiKeyEnv":"CF_AIG_TOKEN","models":["openai/gpt-5.5"],"headers":{"cf-aig-gateway-id":"private-gateway-name"}}}}"#,
        )
        .unwrap();
        let _config_env = EnvGuard::set("CCP_CONFIG_DIR", config.path());

        let summary = config_override_summary_lines(&load_config()).join("\n");
        assert!(summary.contains("1 custom headers"));
        assert!(!summary.contains("cf-aig-gateway-id"));
        assert!(!summary.contains("private-gateway-name"));
    }

    #[test]
    fn openai_compatible_config_rejects_reserved_name_and_duplicate_models() {
        let _guard = ENV_LOCK.lock().unwrap();
        clear_env();
        let config = tempfile::TempDir::new().unwrap();
        let _config_env = EnvGuard::set("CCP_CONFIG_DIR", config.path());
        std::fs::write(
            config.path().join("config.json"),
            r#"{"openaiCompatible":{"codex":{"baseUrl":"https://example.com/v1","apiKeyEnv":"KEY","models":["x"]}}}"#,
        )
        .unwrap();
        assert!(openai_compatible_providers().is_err());

        std::fs::write(
            config.path().join("config.json"),
            r#"{"openaiCompatible":{"custom":{"baseUrl":"https://example.com/v1","apiKeyEnv":"KEY","models":["x","x"]}}}"#,
        )
        .unwrap();
        assert!(openai_compatible_providers().is_err());
    }

    #[test]
    fn openai_compatible_config_loads_model_rewrites() {
        let _guard = ENV_LOCK.lock().unwrap();
        clear_env();
        let config = tempfile::TempDir::new().unwrap();
        let _config_env = EnvGuard::set("CCP_CONFIG_DIR", config.path());
        std::fs::write(
            config.path().join("config.json"),
            r#"{"openaiCompatible":{"cloudflare":{"baseUrl":"https://api.cloudflare.com/client/v4/accounts/test/ai/v1","apiKeyEnv":"CF_AIG_TOKEN","protocol":"anthropic-messages","models":["anthropic/claude-sonnet-5"],"modelRewrites":{"claude-opus-4-8":"anthropic/claude-opus-4.8"}}}}"#,
        )
        .unwrap();

        let providers = openai_compatible_providers().unwrap();
        assert_eq!(
            providers[0].model_rewrites.get("claude-opus-4-8"),
            Some(&"anthropic/claude-opus-4.8".to_string())
        );
    }

    #[test]
    fn openai_compatible_config_rejects_bad_model_rewrites() {
        let _guard = ENV_LOCK.lock().unwrap();
        clear_env();
        let config = tempfile::TempDir::new().unwrap();
        let _config_env = EnvGuard::set("CCP_CONFIG_DIR", config.path());

        // Empty rewrite target.
        std::fs::write(
            config.path().join("config.json"),
            r#"{"openaiCompatible":{"cloudflare":{"baseUrl":"https://example.com/v1","apiKeyEnv":"KEY","models":["x"],"modelRewrites":{"claude-opus-4-8":""}}}}"#,
        )
        .unwrap();
        assert!(openai_compatible_providers().is_err());

        // Rewrite key also present in models.
        std::fs::write(
            config.path().join("config.json"),
            r#"{"openaiCompatible":{"cloudflare":{"baseUrl":"https://example.com/v1","apiKeyEnv":"KEY","models":["claude-opus-4-8"],"modelRewrites":{"claude-opus-4-8":"anthropic/claude-opus-4.8"}}}}"#,
        )
        .unwrap();
        assert!(openai_compatible_providers().is_err());
    }

    #[test]
    fn openai_compatible_config_validates_cache_ttl() {
        let _guard = ENV_LOCK.lock().unwrap();
        clear_env();
        let config = tempfile::TempDir::new().unwrap();
        let _config_env = EnvGuard::set("CCP_CONFIG_DIR", config.path());

        // Valid 1h TTL on an anthropic-messages provider loads through.
        std::fs::write(
            config.path().join("config.json"),
            r#"{"openaiCompatible":{"cf":{"baseUrl":"https://example.com/v1","apiKeyEnv":"KEY","protocol":"anthropic-messages","models":["anthropic/claude-sonnet-5"],"cacheTtl":"1h"}}}"#,
        )
        .unwrap();
        let providers = openai_compatible_providers().unwrap();
        assert_eq!(providers[0].cache_ttl.as_deref(), Some("1h"));

        // Unsupported TTL value is rejected.
        std::fs::write(
            config.path().join("config.json"),
            r#"{"openaiCompatible":{"cf":{"baseUrl":"https://example.com/v1","apiKeyEnv":"KEY","protocol":"anthropic-messages","models":["x"],"cacheTtl":"30m"}}}"#,
        )
        .unwrap();
        assert!(openai_compatible_providers().is_err());

        // cacheTtl on the default openai-chat protocol is rejected (it would be a
        // silent no-op otherwise).
        std::fs::write(
            config.path().join("config.json"),
            r#"{"openaiCompatible":{"cf":{"baseUrl":"https://example.com/v1","apiKeyEnv":"KEY","models":["x"],"cacheTtl":"1h"}}}"#,
        )
        .unwrap();
        assert!(openai_compatible_providers().is_err());
    }

    #[test]
    fn openai_compatible_summary_reports_model_rewrite_count_only() {
        let _guard = ENV_LOCK.lock().unwrap();
        clear_env();
        let config = tempfile::TempDir::new().unwrap();
        std::fs::write(
            config.path().join("config.json"),
            r#"{"openaiCompatible":{"cloudflare":{"baseUrl":"https://api.cloudflare.com/client/v4/accounts/test/ai/v1","apiKeyEnv":"CF_AIG_TOKEN","protocol":"anthropic-messages","models":["anthropic/claude-sonnet-5"],"modelRewrites":{"claude-opus-4-8":"anthropic/claude-opus-4.8"}}}}"#,
        )
        .unwrap();
        let _config_env = EnvGuard::set("CCP_CONFIG_DIR", config.path());

        let summary = config_override_summary_lines(&load_config()).join("\n");
        assert!(summary.contains("1 model rewrites"));
        assert!(!summary.contains("claude-opus-4.8"));
    }

    #[test]
    fn codex_transport_defaults_to_websocket() {
        let _guard = ENV_LOCK.lock().unwrap();
        clear_env();
        let result = codex_transport();
        assert_eq!(result, CodexTransport::WebSocket);
    }

    #[test]
    fn codex_transport_reads_env() {
        let _guard = ENV_LOCK.lock().unwrap();
        clear_env();
        unsafe {
            std::env::set_var("CCP_CODEX_TRANSPORT", "auto");
        }
        assert_eq!(codex_transport(), CodexTransport::Auto);
    }

    #[test]
    fn codex_transport_env_websocket() {
        let _guard = ENV_LOCK.lock().unwrap();
        clear_env();
        unsafe {
            std::env::set_var("CCP_CODEX_TRANSPORT", "websocket");
        }
        assert_eq!(codex_transport(), CodexTransport::WebSocket);
    }

    #[test]
    fn codex_transport_invalid_env_falls_back_to_websocket() {
        let _guard = ENV_LOCK.lock().unwrap();
        clear_env();
        unsafe {
            std::env::set_var("CCP_CODEX_TRANSPORT", "invalid");
        }
        assert_eq!(codex_transport(), CodexTransport::WebSocket);
    }

    #[test]
    fn codex_transport_empty_env_falls_back_to_websocket() {
        let _guard = ENV_LOCK.lock().unwrap();
        clear_env();
        unsafe {
            std::env::set_var("CCP_CODEX_TRANSPORT", "");
        }
        assert_eq!(codex_transport(), CodexTransport::WebSocket);
    }

    #[test]
    fn parse_codex_transport_variants() {
        assert_eq!(parse_codex_transport("http"), Some(CodexTransport::Http));
        assert_eq!(
            parse_codex_transport("websocket"),
            Some(CodexTransport::WebSocket)
        );
        assert_eq!(parse_codex_transport("auto"), Some(CodexTransport::Auto));
        assert_eq!(parse_codex_transport(""), None);
        assert_eq!(parse_codex_transport("HTTP"), None);
        assert_eq!(parse_codex_transport("ws"), None);
    }

    #[test]
    fn codex_transport_as_str() {
        assert_eq!(CodexTransport::Http.as_str(), "http");
        assert_eq!(CodexTransport::WebSocket.as_str(), "websocket");
        assert_eq!(CodexTransport::Auto.as_str(), "auto");
    }

    #[test]
    fn log_env_presence_enables_legacy_verbose_and_stderr() {
        let _guard = ENV_LOCK.lock().unwrap();
        clear_env();
        let config = tempfile::TempDir::new().unwrap();
        let _config_env = EnvGuard::set("CCP_CONFIG_DIR", config.path());
        let _verbose_env = EnvGuard::set("CCP_LOG_VERBOSE", "0");
        let _stderr_env = EnvGuard::set("CCP_LOG_STDERR", "");

        let loaded = load_config();
        assert!(loaded.log_verbose);
        assert!(loaded.log_stderr);
    }

    #[test]
    fn log_config_values_apply_without_env() {
        let _guard = ENV_LOCK.lock().unwrap();
        clear_env();
        let config = tempfile::TempDir::new().unwrap();
        std::fs::write(
            config.path().join("config.json"),
            r#"{"log":{"verbose":true,"stderr":true}}"#,
        )
        .unwrap();
        let _config_env = EnvGuard::set("CCP_CONFIG_DIR", config.path());

        let loaded = load_config();
        assert!(loaded.log_verbose);
        assert!(loaded.log_stderr);
    }

    #[test]
    fn codex_responses_api_defaults_to_disabled() {
        let _guard = ENV_LOCK.lock().unwrap();
        clear_env();
        let config = tempfile::TempDir::new().unwrap();
        let _config_env = EnvGuard::set("CCP_CONFIG_DIR", config.path());

        assert!(!codex_responses_api());
    }

    #[test]
    fn codex_responses_api_reads_config_and_env_takes_precedence() {
        let _guard = ENV_LOCK.lock().unwrap();
        clear_env();
        let config = tempfile::TempDir::new().unwrap();
        std::fs::write(
            config.path().join("config.json"),
            r#"{"codex":{"responsesApi":true}}"#,
        )
        .unwrap();
        let _config_env = EnvGuard::set("CCP_CONFIG_DIR", config.path());

        assert!(codex_responses_api());
        let _responses_env = EnvGuard::set("CCP_CODEX_RESPONSES_API", "false");
        assert!(!codex_responses_api());
    }

    #[test]
    fn codex_responses_api_accepts_enabled_env_values() {
        let _guard = ENV_LOCK.lock().unwrap();
        clear_env();
        let config = tempfile::TempDir::new().unwrap();
        let _config_env = EnvGuard::set("CCP_CONFIG_DIR", config.path());

        for value in ["1", "true", "TRUE", "yes"] {
            let _responses_env = EnvGuard::set("CCP_CODEX_RESPONSES_API", value);
            assert!(codex_responses_api(), "{value}");
        }
    }

    #[test]
    fn codex_reasoning_summary_reads_config() {
        let _guard = ENV_LOCK.lock().unwrap();
        clear_env();
        let config = tempfile::TempDir::new().unwrap();
        std::fs::write(
            config.path().join("config.json"),
            r#"{"codex":{"reasoningSummary":"off"}}"#,
        )
        .unwrap();
        let _config_env = EnvGuard::set("CCP_CONFIG_DIR", config.path());

        assert_eq!(codex_reasoning_summary().as_deref(), Some("off"));
    }

    #[test]
    fn codex_reasoning_summary_env_overrides_config_and_empty_falls_through() {
        let _guard = ENV_LOCK.lock().unwrap();
        clear_env();
        let config = tempfile::TempDir::new().unwrap();
        std::fs::write(
            config.path().join("config.json"),
            r#"{"codex":{"reasoningSummary":"off"}}"#,
        )
        .unwrap();
        let _config_env = EnvGuard::set("CCP_CONFIG_DIR", config.path());
        {
            let _summary_env = EnvGuard::set("CCP_CODEX_REASONING_SUMMARY", "auto");
            assert_eq!(codex_reasoning_summary().as_deref(), Some("auto"));
        }
        {
            let _summary_env = EnvGuard::set("CCP_CODEX_REASONING_SUMMARY", "");
            assert_eq!(codex_reasoning_summary().as_deref(), Some("off"));
        }
    }

    #[test]
    fn codex_server_compaction_defaults_and_overrides() {
        let _guard = ENV_LOCK.lock().unwrap();
        clear_env();
        let config = tempfile::TempDir::new().unwrap();
        let _config_env = EnvGuard::set("CCP_CONFIG_DIR", config.path());

        assert!(!codex_server_compaction());
        {
            let _enabled_env = EnvGuard::set("CCP_CODEX_SERVER_COMPACTION", "on");
            assert!(codex_server_compaction());
        }
        std::fs::write(
            config.path().join("config.json"),
            r#"{"codex":{"serverCompaction":true}}"#,
        )
        .unwrap();
        assert!(codex_server_compaction());
        let _disabled_env = EnvGuard::set("CCP_CODEX_SERVER_COMPACTION", "false");
        assert!(!codex_server_compaction());
    }
}
