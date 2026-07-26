use crate::{config::OpenAiCompatibleProviderConfig, provider::Provider};
use anyhow::{Result, anyhow};
use std::collections::{BTreeMap, HashSet};
use std::sync::Arc;

pub const ANTHROPIC_STYLE_ALIASES: &[&str] = &[
    "haiku",
    "claude-haiku-4-5",
    "claude-haiku-4-5-20251001",
    "sonnet",
    "claude-sonnet-4-6",
    "claude-sonnet-5",
    "opus",
    "claude-opus-4-7",
    "claude-opus-4-8",
    "fable",
    "claude-fable-5",
];

pub const CURSOR_PREFIXES: &[&str] = &["cursor:", "cursor-plan:", "cursor-ask:"];

const CURSOR_LEGACY_MODELS: &[&str] = &[
    "cursor",
    "cursor-agent",
    "cursor-composer",
    "cursor-composer-fast",
    "cursor-plan",
    "cursor-ask",
    "composer-2.5",
    "composer-2.5-fast",
];

pub(crate) const CODEX_MODELS: &[&str] = &[
    "gpt-5.2",
    "gpt-5.3-codex",
    "gpt-5.3-codex-spark",
    "gpt-5.4",
    "gpt-5.4-mini",
    "gpt-5.5",
    "gpt-5.6-luna",
    "gpt-5.6-sol",
    "gpt-5.6-terra",
];

pub(crate) const KIMI_MODELS: &[&str] = &["kimi-for-coding", "kimi-k2.6", "kimi-k3", "k2.6", "k3"];
pub(crate) const GROK_MODELS: &[&str] = &["grok-composer-2.5-fast", "grok-4.5"];

pub struct Registry {
    /// Per-provider routable model IDs (each provider's `models` unioned with
    /// its `modelRewrites` keys). Routing is exact-match passthrough against
    /// this map — there is no alias fallback to a default provider.
    models: BTreeMap<String, Vec<String>>,
    handlers: BTreeMap<String, Arc<dyn Provider>>,
}

impl Registry {
    pub fn new() -> Self {
        Self::new_with_openai_compatible(Vec::new())
            .expect("built-in provider registry must be valid")
    }

    pub fn new_with_openai_compatible(
        configured: Vec<OpenAiCompatibleProviderConfig>,
    ) -> Result<Self> {
        validate_configured_models(&configured)?;
        let mut models: BTreeMap<String, Vec<String>> = BTreeMap::new();
        models.insert("codex".into(), expand_codex_models());
        models.insert(
            "kimi".into(),
            KIMI_MODELS.iter().map(|m| (*m).to_string()).collect(),
        );
        models.insert("cursor".into(), build_cursor_models());
        models.insert(
            "grok".into(),
            GROK_MODELS
                .iter()
                .map(|model| (*model).to_string())
                .collect(),
        );
        for provider in &configured {
            models.insert(provider.name.clone(), routable_models(provider));
        }

        let configured_by_name: BTreeMap<_, _> = configured
            .into_iter()
            .map(|provider| (provider.name.clone(), provider))
            .collect();
        let mut handlers = BTreeMap::new();
        for name in models.keys() {
            let handler: Arc<dyn Provider> = match name.as_str() {
                "codex" => Arc::new(crate::providers::codex::CodexProvider::new()),
                "kimi" => Arc::new(crate::providers::kimi::KimiProvider::new()),
                "cursor" => Arc::new(crate::providers::cursor::CursorProvider::new()),
                "grok" => Arc::new(crate::providers::grok::GrokProvider::new()),
                _ => {
                    let provider = configured_by_name
                        .get(name)
                        .ok_or_else(|| anyhow!("missing configuration for provider {name:?}"))?;
                    let headers = crate::config::parse_openai_compatible_headers(
                        &provider.name,
                        &provider.headers,
                    )?;
                    Arc::new(
                        crate::providers::openai_compat::OpenAiCompatibleProvider::new(
                            provider.name.clone(),
                            provider.base_url.clone(),
                            provider.api_key_env.clone(),
                            provider.models.clone(),
                            provider.protocol,
                            headers,
                            provider.model_rewrites.clone(),
                            provider.cache_ttl.clone(),
                        ),
                    )
                }
            };
            handlers.insert(name.clone(), handler);
        }

        Ok(Self { models, handlers })
    }

    pub fn try_with_default_alias() -> Result<Self> {
        Self::new_with_openai_compatible(crate::config::openai_compatible_providers()?)
    }

    pub fn with_default_alias() -> Self {
        Self::try_with_default_alias().expect("invalid provider configuration")
    }

    pub fn list_provider_names(&self) -> Vec<String> {
        let mut names: Vec<String> = self.handlers.keys().cloned().collect();
        names.sort_unstable();
        names
    }

    pub fn provider(&self, name: &str) -> Option<Arc<dyn Provider>> {
        self.handlers.get(name).cloned()
    }

    pub fn supported_models_for(&self, provider: &str) -> Vec<String> {
        let mut models = self.models.get(provider).cloned().unwrap_or_default();
        models.sort_unstable();
        models
    }

    pub fn all_supported_models(&self) -> Vec<(String, String)> {
        let mut out = Vec::new();
        for provider in self.handlers.keys() {
            for model in self.supported_models_for(provider) {
                out.push((model, provider.clone()));
            }
        }
        out
    }

    pub fn grouped_models(&self) -> BTreeMap<String, Vec<String>> {
        let mut out = BTreeMap::new();
        for provider in self.handlers.keys() {
            out.insert(provider.clone(), self.supported_models_for(provider));
        }
        out
    }

    /// Route a model ID to a provider by exact match (passthrough).
    ///
    /// Precedence: Cursor prefix routing, then exact match against each
    /// provider's routable model set (configured `models` plus `modelRewrites`
    /// keys). There is no Anthropic-alias fallback to a default provider: a
    /// model ID that no configured provider claims returns `None`.
    pub fn provider_for_model(&self, raw_model: &str) -> Option<Arc<dyn Provider>> {
        let normalized = normalize_incoming_model(raw_model);
        if is_cursor_model(&normalized) {
            return self.handlers.get("cursor").cloned();
        }

        for (name, models) in &self.models {
            if models.iter().any(|candidate| candidate == &normalized) {
                return self.handlers.get(name).cloned();
            }
        }

        None
    }

    pub fn unknown_model_message(&self) -> String {
        let mut parts = Vec::new();
        for (provider, models) in self.grouped_models() {
            let mut models = models;
            models.sort_unstable();
            parts.push(format!("{}: {}", provider, models.join(", ")));
        }
        format!("Supported: {}.", parts.join("; "))
    }
}

impl Default for Registry {
    fn default() -> Self {
        Self::new()
    }
}

/// A configured provider's full routable set: its `models` plus every
/// `modelRewrites` key (the client-facing IDs the proxy accepts and rewrites).
fn routable_models(provider: &OpenAiCompatibleProviderConfig) -> Vec<String> {
    let mut entries = provider.models.clone();
    entries.extend(provider.model_rewrites.keys().cloned());
    entries.sort_unstable();
    entries.dedup();
    entries
}

fn validate_configured_models(configured: &[OpenAiCompatibleProviderConfig]) -> Result<()> {
    // Reserve the built-in providers' exact model IDs so a configured provider
    // cannot shadow them. Anthropic-style aliases are intentionally NOT reserved
    // anymore: routing is exact-match passthrough, so a configured provider may
    // claim a recognized ID like `claude-opus-4-8` (typically via modelRewrites).
    let mut seen: BTreeMap<String, String> = BTreeMap::new();
    for model in expand_codex_models() {
        seen.insert(model, "built-in provider".to_string());
    }
    for model in KIMI_MODELS
        .iter()
        .chain(GROK_MODELS)
        .chain(CURSOR_LEGACY_MODELS)
    {
        seen.insert((*model).to_string(), "built-in provider".to_string());
    }
    for provider in configured {
        for model in provider.models.iter().chain(provider.model_rewrites.keys()) {
            if is_cursor_model(model) {
                return Err(anyhow!(
                    "openaiCompatible.{}.models contains reserved Cursor model {model:?}",
                    provider.name
                ));
            }
            if let Some(existing) = seen.insert(model.clone(), provider.name.clone()) {
                return Err(anyhow!(
                    "model {model:?} is configured by both {existing} and {}",
                    provider.name
                ));
            }
        }
    }
    Ok(())
}

pub fn normalize_incoming_model(model: &str) -> String {
    let suffix = "[1m]";
    if model.len() >= suffix.len() && model.to_ascii_lowercase().ends_with(suffix) {
        return model[..model.len() - suffix.len()].to_string();
    }
    model.to_string()
}

pub fn is_anthropic_alias(model: &str) -> bool {
    ANTHROPIC_STYLE_ALIASES.contains(&model)
}

pub fn is_cursor_model(model: &str) -> bool {
    if CURSOR_LEGACY_MODELS.contains(&model) {
        return true;
    }

    CURSOR_PREFIXES
        .iter()
        .any(|prefix| model.starts_with(prefix))
}

fn expand_codex_models() -> Vec<String> {
    let mut set = HashSet::new();
    let mut out = Vec::new();
    for model in CODEX_MODELS {
        if set.insert((*model).to_string()) {
            out.push((*model).to_string());
        }
        let fast = format!("{model}-fast");
        if set.insert(fast.clone()) {
            out.push(fast);
        }
    }
    out.sort_unstable();
    out
}

fn build_cursor_models() -> Vec<String> {
    let mut out: Vec<String> = CURSOR_LEGACY_MODELS
        .iter()
        .map(|s| (*s).to_string())
        .collect();
    out.sort_unstable();
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::CompatibleProtocol;

    fn cloudflare_opus_provider() -> OpenAiCompatibleProviderConfig {
        OpenAiCompatibleProviderConfig {
            name: "cloudflare-anthropic".to_string(),
            base_url: "https://api.cloudflare.com/client/v4/accounts/ACCOUNT/ai/v1".to_string(),
            api_key_env: "CF_AIG_TOKEN".to_string(),
            models: vec!["anthropic/claude-sonnet-5".to_string()],
            protocol: CompatibleProtocol::AnthropicMessages,
            headers: BTreeMap::new(),
            cache_ttl: None,
            model_rewrites: BTreeMap::from([(
                "claude-opus-4-8".to_string(),
                "anthropic/claude-opus-4.8".to_string(),
            )]),
        }
    }

    #[test]
    fn normalize_model_trims_hint() {
        assert_eq!(normalize_incoming_model("gpt-5.4-fast[1m]"), "gpt-5.4-fast");
        assert_eq!(normalize_incoming_model("gpt-5.4-fast"), "gpt-5.4-fast");
    }

    #[test]
    fn unclaimed_anthropic_alias_returns_none() {
        // Passthrough routing: with no provider claiming these aliases there is
        // no Codex/alias fallback, so they are unknown models.
        let registry = Registry::new();
        assert!(registry.provider_for_model("haiku").is_none());
        assert!(registry.provider_for_model("claude-opus-4-8").is_none());
        assert!(registry.provider_for_model("sonnet").is_none());
    }

    #[test]
    fn builtin_exact_models_still_route() {
        let registry = Registry::new();
        assert_eq!(
            registry.provider_for_model("gpt-5.5").unwrap().name(),
            "codex"
        );
        assert_eq!(
            registry
                .provider_for_model("kimi-for-coding")
                .unwrap()
                .name(),
            "kimi"
        );
        assert_eq!(
            registry.provider_for_model("grok-4.5").unwrap().name(),
            "grok"
        );
    }

    #[test]
    fn rewrite_key_routes_to_configured_provider() {
        // A recognized Claude Code id used as a rewrite key routes to the
        // configured gateway, and the `[1m]` suffix is stripped before matching.
        let registry =
            Registry::new_with_openai_compatible(vec![cloudflare_opus_provider()]).unwrap();
        for model in ["claude-opus-4-8", "claude-opus-4-8[1m]"] {
            let provider = registry
                .provider_for_model(model)
                .unwrap_or_else(|| panic!("{model} should route"));
            assert_eq!(provider.name(), "cloudflare-anthropic");
        }
    }

    #[test]
    fn qualified_model_routes_by_exact_match() {
        let registry =
            Registry::new_with_openai_compatible(vec![cloudflare_opus_provider()]).unwrap();
        let provider = registry
            .provider_for_model("anthropic/claude-sonnet-5[1m]")
            .unwrap();
        assert_eq!(provider.name(), "cloudflare-anthropic");
    }

    #[test]
    fn rewrite_keys_are_advertised_for_discovery() {
        let registry =
            Registry::new_with_openai_compatible(vec![cloudflare_opus_provider()]).unwrap();
        let models = registry.supported_models_for("cloudflare-anthropic");
        assert!(models.iter().any(|m| m == "claude-opus-4-8"));
        assert!(models.iter().any(|m| m == "anthropic/claude-sonnet-5"));
    }

    #[test]
    fn configured_openai_model_routes_with_slash_and_hint() {
        let registry = Registry::new_with_openai_compatible(vec![OpenAiCompatibleProviderConfig {
            name: "arcee".to_string(),
            base_url: "https://api.arcee.ai/api/v1".to_string(),
            api_key_env: "ARCEE_API_KEY".to_string(),
            models: vec!["moonshotai/kimi-k2.7-code".to_string()],
            protocol: Default::default(),
            headers: BTreeMap::new(),
            cache_ttl: None,
            model_rewrites: BTreeMap::new(),
        }])
        .unwrap();
        let provider = registry
            .provider_for_model("moonshotai/kimi-k2.7-code[1m]")
            .unwrap();
        assert_eq!(provider.name(), "arcee");
    }

    #[test]
    fn configured_models_must_not_collide() {
        let result = Registry::new_with_openai_compatible(vec![
            OpenAiCompatibleProviderConfig {
                name: "one".to_string(),
                base_url: "https://one.example/v1".to_string(),
                api_key_env: "ONE_API_KEY".to_string(),
                models: vec!["shared/model".to_string()],
                protocol: Default::default(),
                headers: BTreeMap::new(),
                cache_ttl: None,
                model_rewrites: BTreeMap::new(),
            },
            OpenAiCompatibleProviderConfig {
                name: "two".to_string(),
                base_url: "https://two.example/v1".to_string(),
                api_key_env: "TWO_API_KEY".to_string(),
                models: vec!["shared/model".to_string()],
                protocol: Default::default(),
                headers: BTreeMap::new(),
                cache_ttl: None,
                model_rewrites: BTreeMap::new(),
            },
        ]);
        assert!(result.is_err());
    }

    #[test]
    fn configured_rewrite_keys_must_not_collide() {
        // Two providers cannot both claim the same recognized id via rewrite keys.
        let result = Registry::new_with_openai_compatible(vec![
            OpenAiCompatibleProviderConfig {
                name: "one".to_string(),
                base_url: "https://one.example/v1".to_string(),
                api_key_env: "ONE_API_KEY".to_string(),
                models: vec!["one/model".to_string()],
                protocol: Default::default(),
                headers: BTreeMap::new(),
                cache_ttl: None,
                model_rewrites: BTreeMap::from([(
                    "claude-opus-4-8".to_string(),
                    "one/opus".to_string(),
                )]),
            },
            OpenAiCompatibleProviderConfig {
                name: "two".to_string(),
                base_url: "https://two.example/v1".to_string(),
                api_key_env: "TWO_API_KEY".to_string(),
                models: vec!["two/model".to_string()],
                protocol: Default::default(),
                headers: BTreeMap::new(),
                cache_ttl: None,
                model_rewrites: BTreeMap::from([(
                    "claude-opus-4-8".to_string(),
                    "two/opus".to_string(),
                )]),
            },
        ]);
        assert!(result.is_err());
    }

    #[test]
    fn configured_provider_must_not_shadow_builtin_model() {
        let result = Registry::new_with_openai_compatible(vec![OpenAiCompatibleProviderConfig {
            name: "custom".to_string(),
            base_url: "https://example.test/v1".to_string(),
            api_key_env: "CUSTOM_API_KEY".to_string(),
            models: vec!["gpt-5.5".to_string()],
            protocol: Default::default(),
            headers: BTreeMap::new(),
            cache_ttl: None,
            model_rewrites: BTreeMap::new(),
        }]);
        assert!(result.is_err());
    }

    #[test]
    fn configured_headers_must_not_override_authorization() {
        let result = Registry::new_with_openai_compatible(vec![OpenAiCompatibleProviderConfig {
            name: "custom".to_string(),
            base_url: "https://example.test/v1".to_string(),
            api_key_env: "CUSTOM_API_KEY".to_string(),
            models: vec!["custom/model".to_string()],
            protocol: Default::default(),
            headers: BTreeMap::from([("Authorization".to_string(), "secret".to_string())]),
            cache_ttl: None,
            model_rewrites: BTreeMap::new(),
        }]);
        let error = result
            .err()
            .expect("reserved header should fail")
            .to_string();
        assert!(error.contains("reserved header"));
        assert!(!error.contains("secret"));
    }

    #[test]
    fn cursor_prefix_routes() {
        let registry = Registry::new();
        assert_eq!(
            registry
                .provider_for_model("cursor:gpt-5.5")
                .unwrap()
                .name(),
            "cursor"
        );
        assert_eq!(
            registry
                .provider_for_model("cursor-plan:gpt-5.5")
                .unwrap()
                .name(),
            "cursor"
        );
        assert_eq!(
            registry
                .provider_for_model("cursor-ask:gpt-5.5")
                .unwrap()
                .name(),
            "cursor"
        );
    }
}
