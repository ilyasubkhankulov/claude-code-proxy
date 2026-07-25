---
title: OpenAI-compatible APIs
description: Configure OpenAI Chat Completions and Anthropic Messages compatible APIs, including Arcee and Cloudflare AI Gateway.
---

The proxy can route exact model IDs to user-defined APIs using either OpenAI Chat Completions or native Anthropic Messages. This is useful for smaller providers and self-hosted gateways without adding provider-specific code. The historical `openaiCompatible` configuration name covers both protocols.

## Configure Arcee

Add this to the proxy configuration file shown by `claude-code-proxy serve`:

```json
{
  "openaiCompatible": {
    "arcee": {
      "baseUrl": "https://api.arcee.ai/api/v1",
      "apiKeyEnv": "ARCEE_API_KEY",
      "models": [
        "trinity-large-thinking",
        "deepseek-ai/deepseek-v4-pro",
        "moonshotai/kimi-k2.7-code",
        "moonshotai/kimi-k2.6",
        "zai-org/glm-5.2",
        "minimaxai/minimax-m3"
      ]
    }
  }
}
```

`baseUrl` is the API root. The proxy appends `/chat/completions`. `apiKeyEnv` is the name of an environment variable, not the API key itself.

If your key is stored in `.env`, source it into the shell that starts the proxy:

```sh
set -a
. ./.env
set +a
claude-code-proxy serve
```

The file should contain:

```sh
ARCEE_API_KEY=your-key
```

A `.env` file is not loaded automatically. This keeps startup behavior explicit and also works with other secret managers that export environment variables.

## Start Claude Code

In another terminal, select exact IDs from the configured catalog:

```sh
ANTHROPIC_BASE_URL=http://127.0.0.1:18765 \
ANTHROPIC_AUTH_TOKEN=unused \
ANTHROPIC_MODEL=moonshotai/kimi-k2.7-code \
ANTHROPIC_SMALL_FAST_MODEL=minimaxai/minimax-m3 \
CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1 \
CLAUDE_CODE_DISABLE_NONSTREAMING_FALLBACK=1 \
  claude
```

Run `claude-code-proxy models` to confirm that the `arcee` group and its model IDs were loaded.

## Configure Cloudflare AI Gateway

Cloudflare AI Gateway's current REST API uses the same generic provider configuration, with one custom header to select a named gateway:

```json
{
  "openaiCompatible": {
    "cloudflare-anthropic": {
      "baseUrl": "https://api.cloudflare.com/client/v4/accounts/<ACCOUNT_ID>/ai/v1",
      "apiKeyEnv": "CF_AIG_TOKEN",
      "protocol": "anthropic-messages",
      "headers": {
        "cf-aig-gateway-id": "<GATEWAY_ID>"
      },
      "models": [
        "anthropic/claude-opus-5",
        "anthropic/claude-sonnet-5",
        "anthropic/claude-fable-5",
        "anthropic/claude-haiku-4-5-20251001"
      ]
    },
    "cloudflare-openai": {
      "baseUrl": "https://api.cloudflare.com/client/v4/accounts/<ACCOUNT_ID>/ai/v1",
      "apiKeyEnv": "CF_AIG_TOKEN",
      "headers": {
        "cf-aig-gateway-id": "<GATEWAY_ID>"
      },
      "models": ["openai/gpt-5.5"]
    }
  }
}
```

Export `CF_AIG_TOKEN` in the shell that starts the proxy. For example, add it to the `.env` file described above, then source that file before running `claude-code-proxy serve`:

```sh
CF_AIG_TOKEN=your-cloudflare-api-token
```

For `anthropic-messages`, the proxy calls `.../ai/v1/messages` and forwards native Messages tools and message content blocks. For compatibility with gateways that accept only Anthropic's string form, top-level system blocks plus any `system` or `developer` history entries are flattened into one top-level `system` string before forwarding; if a system block carried a `cache_control` breakpoint, it is re-expressed as a top-level `cache_control` so system-prompt caching still works (see [Prompt caching](#prompt-caching-and-cost-control)). Claude Code's beta `context_management` extension is omitted because compatible gateways may reject it. For the default `openai-chat` protocol, it calls `.../ai/v1/chat/completions` and uses the OpenAI translator. Both authenticate with `Authorization: Bearer $CF_AIG_TOKEN` and send `cf-aig-gateway-id` to select the configured gateway. Incoming `anthropic-version` and `anthropic-beta` headers are forwarded only on native Anthropic requests; authorization, cookies, and arbitrary client headers are not.

The `models` array is the proxy's exact local routing allowlist, not a catalog fetched from Cloudflare. Cloudflare's REST API can route available OpenAI, Anthropic, Google, and Workers AI models; update this array when you want to expose another model to Claude Code.

Claude Code can use any configured model through the local proxy:

```sh
ANTHROPIC_BASE_URL=http://127.0.0.1:18765 \
ANTHROPIC_AUTH_TOKEN=unused \
ANTHROPIC_MODEL=anthropic/claude-sonnet-5 \
ANTHROPIC_SMALL_FAST_MODEL=openai/gpt-5.5 \
CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1 \
CLAUDE_CODE_DISABLE_NONSTREAMING_FALLBACK=1 \
  claude
```

Cloudflare introduced this REST API in May 2026 to replace the deprecated gateway-specific `/compat` interface. Existing `/compat` configurations remain usable, but new configurations should use the REST base URL above.

Custom `headers` values are stored literally in `config.json`. Use them only for non-secret routing metadata such as a gateway ID; credentials belong in the environment variable named by `apiKeyEnv`.

### Model routing is exact-match passthrough

The proxy routes each incoming model ID to whichever configured provider claims it by exact match (after stripping any `[1m]` suffix). There is no built-in alias fallback: a bare Anthropic-style name such as `sonnet`, `opus`, or `claude-opus-4-8` is only routable when a provider explicitly lists it — in `models`, or as a `modelRewrites` key (below). An unclaimed model ID returns an "unknown model" error rather than silently routing to a default provider. Built-in providers (Codex `gpt-*`, Kimi, Cursor, Grok) continue to route by their exact model IDs.

### Rewriting a client model ID to a gateway model ID

Claude Code enables its **auto permission mode** and **1M context window** only for model IDs it recognizes on its built-in list — the dash form, e.g. `claude-opus-4-8`, plus the client-side `[1m]` suffix. Cloudflare's gateway, however, only accepts the qualified id `anthropic/claude-opus-4.8` (the dash form 404s). Resolving the alias to the gateway id on the client (e.g. `ANTHROPIC_DEFAULT_OPUS_MODEL`) makes Claude Code see an unrecognized id and disables auto mode.

`modelRewrites` bridges this inside the proxy, invisibly to the client. Each entry maps an incoming (client-facing) model ID to the model ID actually sent upstream:

```json
{
  "openaiCompatible": {
    "cloudflare-anthropic": {
      "baseUrl": "https://api.cloudflare.com/client/v4/accounts/<ACCOUNT_ID>/ai/v1",
      "apiKeyEnv": "CF_AIG_TOKEN",
      "protocol": "anthropic-messages",
      "headers": { "cf-aig-gateway-id": "<GATEWAY_ID>" },
      "models": ["anthropic/claude-sonnet-5"],
      "modelRewrites": {
        "claude-opus-4-8": "anthropic/claude-opus-4.8",
        "claude-sonnet-5": "anthropic/claude-sonnet-5",
        "claude-haiku-4-5-20251001": "anthropic/claude-haiku-4-5-20251001"
      }
    }
  }
}
```

A rewrite key is implicitly routable and is what `/v1/models` advertises, so Claude Code discovers the recognized `claude-opus-4-8`. When a request arrives for that id, the proxy routes it to this provider and rewrites the wire `model` to `anthropic/claude-opus-4.8` before calling Cloudflare. Because the client only ever sees the recognized id, selecting `claude-opus-4-8[1m]` keeps auto mode on and raises the local context ceiling to 1,000,000 tokens while the gateway serves Opus 4.8.

Rules: a rewrite key must not also appear in the same provider's `models`; rewrite targets must be non-empty; and no two configured providers may claim the same incoming id (via `models` or `modelRewrites`). Keys may not shadow a built-in provider's exact model ID.

### Using the full 1M context window

Cloudflare's gateway serves Claude models with a 1,000,000-token upstream context window, but Claude Code's `/v1/models` discovery response carries no context-size field, so Claude Code has no way to learn that from the wire. By default it treats every discovered model as a 200k-token model and caps local auto-compaction there. Setting `CLAUDE_CODE_AUTO_COMPACT_WINDOW` alone does not help either: Claude Code clamps that variable to whatever context size it already believes the model has.

The only thing that raises Claude Code's local auto-compact ceiling to 1,000,000 tokens is a literal `[1m]` suffix in the model ID string it operates on. This is a pure client-side string check on any model name; when present, Claude Code also adds the `anthropic-beta: context-1m-2025-08-07` header to the outgoing request.

Cloudflare's gateway accepts that beta header and serves the larger context window, but it unconditionally rejects Claude Code's `context_management` beta extension (`context_management: Extra inputs are not permitted`), which is why the proxy strips that field before forwarding native Anthropic requests (see above). The `anthropic-beta` header itself is still forwarded, so the 1M path works once Claude Code decides to send it.

Two ways to select a 1M gateway model:

- **Recognized id via `modelRewrites` (keeps auto mode).** Configure a rewrite key as shown above, then select the recognized id with the suffix — for example `/model claude-opus-4-8[1m]`, or at launch:

  ```sh
  ANTHROPIC_BASE_URL=http://127.0.0.1:18766 \
  ANTHROPIC_AUTH_TOKEN=unused \
  ANTHROPIC_MODEL=claude-opus-4-8[1m] \
  CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY=1 \
    claude
  ```

  Because Claude Code sees a recognized id, **auto permission mode stays enabled**. The proxy rewrites the wire model to the gateway id and routes it to `cloudflare-anthropic`.

- **Qualified gateway id directly (auto mode off).** You can also select the gateway id itself, e.g. `/model anthropic/claude-sonnet-5[1m]`. This routes correctly and gives 1M, but Claude Code does not recognize the id and therefore runs in default (prompt-per-action) mode rather than auto mode.

Confirm with `/context`. It should report `Auto-compact window: 1000000 tokens`, and the proxy TUI should attribute the request to `cloudflare-anthropic`.

The proxy's `[1m]`-suffix normalization (`normalize_incoming_model`) strips the suffix before matching, so both `claude-opus-4-8[1m]` (a rewrite key) and `anthropic/claude-sonnet-5[1m]` (an exact `models` entry) route to the `cloudflare-anthropic` provider.

## Prompt caching and cost control

When you point Claude Code at a gateway with your own API key, you pay per token, so keeping Anthropic's automatic prompt caching intact matters. Prompt caching **does work** through the Cloudflare AI Gateway, with one nuance worth understanding.

Anthropic prompt caching is *prefix-based*: a `cache_control` breakpoint caches everything before it in the request (tools → system → messages, in that order). The Cloudflare gateway requires the `anthropic-messages` `system` field to be a plain string, so a `cache_control` breakpoint placed *on a system block* cannot be sent as-is. To avoid silently dropping system-prompt caching, the proxy **re-expresses that breakpoint as a top-level `cache_control`** (preserving the client's TTL) when — and only when — the client asked for system caching. The gateway accepts top-level `cache_control`, and because it auto-applies a breakpoint to the last cacheable block, the whole prefix (including the flattened system string) is cached. `cache_control` markers on tools and message content blocks are forwarded unchanged and cache on their own. Net effect: Claude Code's caching, including the system prompt, is preserved.

Two caveats:

- The gateway does **not** return `cache_creation_input_tokens` / `cache_read_input_tokens` in its usage object, so cache activity is invisible in reported usage even though the savings are real (cached tokens drop out of `input_tokens`).
- Caching only kicks in above Anthropic's minimum cacheable prefix (~1024 tokens for Sonnet/Opus, ~2048 for Haiku); shorter prefixes are billed in full regardless.

To keep caching effective and background traffic cheap, on the Claude Code side:

- **Avoid mid-session model switches.** Each model has its own cache; `/model`, `opusplan` plan-mode toggles, and auto-fallback all invalidate it and force a full uncached re-read.
- **Do not set `DISABLE_PROMPT_CACHING`** (or the per-tier variants). They raise cost — they exist for debugging only.
- **Route background work to a cheap model** with `ANTHROPIC_DEFAULT_HAIKU_MODEL` (session titles, summaries, and other background calls use the `haiku` alias). Point it at a configured, cheap gateway model.
- **Cut nonessential traffic** with `CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1` (disables auto-updates, telemetry, error reporting, release notes, and background availability/model-discovery refreshes).
- **Cap output** with `CLAUDE_CODE_MAX_OUTPUT_TOKENS` and thinking with `MAX_THINKING_TOKENS` if you want tighter per-turn ceilings — but keep `CLAUDE_CODE_MAX_OUTPUT_TOKENS` modest, since a large value shrinks usable context before auto-compaction.

## What Claude Code disables behind a custom base URL

Some Claude Code features are gated on the client side by `ANTHROPIC_BASE_URL` and are switched off whenever the host is not `api.anthropic.com`. The proxy cannot re-enable these — they are decided before any request reaches it:

- **Remote Control** is unavailable through any custom base URL.
- **MCP tool search** is off by default on a non-first-party host. Re-enable it with `ENABLE_TOOL_SEARCH=true` *only* if your gateway forwards `tool_reference` blocks and serves a model that supports them; otherwise leave it off.
- **The WebFetch preflight** and the **fast-mode availability check** call `api.anthropic.com` directly rather than through the proxy. If egress to Anthropic is blocked they can report spurious errors even when inference works. Set `"skipWebFetchPreflight": true` in `settings.json` to skip the preflight.

If your gateway strips the `anthropic-beta` header (this proxy does **not** — it forwards `anthropic-version` and `anthropic-beta` verbatim) you may see `400 Extra inputs are not permitted`; the client-side fallback is `CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS=1`.

For a TLS-inspecting corporate proxy in front of the gateway, point Node at the CA bundle with `NODE_EXTRA_CA_CERTS=/path/to/ca.pem` rather than disabling verification.

## Add another compatible API

Add another entry beneath `openaiCompatible`. Provider names must be unique and may contain letters, numbers, hyphens, and underscores. Model IDs must be unique across all built-in and configured providers.

```json
{
  "openaiCompatible": {
    "local-gateway": {
      "baseUrl": "http://127.0.0.1:8000/v1",
      "apiKeyEnv": "LOCAL_GATEWAY_API_KEY",
      "models": ["my-org/my-model"]
    }
  }
}
```

The proxy validates the provider catalog at startup. It does not require every configured API key until a request is routed to that provider.

## Compatibility

Choose a protocol per entry:

- `openai-chat` (the default) requires `POST <baseUrl>/chat/completions`, standard Chat Completions messages/function tools, JSON responses, and OpenAI-style SSE chunks ending in `[DONE]`. Requests and responses are translated.
- `anthropic-messages` requires `POST <baseUrl>/messages`. Anthropic request JSON and JSON/SSE response bodies pass through natively, while only safe response headers are copied. Note that the Cloudflare gateway's validator is stricter than `api.anthropic.com`: it requires `system` to be a plain string (structured system arrays are rejected), so the proxy flattens system blocks and drops the `context_management` beta field. A `cache_control` breakpoint on a system block is re-expressed as a top-level `cache_control` so system-prompt caching is preserved through the gateway.

Both protocols use bearer-token authentication and configured literal headers. Token counting is an approximation performed locally and does not call the upstream API. The OpenAI translator recognizes `reasoning_content` and `reasoning` response fields in addition to standard text and tool calls.

An existing Homebrew installation needs a release containing this feature. After such a release, update with:

```sh
brew upgrade claude-code-proxy
```
