---
title: Choosing a provider
description: Compare claude-code-proxy providers by account, protocol, models, reasoning, multimodal input, tools, and operational tradeoffs.
---

One `serve` process supports every provider. Choose based on the account you have, model access, and the capabilities your work needs.

| Provider | Account | Upstream protocol | Model selection | Notable capabilities |
| --- | --- | --- | --- | --- |
| [Codex](/providers/codex/) | ChatGPT Plus or Pro | OpenAI Responses over WebSocket or HTTP SSE | Named Codex catalog, `-fast` variants | Function tools, image input, hosted web search, reasoning summaries, optional native Responses route |
| [Kimi](/providers/kimi/) | kimi.com with Kimi Code access | OpenAI-style chat completions | `kimi-for-coding` and aliases | Function tools, reasoning, image and video input |
| [Grok](/providers/grok/) | grok.com | Responses API | `grok-composer-2.5-fast`, `grok-4.5` | Function tools, reasoning, web search, X search, citations |
| [Cursor Agent](/providers/cursor-agent/) | Cursor account | HTTP/2 Connect stream | Cursor modes and `cursor:<model-id>` prefixes | Dynamic model catalog, effort variants, images, plan and ask modes, session continuation |
| [OpenAI-compatible API](/providers/openai-compatible/) | Provider API key | OpenAI Chat Completions | Exact IDs from `config.json` | Generic text, reasoning fields, images, function tools, and live SSE translation |

## Practical guidance

- Start with **Codex** when you have a ChatGPT subscription and want the most developed Claude Code translation path.
- Choose **Kimi** for the Kimi Code model and multimodal coding input.
- Choose **Grok** for Grok models and hosted web or X search.
- Choose **Cursor Agent** when you want Cursor's model catalog and agent modes. It depends on an installed Cursor Agent bundle for protobuf schemas.
- Configure an **OpenAI-compatible API** for API-key services such as Arcee or a self-hosted Chat Completions gateway.

## Shared behavior

All providers route by model ID, stream responses, translate Claude Code tool definitions, and report failures through the same Anthropic-shaped API. Built-in providers manage their own credentials; configurable OpenAI-compatible providers read bearer tokens from the named environment variables.

<div class="security-callout">
<strong>Account policy matters.</strong> Provider subscriptions, model access, regional availability, rate limits, and rules for unofficial clients can change. Review the terms for your account before using a provider through the proxy.
</div>

Use `claude-code-proxy models` for the current compact catalog and `claude-code-proxy models --full` for all advertised aliases. The [Models and routing](/using/models-and-routing/) page explains aliases and discovery.
