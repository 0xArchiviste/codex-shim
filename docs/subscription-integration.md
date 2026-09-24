# Subscription passthrough integrations

`codex-shim` can expose subscription-backed models without storing Dashboard
API keys in `~/.codex-shim/models.json`:

- **ChatGPT/Codex passthrough** uses the Codex access token created by
  `codex login` and forwards native `/v1/responses` requests to ChatGPT's Codex
  backend.
- **Cursor model passthrough** uses the local `cursor-agent` OAuth session
  created by `cursor-agent login` and exposes selected models with stable
  `cx-*` aliases.

Both integrations are optional, auth-gated, and advertised only when the local
login state is usable. They are different from BYOK routes: you do not add a
Dashboard API key for these subscription flows.

---

## Quick check

```bash
codex-shim doctor
codex-shim list
codex-shim status
```

Useful health fields:

```json
{
  "chatgpt_passthrough": true,
  "cursor_passthrough": true
}
```

`codex-shim doctor` is the safest first diagnostic because it does not start or
stop the daemon, write config, call model providers, or print token contents.

---

## ChatGPT/Codex passthrough

### What it does

When `~/.codex/auth.json` exists and contains `tokens.access_token`, the shim
adds ChatGPT/Codex model slugs to discovery surfaces such as:

- `codex-shim list`
- `/health`
- `/v1/models`
- the generated `.codex-shim/custom_model_catalog.json`

Current fallback slugs include `gpt-5.5` and related GPT/Codex slugs. The shim
keeps Codex's native Responses payload shape and forwards it to:

```text
https://chatgpt.com/backend-api/codex/responses
```

It sends the Codex access token as `Authorization: Bearer ...` and, when
present, the account id from `auth.json`. The token is not written into the
custom model catalog.

### Setup

```bash
codex login
codex-shim generate
codex-shim list
```

If `gpt-5.5` appears, you can select it from the Codex picker or run:

```bash
codex-shim model use gpt-5.5
```

For passthrough-only use, `~/.codex-shim/models.json` may be missing. The shim
can still generate a catalog containing subscription-backed entries when the
Codex auth file is valid.

### Disable

```bash
export CODEX_SHIM_DISABLE_CHATGPT=1
```

After disabling, regenerate or restart the shim if you need discovery surfaces
to stop listing ChatGPT passthrough entries immediately.

### Troubleshooting

- Run `codex login` again if `codex-shim doctor` reports ChatGPT passthrough as
  unavailable.
- Confirm the auth file exists at `~/.codex/auth.json`. Do not paste or upload
  the file; it contains tokens.
- If the model picker still does not show GPT/Codex slugs, run
  `codex-shim generate` and check `codex-shim list` before debugging Desktop
  picker behavior.
- If `/health` reports `chatgpt_passthrough: false`, the daemon process may have
  been started before login or with `CODEX_SHIM_DISABLE_CHATGPT` set.

---

## Cursor model passthrough

### What it does

When `cursor-agent status` reports an active login, the shim exposes:

```text
composer-2-5
cx-auto
cx-grok-4-7
cx-fable-5-1
cx-fable-5-1-high
cx-fable-5
cx-fable-5-high
cx-opus-5
cx-opus-5-5
cx-sol-5-6
cx-sol-5-6-high
```

Requests to that slug are converted into a prompt for `cursor-agent --print`
using your local CLI OAuth session. This is subscription passthrough, not
Dashboard API-key billing.

Every `cx-*` alias also has `-jev-io` and `-jev-io-max` profiles when the
top-level `jev_io` configuration is enabled. `cx-autogrok` is a combination
profile that applies active Jev IO to the shared input, runs `cx-auto` and
`cx-grok-4-7` concurrently in read-only `ask` mode, then asks Jev to select the
result. The read-only constraint prevents competing agents from mutating the
same workspace.

### Setup

```bash
cursor-agent login
cursor-agent status
codex-shim generate
codex-shim list
```

Then select a Cursor model in the picker or run:

```bash
codex-shim model use cx-auto
```

The helper script is optional, but convenient:

```bash
scripts/codex-shim-install-cursor-composer
```

It regenerates the local catalog/config and sets `composer-2-5` as the active
model when `cursor-agent status` reports an active login.

### Binary and workspace overrides

If `cursor-agent` is not on `PATH`, point the shim at it explicitly:

```bash
export CURSOR_AGENT_BIN=/path/to/cursor-agent
```

By default, the cursor-agent child process runs in the current working
directory. Override that with:

```bash
export CODEX_SHIM_CURSOR_WORKSPACE=/path/to/workspace
```

### Disable

```bash
export CODEX_SHIM_DISABLE_CURSOR=1
```

### Important: do not use Dashboard API keys for this flow

Do **not** configure Composer through `cursor-api.standardagents.ai` unless you
intentionally want Dashboard API-key billing (`crsr_...`). For subscription
passthrough, the shim relies on `cursor-agent login` instead.

The shim also removes `CURSOR_API_KEY` from the child `cursor-agent` environment
so a stale shell variable cannot override your CLI OAuth login.

### Current limitations

- The bridge is prompt-based because `cursor-agent --print` is a CLI interface,
  not a native OpenAI/Anthropic provider endpoint.
- Image inputs are described/omitted in the prompt bridge rather than forwarded
  as a native multimodal API payload.
- Tool-call fidelity is lower than native ChatGPT/Codex passthrough or BYOK
  providers that support structured tool calls directly.

### Troubleshooting

- Run `cursor-agent status` first. If it says you are not logged in, run
  `cursor-agent login`.
- Run `codex-shim doctor` and check the `Cursor passthrough` section.
- Check `/health`; `cursor_passthrough: true` means the daemon can expose
  Composer.
- If the daemon was already running when you logged in, restart it so discovery
  endpoints and generated catalog/config are refreshed.
- If `CURSOR_AGENT_BIN` is set, verify it points to an executable
  `cursor-agent` binary.

---

## Claude Code CLI passthrough

Claude Code passthrough uses a separately launched headless CLI request, not
an attachment to an existing interactive session or the Claude desktop app.
The intended public aliases are:

- `cd-opus-5-5-medium`: Opus 5.5, medium effort.
- `cd-fable-5-1-medium`: Fable 5.1, medium effort.
- `cd-fable-high`: Fable 5.1, high effort.
- `cd-fable-5-medium`: Fable 5, medium effort.

Availability depends on an installed, authenticated Claude Code CLI and the
models available to that account. A running Claude desktop app does not prove
CLI availability or authentication. Do not extract desktop credentials to
make this transport work.

Set `CLAUDE_CODE_BIN` to the real CLI executable when it is not on PATH.
For a native Windows CLI invoked from WSL, use its `/mnt/c/.../claude.exe`
path, not the desktop app executable. An explicitly selected profile uses
`CODEX_SHIM_CLAUDE_CONFIG_DIR`; the CLI remains responsible for authentication
and refresh. Verify the selected CLI's login before enabling it in the daemon.
Interactive sessions must never be resumed or reused by API requests.

The initial adapter supports Chat Completions and Responses with explicit
text history only. Anthropic Messages, Responses compaction, images,
structured-output formats, implicit prior-response history, and client tools
are unsupported. It disables local tools and session persistence, and rejects
client tool requests instead of silently executing commands on the shim host
or claiming transparent tool compatibility. No Claude Jev IO profiles are added.

`CODEX_SHIM_DISABLE_CLAUDE=1` disables discovery and routing. To correct model
IDs for your account, set `CODEX_SHIM_CLAUDE_MODEL_OVERRIDES` to a JSON object
mapping public aliases to verified upstream IDs. Effort stays fixed by alias.
For a Windows executable, `CODEX_SHIM_CLAUDE_CONFIG_DIR` must be a native
Windows path. Authentication is probed with `auth status --json`; aliases are
advertised only after a subscription login is detected. This probe does not
verify model entitlement. Requests have a 180-second timeout. The isolated
working directory and disabled tools are not an OS sandbox.
Model identifiers must be checked against the installed CLI/account before
advertising a deployment as verified.

### Cache efficiency and call cost

Claude requests use a stable, private `~/.codex-shim/claude-workspace`
working directory. Random per-request paths can change the CLI-generated
prompt prefix and defeat provider prompt caching. Tools, project settings,
hooks, and session persistence remain disabled; a stable cwd does not mean
sharing conversation state or resuming sessions.

A small Opus 5.5 repeat-prompt check on 2026-09-24 measured zero cache reads
with random working directories. With the stable directory, the first call
wrote 2,248 cache tokens and the second read 2,248 with no new cache writes.
This is evidence for that prompt/account, not a guaranteed hit rate or billing
savings estimate. Cache eligibility, prefix stability, model, expiration, and
provider policy still matter. Output tokens are not made free by input caching.

For minimum call amplification, prefer a plain model alias. `cx-autogrok`
runs two candidates plus adjudication; Jev IO can add relevance judging,
rewriting, and recall rounds. These modes may save context tokens but are not
a guaranteed cost reduction. No provider billing limit is enforced by this
shim; use account-side spending controls where available.

Usage reporting is not a billing ledger: ensemble responses currently report
zero usage rather than aggregating candidate/adjudicator calls, and auxiliary
Jev/router calls are not guaranteed to be included in the returned model usage.
Failures can consume tokens before usage is returned. Client or CLI retries
can also repeat work. Consult provider usage for actual account consumption;
never interpret a zero or missing usage field as a free request.

### Cursor clients and other subscription backends

There are two different Cursor roles: Cursor can consume this shim through its
OpenAI-compatible Chat Completions endpoint, or the shim can spawn Cursor CLI
for a `cx-*` upstream. Validate both paths; a working cache on one does not
prove the other preserves cache metadata or usage.

Chat/Responses translation preserves explicit `prompt_cache_key` and
`prompt_cache_retention` in both directions, including Cursor BYOK requests
forwarded to Codex. The shim does not generate cache keys or set a retention
default; the selected upstream must support any caller-supplied controls.
Streaming and nonstreaming regression tests cover forwarding and cache usage,
and verify that transport request IDs do not change the upstream body.
Cursor CLI uses the same absolute workspace for its argument and process cwd;
its reported cache counters and native usage details are preserved.
These checks establish shim behavior, not guaranteed provider cache hits.

A two-request read-only Cursor Auto probe on 2026-09-24 reported 1,152 then
4,352 cache-read tokens. Each request included roughly 16.9K total input tokens
including CLI context despite a tiny user prompt. Avoid comparing only user
prompt length when evaluating CLI overhead. Auto may select different backing
models; changing models, context, instructions, or tools can reduce prefix reuse.
Keep conversation prefixes stable and append new turns rather than rebuilding
prior history with timestamps or random identifiers. Do not strip meaningful
instructions or tools just to increase cache hit rate.

### UltraCode scope

[claude-shim](https://github.com/petr-korobeinikov/claude-shim) provides useful
profile-selection patterns using `CLAUDE_CONFIG_DIR`; it is not an inference
server. [UltraCode-Shim](https://github.com/OnlyTerp/UltraCode-Shim) routes and
translates requests while Claude Code supplies the workflow tool and agent
loop. Adding `cd-*` aliases does not add that workflow runtime to an OpenAI
client. UltraCode compatibility remains a separate milestone requiring
structured tool round-trip tests and explicit worker-routing support; it is
not enabled by this text-only transport.

---

## Security and privacy notes

- The generated catalog does not contain ChatGPT tokens or Cursor OAuth tokens.
- `codex-shim doctor` reports auth availability and paths, but does not print
  token contents.
- ChatGPT passthrough reads `~/.codex/auth.json` at request time and forwards
  the access token only to ChatGPT's Codex backend.
- Cursor passthrough spawns `cursor-agent` locally and sends the constructed
  prompt to that CLI process through stdin.
- Do not share `~/.codex/auth.json`, shell history containing tokens, or any
  request dump/log that may contain private prompts.

---

## Subscription passthrough vs BYOK models

| Flow | Credential source | Slug examples | Upstream shape |
|---|---|---|---|
| ChatGPT/Codex passthrough | `codex login` / `~/.codex/auth.json` | `gpt-5.5` | Native Codex Responses backend |
| Cursor model passthrough | `cursor-agent login` | `cx-auto`, `cx-fable-5-1`, `cx-grok-4-7` | `cursor-agent --print` bridge |
| BYOK OpenAI-compatible | `api_key` or `api_key_env` in settings | your configured slug | `/chat/completions` |
| BYOK Anthropic-compatible | `api_key` or `api_key_env` in settings | your configured slug | `/messages` |

Use subscription passthrough when you want to spend subscription quota through
the local authenticated CLI. Use BYOK models when you want to route to a provider
endpoint and API key you control directly.

---

## Reverse BYOK: Codex models as a provider

When ChatGPT/Codex (or Cursor) passthrough is available, the same slugs also
work on the shim's OpenAI chat and Anthropic Messages surfaces:

```text
POST http://127.0.0.1:8765/v1/chat/completions   # OpenAI chat clients
POST http://127.0.0.1:8765/v1/messages            # Anthropic Messages clients
```

Point another Codex profile (or any chat/Messages client) at the shim with a
dummy API key and select `gpt-5.5` / `composer-2-5`. The shim translates:

```text
chat / Messages  →  Responses  →  ChatGPT Codex or cursor-agent
                 ←  chat / Messages reply
```

Tools, images, reasoning, and streaming are preserved on the reverse path the
same way the forward BYOK path preserves them for Codex Desktop.
