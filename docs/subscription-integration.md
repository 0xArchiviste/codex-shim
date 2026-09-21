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
