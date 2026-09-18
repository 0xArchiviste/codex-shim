# ChatGPT Web compatibility (parked) — `csb-*`

Status: **earmarked for a later pass**. Extract best ideas from
[miuuyy/codex-chatgpt-web](https://github.com/miuuyy/codex-chatgpt-web) so this
shim can optionally expose **ChatGPT Web / browser quota** models without
colliding with the existing **Codex backend** passthrough (`cs-*`).

Upstream is unofficial browser automation + a local Responses bridge; UI drift
can break it. Treat anything here as a compatibility design note, not a promise
to reimplement their launcher.

## Naming: `cs-*` vs `csb-*`

| Prefix | Backend | Quota | Today in this repo |
| --- | --- | --- | --- |
| `cs-*` | ChatGPT **Codex** API (`chatgpt.com/backend-api/codex/responses`) via `codex login` | Codex / Work | Yes — e.g. `cs-sol-medium`, `cs-astra-light` |
| `csb-*` | ChatGPT **Web** (browser / Temporary Chat path) | ChatGPT Web message limits | **Not implemented** — reserved |

Do **not** overload `cs-sol-medium` for Web. Use a parallel namespace, e.g.:

| Proposed shim slug | Maps from upstream `chatgpt-web/…` | Intent |
| --- | --- | --- |
| `csb-sol-instant` / `csb-sol-light` | `chatgpt-web/light` | Instant / low effort |
| `csb-sol-medium` | `chatgpt-web/medium` | Medium |
| `csb-sol-high` | `chatgpt-web/high` | High |
| `csb-sol-xhigh` | `chatgpt-web/extra-high` | Extra High (account-gated) |
| `csb-sol-pro` | `chatgpt-web/pro` | Pro / max (account-gated) |
| `csb-luna` / `csb-luna-think` | `chatgpt-web/luna`, `chatgpt-web/think` | Luna-only accounts |
| `csb-zero-risk` | `chatgpt-web/zero-risk` | Manual paste / send mode |

Optional later aliases may accept upstream `chatgpt-web/medium` and rewrite to
`csb-sol-medium`, but catalog rows should advertise `csb-*` only.

## What they already have (best bits)

Source: [architecture](https://github.com/miuuyy/codex-chatgpt-web/blob/main/docs/architecture.md),
[chatgpt-web-models.ts](https://github.com/miuuyy/codex-chatgpt-web/blob/main/src/chatgpt-web-models.ts),
[README](https://github.com/miuuyy/codex-chatgpt-web).

1. **Separate product surface from Codex quota**  
   Web models use ChatGPT Web allowances (Sol Pro / Astra Web, etc.), not the
   Codex Work quota our `cs-*` path spends.

2. **Fixed effort per picker row**  
   Each routed model binds one immutable ChatGPT browser effort
   (`low`/`medium`/`high`/`xhigh`/`max`). Codex Effort UI does not retarget the
   Web mode. Mirror that with one `csb-*` slug per effort.

3. **Account capability probing**  
   Catalog only exposes Pro / Extra High / Luna vs Sol rows the signed-in
   account actually has. Fail closed rather than advertising dead slugs.

4. **Measured context windows (not “infinite”)**  
   Documented Plus Medium/High ≈ 90k (up to ~270k with experimental 3× context);
   Instant smaller; Pro and Luna have their own measured ceilings and composer
   char limits. Usage via GPT-5 tokenizer + platform/image reserves — copy the
   *idea* of adapter-owned limits into catalog metadata.

5. **Three interaction modes**  
   - **Browser-only** — Responses → browser Temporary Chat; no local tools  
   - **Full harness** — ChatGPT tool calls → local Codex tools via MCP +
     outbound OpenAI `tunnel-client` (`Codex Native2` connector)  
   - **Zero Risk** — user pastes/sends; MCP still available; no DOM automation  

6. **Loopback Responses daemon**  
   Same high-level shape as this shim: Codex → local Responses → backend.
   They keep Codex’s `openai` provider and set `openai_base_url` to their
   daemon; we keep `codex_shim` provider. Compatibility can be either:
   - **sidecar proxy**: shim `csb-*` → their daemon base URL, or  
   - **native adapter**: reimplement a subset (hard; browser ownership is theirs).

7. **Compaction & Bigger Context**  
   Browser-bound compaction / multipart staging before the composer ceiling;
   native Codex compaction still supported. Useful when wiring long tasks.

8. **Security posture worth copying**  
   Loopback-only; `0600` browser/tunnel secrets; lifecycle endpoints bearer-auth;
   fail closed on unexpected approvals; no silent model/transport fallback when
   UI selectors drift; drain active turns before restart/uninstall.

9. **Subagent protocol note**  
   New installs default to **Compatibility V1** so native and Web-routed
   backends can interoperate; Native/V2 has encrypted-payload edge cases.
   If we ever proxy `csb-*` beside `cs-*`, pick an explicit protocol mode.

## What we should *not* reimplement first

- Full Electron launcher / Playwright DOM automation  
- Their installers, tunnel packaging, or ChatGPT Developer Mode UI  
- Hijacking `cs-*` or bare `gpt-5.6-sol` for Web traffic  

Prefer: detect a healthy local `codex-chatgpt-web` Responses endpoint (or a
user-configured `CHATGPT_WEB_BASE_URL`) and route only `csb-*` there.

## Suggested later interface (this repo)

```text
Codex / Cursor / clients
        │
        ▼
   codex-shim (:8766)
        ├─ cs-* / gpt-*     → existing Codex backend passthrough
        ├─ cs-grok-* / BYOK → existing providers
        ├─ cs-grokvastra    → ensemble (parked judge follow-up)
        └─ csb-*            → ChatGPT Web bridge (external daemon or adapter)
                 │
                 ▼
        codex-chatgpt-web Responses daemon (optional install)
                 │
                 ▼
        ChatGPT Web (browser / MCP harness)
```

Config sketch (not implemented):

```json
"chatgpt_web": {
  "enabled": true,
  "base_url": "http://127.0.0.1:<their-port>/v1",
  "prefix": "csb-",
  "routes": {
    "csb-sol-medium": "chatgpt-web/medium",
    "csb-sol-high": "chatgpt-web/high",
    "csb-sol-pro": "chatgpt-web/pro"
  }
}
```

## Relationship to current ChatGPT passthrough

Our existing path (`docs/subscription-integration.md`) is **Codex API** auth
from `~/.codex/auth.json`. ChatGPT Web is a **different** credential and quota
surface (browser session / their launcher profile). Both can coexist if slug
namespaces stay disjoint (`cs-*` vs `csb-*`).

## Do not expand until then

- No `csb-*` catalog rows or server routes without revisiting this doc.
- No renaming of existing `cs-sol-*` / `cs-astra-*` aliases.
- When work resumes: start with **sidecar proxy to their daemon** + capability
  gating + immutable effort per slug; only then consider tighter integration.
