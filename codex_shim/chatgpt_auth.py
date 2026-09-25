"""Saved Codex login selection and failover for the chatgpt.com passthrough.

The shim only reads saved Codex logins; it never refreshes or writes them.
Refresh tokens rotate, so a second refresher would log out the owner (for
example the Windows Codex app whose auth.json is listed as a fallback).
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable

# os.pathsep-separated auth.json paths tried after the primary login.
FALLBACKS_ENV = "CODEX_SHIM_CHATGPT_AUTH_FALLBACKS"
# Seconds a login is tried last after a non-transient 401.
HARD_FAIL_COOLDOWN = 300.0
TRANSIENT_RETRY_DELAY = 0.25
EXPIRY_SKEW = 60.0

_last_good: str | None = None
_hard_failed: dict[str, float] = {}


@dataclass(frozen=True)
class ChatGPTCredential:
    path: str
    access_token: str
    account_id: str
    expires_at: float | None

    def headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.access_token}", "chatgpt-account-id": self.account_id}


def _jwt_exp(token: str) -> float | None:
    try:
        part = token.split(".")[1]
        part += "=" * (-len(part) % 4)
        exp = json.loads(base64.urlsafe_b64decode(part)).get("exp")
    except (IndexError, ValueError, AttributeError):
        return None
    return float(exp) if isinstance(exp, (int, float)) else None


def auth_paths(primary: Path) -> list[Path]:
    paths = [Path(primary).expanduser()]
    for raw in os.environ.get(FALLBACKS_ENV, "").split(os.pathsep):
        if raw.strip():
            paths.append(Path(raw.strip()).expanduser())
    return paths


def load_credentials(primary: Path) -> list[ChatGPTCredential]:
    """Return usable logins, best first.

    Unexpired logins come first, preferring the last one that worked and
    demoting ones in hard-failure cooldown; expired ones are kept last so the
    upstream still gets the final say.
    """
    now = time.time()
    seen: set[str] = set()
    fresh: list[ChatGPTCredential] = []
    stale: list[ChatGPTCredential] = []
    for path in auth_paths(primary):
        try:
            data = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        tokens = data.get("tokens") if isinstance(data, dict) else None
        token = tokens.get("access_token") if isinstance(tokens, dict) else None
        if not token or token in seen:
            continue
        seen.add(token)
        cred = ChatGPTCredential(str(path), str(token), str(tokens.get("account_id") or ""), _jwt_exp(str(token)))
        expired = cred.expires_at is not None and cred.expires_at < now + EXPIRY_SKEW
        (stale if expired else fresh).append(cred)
    fresh.sort(key=lambda c: (_hard_failed.get(c.path, 0.0) > now, c.path != _last_good))
    return fresh + stale


def is_transient_401(text: str) -> bool:
    # The Codex backend intermittently rejects valid ChatGPT logins with its
    # own internal service-account key (sk-svcac...); retrying succeeds.
    return "invalid_api_key" in text or "sk-svcac" in text


async def post_with_failover(
    post: Callable[..., Awaitable[Any]],
    credentials: list[ChatGPTCredential],
    url: str,
    body: dict[str, Any],
    headers: dict[str, str],
) -> Any:
    """POST with each login in turn until one is not rejected with 401.

    A transient 401 is retried once on the same login before failing over; a
    hard 401 fails over immediately. Returns the last upstream response.
    """
    global _last_good
    upstream = None
    for cred in credentials:
        for attempt in range(2):
            if upstream is not None:
                upstream.release()
            upstream = await post(url, json=body, headers={**headers, **cred.headers()})
            if upstream.status != 401:
                _last_good = cred.path
                _hard_failed.pop(cred.path, None)
                return upstream
            transient = is_transient_401(await upstream.text())
            print(f"[chatgpt] 401 ({'transient' if transient else 'hard'}) with login {cred.path} attempt={attempt + 1}", flush=True)
            if not transient:
                _hard_failed[cred.path] = time.time() + HARD_FAIL_COOLDOWN
                break
            if attempt == 0:
                await asyncio.sleep(TRANSIENT_RETRY_DELAY)
    return upstream
