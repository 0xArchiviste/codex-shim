"""Host-header allowlist for the loopback shim server.

Binding to 127.0.0.1 keeps the shim off the network, but it does not stop a
web page the user is already viewing from reaching it. A browser resolves an
attacker-controlled domain to 127.0.0.1 (DNS rebinding) and then issues
same-origin requests to the shim; the loopback socket happily accepts them.
Because the shim forwards each request upstream with the user's BYOK API keys
or ChatGPT access token, an unguarded shim lets any visited page spend the
user's model credits and read the responses.

The browser still sends the attacker's domain in the ``Host`` header, so a
Host allowlist is the standard defense: accept loopback names plus whatever
bind host the operator configured, reject everything else.
"""

from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import urlparse

from aiohttp import web

DEFAULT_ALLOWED_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})
ALLOWED_HOSTS_ENV = "CODEX_SHIM_ALLOWED_HOSTS"
PUBLIC_BASE_URL_ENV = "CODEX_SHIM_PUBLIC_BASE_URL"
PUBLIC_BASE_URL_FILE = Path.home() / ".codex-shim" / "public-base-url"
# Kept in sync with the systemd unit / reverse-byok.env so CLI restarts
# still accept the ngrok tunnel Host header.
DEFAULT_TUNNEL_HOST_PATTERNS = (
    "*.ngrok-free.app",
    "*.ngrok-free.dev",
    "*.ngrok.app",
    "*.ngrok.io",
    "*.ngrok.dev",
)
_WILDCARD_BINDS = frozenset({"", "0.0.0.0", "::"})


def host_only(host_header: str) -> str:
    """Return the hostname from a ``Host`` header value, port stripped."""
    value = (host_header or "").strip()
    if not value:
        return ""
    if value.startswith("["):  # bracketed IPv6 literal, e.g. [::1]:8765
        end = value.find("]")
        if end != -1:
            return value[1:end]
        return value
    if value.count(":") == 1:  # host:port (bare IPv6 has 2+ colons)
        return value.rsplit(":", 1)[0]
    return value


def _hostname_from_url(value: str) -> str:
    raw = (value or "").strip()
    if not raw:
        return ""
    if "://" not in raw:
        raw = "https://" + raw
    try:
        return (urlparse(raw).hostname or "").strip().lower()
    except ValueError:
        return ""


def public_base_url() -> str:
    """Configured public base URL from env, else ``~/.codex-shim/public-base-url``."""
    env = os.environ.get(PUBLIC_BASE_URL_ENV, "").strip()
    if env:
        return env
    try:
        return PUBLIC_BASE_URL_FILE.read_text().strip()
    except OSError:
        return ""


def build_allowed_hosts(bind_host: str) -> set[str]:
    """Loopback names + bind host + env allowlist + public tunnel hosts."""
    allowed = {host.lower() for host in DEFAULT_ALLOWED_HOSTS}
    bind = (bind_host or "").strip().lower()
    if bind and bind not in _WILDCARD_BINDS:
        allowed.add(bind)
    for part in os.environ.get(ALLOWED_HOSTS_ENV, "").split(","):
        part = part.strip().lower()
        if part:
            allowed.add(part)

    public = public_base_url()
    public_host = _hostname_from_url(public)
    if public_host:
        allowed.add(public_host)
        # Tunnel hostnames rotate; keep the standard ngrok patterns available
        # whenever a public base URL is configured so CLI restarts do not
        # accidentally lock the tunnel out.
        for pattern in DEFAULT_TUNNEL_HOST_PATTERNS:
            allowed.add(pattern.lower())
    return allowed


def host_matches(hostname: str, allowed_hosts: set[str]) -> bool:
    """Exact allowlist match, plus ``*.example.com`` suffix patterns from env."""
    host = (hostname or "").lower()
    if host in allowed_hosts:
        return True
    for pattern in allowed_hosts:
        if pattern.startswith("*.") and len(pattern) > 2:
            suffix = pattern[1:]  # .example.com
            if host.endswith(suffix) or host == pattern[2:]:
                return True
    return False


def host_guard_middleware(allowed_hosts: set[str]):
    """aiohttp middleware that rejects requests with a non-allowlisted Host."""
    allowed = {host.lower() for host in allowed_hosts}

    @web.middleware
    async def _guard(request: web.Request, handler):
        hostname = host_only(request.headers.get("Host", "")).lower()
        if not host_matches(hostname, allowed):
            print(f"[host] reject Host={hostname!r} path={request.path}", flush=True)
            raise web.HTTPForbidden(
                text='{"error":{"message":"Forbidden: Host header not allowed","type":"invalid_request_error"}}',
                content_type="application/json",
            )
        return await handler(request)

    return _guard
