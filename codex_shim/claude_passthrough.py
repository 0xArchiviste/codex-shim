"""Text-only Claude Code subscription bridge; never reads credential files.

Alias targets are proposed identifiers, not a claim of account entitlement.
CODEX_SHIM_CLAUDE_MODEL_OVERRIDES is a JSON alias -> upstream ID mapping.
cd-fable-high intentionally means Fable 5.1 with high effort.
"""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from collections.abc import AsyncIterator
from typing import Any

from .translate import responses_to_chat

CLAUDE_MODEL_ALIASES = {
    "cd-opus-5-5-medium": ("claude-opus-5-5", "medium", "Claude Code Opus 5.5 Medium"),
    "cd-fable-5-1-medium": ("claude-fable-5-1", "medium", "Claude Code Fable 5.1 Medium"),
    "cd-fable-high": ("claude-fable-5-1", "high", "Claude Code Fable 5.1 High"),
    "cd-fable-5-medium": ("claude-fable-5", "medium", "Claude Code Fable 5 Medium"),
}
_auth_probe_cache: tuple[float, tuple[str, ...], bool] | None = None


def claude_bin() -> str:
    return os.environ.get("CLAUDE_CODE_BIN", "").strip() or shutil.which("claude") or "claude"


def claude_spawn_env() -> dict[str, str]:
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(("ANTHROPIC_", "CLAUDE_CODE_"))
           and key not in {"CLAUDECODE", "CLAUDE_CONFIG_DIR"}}
    config = os.environ.get("CODEX_SHIM_CLAUDE_CONFIG_DIR", "").strip()
    if config:
        env["CLAUDE_CONFIG_DIR"] = config
    # Prevent inherited API-provider switches and nesting, including WSL exports.
    env["CLAUDE_CODE_USE_BEDROCK"] = "0"
    env["CLAUDE_CODE_USE_VERTEX"] = "0"
    env["CLAUDE_CODE_USE_FOUNDRY"] = "0"
    env["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] = "1"
    names = [part for part in env.get("WSLENV", "").split(":")
             if part and not part.split("/")[0].startswith(("ANTHROPIC_", "CLAUDE"))]
    names.extend(key for key in env if key.startswith("CLAUDE_CODE_"))
    if config:
        names.append("CLAUDE_CONFIG_DIR")  # caller supplies native Windows path for .exe
    env["WSLENV"] = ":".join(names)
    return env


def claude_passthrough_available(*, force_refresh: bool = False) -> bool:
    global _auth_probe_cache
    if os.environ.get("CODEX_SHIM_DISABLE_CLAUDE", "").lower() in {"1", "true", "yes", "on"}:
        return False
    key = (claude_bin(), os.environ.get("CODEX_SHIM_CLAUDE_CONFIG_DIR", ""))
    now = time.monotonic()
    if not force_refresh and _auth_probe_cache and _auth_probe_cache[1] == key and now - _auth_probe_cache[0] < 30:
        return _auth_probe_cache[2]
    available = False
    try:
        with tempfile.TemporaryDirectory(prefix="codex-shim-claude-probe-") as cwd:
            result = subprocess.run(
                [key[0], "auth", "status", "--json"], stdin=subprocess.DEVNULL,
                capture_output=True, text=True, timeout=10, cwd=cwd, env=claude_spawn_env(),
            )
        status = json.loads(result.stdout)
        # API-key auth must not masquerade as a subscription login.
        available = (result.returncode == 0 and status.get("loggedIn") is True
                     and status.get("authMethod") in {"claude.ai", "oauth"})
    except (OSError, subprocess.TimeoutExpired, ValueError, AttributeError):
        pass
    _auth_probe_cache = (now, key, available)
    return available


def is_claude_passthrough_slug(slug: str) -> bool:
    return slug in CLAUDE_MODEL_ALIASES


def claude_passthrough_display_names() -> dict[str, str]:
    return {slug: row[2] for slug, row in CLAUDE_MODEL_ALIASES.items()}


def claude_upstream_model(slug: str) -> str:
    overrides = json.loads(os.environ.get("CODEX_SHIM_CLAUDE_MODEL_OVERRIDES", "{}"))
    if not isinstance(overrides, dict):
        raise ValueError("CODEX_SHIM_CLAUDE_MODEL_OVERRIDES must be a JSON object")
    value = overrides.get(slug, CLAUDE_MODEL_ALIASES[slug][0])
    if not isinstance(value, str) or not value.strip() or value.startswith("-"):
        raise ValueError("Invalid Claude model override")
    return value


def claude_catalog_entry(slug: str) -> dict[str, Any]:
    from .cursor_passthrough import cursor_catalog_entry
    entry = cursor_catalog_entry(slug)
    effort = CLAUDE_MODEL_ALIASES[slug][1]
    entry.update(display_name=CLAUDE_MODEL_ALIASES[slug][2],
                 description="Text-only Claude Code login bridge. Proposed model ID; account access is not verified.",
                 default_reasoning_level=effort,
                 supported_reasoning_levels=[{"effort": effort, "description": "Fixed alias effort"}],
                 supports_parallel_tool_calls=False, input_modalities=["text"],
                 supports_image_detail_original=False, available_in_plans=[],
                 base_instructions="You are a helpful assistant.",
                 model_messages={"instructions_template": "You are a helpful assistant.", "instructions_variables": {}},
                 isDefault=False)
    return entry


def build_claude_prompt(body: dict[str, Any]) -> str:
    """Reject unsupported structured I/O rather than silently dropping it."""
    if body.get("tools") or body.get("functions") or body.get("tool_choice") not in (None, "none") or body.get("function_call"):
        raise ValueError("Claude Code bridge does not support client tools or tool choice")
    if body.get("response_format") or (body.get("text") or {}).get("format"):
        raise ValueError("Claude Code bridge does not support structured output formats")
    if body.get("previous_response_id"):
        raise ValueError("Claude Code bridge requires explicit conversation history")
    def validate(value: Any) -> None:
        if isinstance(value, dict):
            if value.get("role") in {"tool", "function"} or value.get("tool_calls") or value.get("function_call"):
                raise ValueError("Claude Code bridge does not support client tool history")
            kind = value.get("type", "")
            if kind and kind not in {"message", "text", "input_text", "output_text"}:
                raise ValueError("Claude Code bridge supports text-only messages")
            for child in value.values():
                validate(child)
        elif isinstance(value, list):
            for child in value:
                validate(child)
    validate(body.get("input"))
    validate(body.get("messages"))
    chat = responses_to_chat(body, str(body.get("model") or ""))
    sections = []
    for message in chat.get("messages", []):
        content = message.get("content", "")
        if isinstance(content, list):
            content = "\n".join(part.get("text", "") for part in content)
        if content:
            sections.append(f"[{str(message.get('role', 'user')).upper()}]\n{content}")
    return "\n\n".join(sections) or "Continue."


class ClaudeStreamParser:
    def __init__(self) -> None:
        self.text = ""
        self.partial = False
        self.result_seen = False
        self.usage: dict[str, Any] | None = None
        self.error: str | None = None

    def feed_line(self, line: str) -> str:
        data = json.loads(line)
        kind = data.get("type")
        delta = ""
        if kind == "stream_event":
            event = data.get("event", {})
            if event.get("type") == "content_block_delta" and event.get("delta", {}).get("type") == "text_delta":
                self.partial = True
                delta = event["delta"].get("text", "")
        elif kind == "assistant":
            blocks = data.get("message", {}).get("content", [])
            if any(block.get("type") == "tool_use" for block in blocks):
                self.error = "Claude Code unexpectedly requested a tool; bridge is text-only"
            if not self.partial:
                delta = "".join(block.get("text", "") for block in blocks if block.get("type") == "text")
        elif kind == "result":
            self.result_seen = True
            self.usage = data.get("usage")
            if data.get("is_error") or data.get("subtype") not in (None, "success"):
                self.error = "Claude Code request failed (" + str(data.get("subtype") or "error") + ")"
            elif not self.text:
                delta = data.get("result", "")
        elif kind == "error":
            self.error = "Claude Code returned an upstream error"
        self.text += delta
        return delta


@contextmanager
def claude_request_workspace():
    """Stable private cwd avoids random paths changing the CLI prompt prefix.

    No conversation is persisted here; request isolation comes from disabled
    tools/settings and separate non-persistent CLI sessions, not a random path.
    """
    path = Path.home() / ".codex-shim" / "claude-workspace"
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    if path.is_symlink() or path.stat().st_uid != os.getuid():
        raise OSError("Claude workspace must be a user-owned directory")
    path.chmod(0o700)
    yield str(path)


async def iter_claude_agent_events(prompt: str, slug: str) -> AsyncIterator[dict[str, Any]]:
    """Run an isolated text-only turn. Closing/cancelling the iterator kills the child."""
    proc = None
    stderr_task = None
    parser = ClaudeStreamParser()
    with claude_request_workspace() as cwd:
        try:
            argv = [claude_bin(), "--print", "--output-format", "stream-json", "--verbose",
                    "--include-partial-messages", "--no-session-persistence", "--disable-slash-commands", "--tools", "",
                    "--strict-mcp-config", "--mcp-config", '{"mcpServers":{}}',
                    "--setting-sources", "", "--settings", '{"disableAllHooks":true}',
                    "--model", claude_upstream_model(slug), "--effort", CLAUDE_MODEL_ALIASES[slug][1]]
            async with asyncio.timeout(180):
                proc = await asyncio.create_subprocess_exec(
                    *argv, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE, cwd=cwd, env=claude_spawn_env(), limit=4 * 1024 * 1024,
                )
                async def drain_stderr() -> None:
                    while await proc.stderr.read(8192):
                        pass  # Never return diagnostics that might contain credentials.
                stderr_task = asyncio.create_task(drain_stderr())
                proc.stdin.write(prompt.encode("utf-8"))
                await proc.stdin.drain()
                proc.stdin.close()
                while raw := await proc.stdout.readline():
                    if not raw.strip():
                        continue
                    delta = parser.feed_line(raw.decode("utf-8"))
                    if parser.error:
                        yield {"type": "error", "code": "claude_upstream_error", "message": parser.error}
                        return
                    if delta:
                        yield {"type": "text_delta", "delta": delta}
                code = await proc.wait()
                if code or not parser.result_seen:
                    yield {"type": "error", "code": "claude_process_error", "message": "Claude Code exited unsuccessfully or without a final result"}
                    return
                if parser.usage:
                    yield {"type": "usage", "usage": parser.usage}
                yield {"type": "completed", "text": parser.text}
        except TimeoutError:
            yield {"type": "error", "code": "claude_timeout", "message": "Claude Code request timed out"}
        except (OSError, ValueError, UnicodeError, TypeError, AttributeError):
            yield {"type": "error", "code": "claude_process_error", "message": "Unable to execute Claude Code or parse its output"}
        finally:
            if proc is not None and proc.returncode is None:
                try:
                    proc.kill()
                except ProcessLookupError:
                    pass
                await proc.wait()
            if stderr_task is not None:
                stderr_task.cancel()
                await asyncio.gather(stderr_task, return_exceptions=True)
