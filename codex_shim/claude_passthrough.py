"""Text-only Claude Code subscription bridge; never reads credential files.

Alias targets are proposed identifiers, not a claim of account entitlement.
CODEX_SHIM_CLAUDE_MODEL_OVERRIDES is a JSON alias -> upstream ID mapping.
cd-fable-high intentionally means Fable 5.1 with high effort.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
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


_TOOL_FENCE = "codex-shim-tool"


def build_claude_prompt(body: dict[str, Any]) -> str:
    """Render text and client tools. Claude's own tools stay disabled."""
    if body.get("response_format") or (body.get("text") or {}).get("format"):
        raise ValueError("Claude Code bridge does not support structured output formats")
    if body.get("previous_response_id"):
        raise ValueError("Claude Code bridge requires explicit conversation history")
    def validate(value: Any) -> None:
        if isinstance(value, dict):
            kind = value.get("type", "")
            if kind in {"input_image", "image_url", "image"}:
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
    tools = body.get("tools") or []
    if tools:
        from .claude_tools import render_claude_tool_definitions
        choice = body.get("tool_choice") or chat.get("tool_choice") or "auto"
        sections.append(
            "[AVAILABLE CLIENT TOOLS]\n"
            + render_claude_tool_definitions(body)
            + f"\n\nTool choice: {json.dumps(choice, sort_keys=True)}\n"
            + "Do not execute tools yourself. When a client tool is required, reply with only this fence:\n"
            + f"```{_TOOL_FENCE}\n"
            + '{"tool_calls":[{"name":"tool_name","arguments":{}}]}\n'
            + "```\nOtherwise answer in normal text and do not emit that fence."
        )
    for message in chat.get("messages", []):
        content = message.get("content") or ""
        if isinstance(content, list):
            content = "\n".join(str(part.get("text") or "") for part in content if isinstance(part, dict))
        calls = message.get("tool_calls") or []
        rendered_calls = []
        for call in calls:
            fn = call.get("function") or {}
            rendered_calls.append(json.dumps({"call_id": call.get("id"), "name": fn.get("name"), "arguments": fn.get("arguments") or "{}"}, ensure_ascii=False))
        if rendered_calls:
            content = f"{content}\nTool calls: {', '.join(rendered_calls)}".strip()
        if content or message.get("role") == "tool":
            if not content:
                content = "[empty tool result]"
            label = message.get("role", "user").upper()
            if label == "TOOL":
                label = f"TOOL {message.get('tool_call_id', '')}"
            sections.append(f"[{label}]\n{content}")
    return "\n\n".join(sections) or "Continue."


def parse_claude_tool_calls(text: str, allowed_names: set[str]) -> list[dict[str, str]]:
    """Extract one complete tool reply atomically; invalid replies stay text."""
    match = re.fullmatch(rf"\s*```{_TOOL_FENCE}\s*(\{{.*\}})\s*```\s*", text, re.DOTALL)
    if not match:
        return []
    try:
        payload = json.loads(match.group(1))
        rows = payload.get("tool_calls")
    except (json.JSONDecodeError, AttributeError):
        return []
    calls = []
    if isinstance(rows, list):
        for row in rows:
            if not isinstance(row, dict) or not isinstance(row.get("name"), str) or row["name"] not in allowed_names:
                return []
            arguments = row.get("arguments", {})
            if not isinstance(arguments, dict):
                return []
            calls.append({"name": row["name"], "arguments": json.dumps(arguments, sort_keys=True)})
    return calls


def permission_control_response(message: dict[str, Any]) -> dict[str, Any] | None:
    """Answer a host permission ask with an explicit allow. Other control requests are rejected."""
    if message.get("type") != "control_request":
        return None
    request_id = message.get("request_id")
    request = message.get("request")
    if not isinstance(request_id, str) or not request_id or not isinstance(request, dict):
        return None
    if request.get("subtype") == "can_use_tool":
        updated = request.get("input")
        if not isinstance(updated, dict):
            updated = {}
        return {"type": "control_response", "response": {"subtype": "success", "request_id": request_id, "response": {"behavior": "allow", "updatedInput": updated}}}
    return {"type": "control_response", "response": {"subtype": "error", "request_id": request_id, "error": "Unsupported control request"}}


def claude_tool_names(body: dict[str, Any]) -> set[str]:
    names = set()
    for tool in body.get("tools") or []:
        if not isinstance(tool, dict):
            continue
        fn = tool.get("function") if isinstance(tool.get("function"), dict) else tool
        if isinstance(fn, dict) and isinstance(fn.get("name"), str):
            names.add(fn["name"])
    return names


def _claude_failure(data: dict[str, Any]) -> tuple[str, str]:
    """Classify diagnostics locally; never echo upstream text or arbitrary subtypes."""
    diagnostic = json.dumps({key: data.get(key) for key in
                             ("error", "errors", "result", "subtype")}).lower()
    if data.get("subtype") == "error_max_output_tokens":
        return "claude_max_tokens", "Claude Code reached the output token limit"
    if any(term in diagnostic for term in (
        "rate_limit", "rate limit", "usage limit", "quota", "credit balance",
        "insufficient_quota", "out of credits", "hit your limit",
    )):
        return "claude_quota_exceeded", "Claude Code reported a quota or rate limit"
    if any(term in diagnostic for term in ("timeout", "timed out", "time out")):
        return "claude_timeout", "Claude Code reported an upstream timeout"
    return "claude_upstream_error", "Claude Code returned an upstream error"


class ClaudeStreamParser:
    def __init__(self) -> None:
        self.text = ""
        self.partial = False
        self.result_seen = False
        self.usage: dict[str, Any] | None = None
        self.error: str | None = None
        self.error_code = "claude_upstream_error"
        self.stop_reason: str | None = None

    def _capture_metadata(self, data: dict[str, Any]) -> None:
        usage = data.get("usage")
        if isinstance(usage, dict):
            # Snapshots, not increments: stream deltas report cumulative output.
            self.usage = {**(self.usage or {}), **usage}
        reason = data.get("stop_reason")
        if isinstance(reason, str):
            self.stop_reason = reason

    def feed_line(self, line: str) -> str:
        data = json.loads(line)
        kind = data.get("type")
        delta = ""
        if kind == "stream_event":
            event = data.get("event", {})
            if event.get("type") == "message_start":
                self._capture_metadata(event.get("message", {}))
            elif event.get("type") == "message_delta":
                self._capture_metadata(event)
                self._capture_metadata(event.get("delta", {}))
            if event.get("type") == "content_block_delta" and event.get("delta", {}).get("type") == "text_delta":
                self.partial = True
                delta = event["delta"].get("text", "")
        elif kind == "assistant":
            self._capture_metadata(data.get("message", {}))
            if data.get("error"):
                self.error_code, self.error = _claude_failure(data)
                return ""
            blocks = data.get("message", {}).get("content", [])
            if any(block.get("type") == "tool_use" for block in blocks):
                self.error = "Claude Code unexpectedly requested a tool; bridge is text-only"
            if not self.partial:
                delta = "".join(block.get("text", "") for block in blocks if block.get("type") == "text")
        elif kind == "result":
            self.result_seen = True
            self._capture_metadata(data)
            if data.get("is_error") or data.get("subtype") not in (None, "success"):
                if self.stop_reason == "max_tokens":
                    self.error_code, self.error = "claude_max_tokens", "Claude Code reached the output token limit"
                else:
                    code, message = _claude_failure(data)
                    if not self.error or code != "claude_upstream_error":
                        self.error_code, self.error = code, message
            elif not self.text and not self.error:
                delta = data.get("result", "")
        elif kind == "error":
            self._capture_metadata(data)
            self.error_code, self.error = _claude_failure(data)
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
    reader = None
    parser = ClaudeStreamParser()
    emitted_usage = None
    with claude_request_workspace() as cwd:
        try:
            argv = [claude_bin(), "--print", "--output-format", "stream-json", "--verbose",
                    "--include-partial-messages", "--no-session-persistence", "--disable-slash-commands", "--tools", "",
                    "--input-format", "stream-json", "--permission-prompts", "host",
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
                events: asyncio.Queue = asyncio.Queue()
                init_future = asyncio.get_running_loop().create_future()

                async def read_stdout() -> None:
                    nonlocal emitted_usage
                    try:
                        while raw := await proc.stdout.readline():
                            if not raw.strip():
                                continue
                            try:
                                incoming = json.loads(raw)
                            except json.JSONDecodeError:
                                incoming = None
                            if isinstance(incoming, dict) and incoming.get("type") == "control_response":
                                response = incoming.get("response")
                                if isinstance(response, dict) and response.get("request_id") == "req_init" and not init_future.done():
                                    init_future.set_result(response.get("subtype") == "success")
                                continue
                            if isinstance(incoming, dict):
                                reply = permission_control_response(incoming)
                                if reply is not None:
                                    if reply["response"].get("subtype") == "success":
                                        print("[claude] allowed host permission request", flush=True)
                                    proc.stdin.write((json.dumps(reply, ensure_ascii=False) + "\n").encode())
                                    await proc.stdin.drain()
                                    continue
                            delta = parser.feed_line(raw.decode("utf-8"))
                            if parser.usage is not None and parser.usage != emitted_usage:
                                emitted_usage = dict(parser.usage)
                                await events.put({"type": "usage", "usage": emitted_usage})
                            if parser.error and parser.result_seen:
                                await events.put({"type": "error", "code": parser.error_code, "message": parser.error})
                                return
                            if delta and not parser.error:
                                await events.put({"type": "text_delta", "delta": delta})
                            # The CLI keeps stdout open until stdin closes, but only after the result.
                            if parser.result_seen:
                                proc.stdin.close()
                        await events.put({"type": "_eof"})
                    except asyncio.CancelledError:
                        raise
                    except TimeoutError:
                        await events.put({"type": "error", "code": "claude_timeout", "message": "Claude Code request timed out"})
                        await events.put({"type": "_eof"})
                    except (OSError, ValueError, UnicodeError, TypeError, AttributeError):
                        await events.put({"type": "error", "code": "claude_process_error", "message": "Unable to execute Claude Code or parse its output"})
                        await events.put({"type": "_eof"})
                    finally:
                        if not init_future.done():
                            init_future.set_result(False)

                reader = asyncio.create_task(read_stdout())
                # The CLI accepts permission replies only after this handshake, with stdin left open.
                initialize = {"type": "control_request", "request_id": "req_init", "request": {"subtype": "initialize", "hooks": None}}
                proc.stdin.write((json.dumps(initialize) + "\n").encode())
                await proc.stdin.drain()
                try:
                    initialized = await asyncio.wait_for(init_future, 30)
                except TimeoutError:
                    initialized = False
                if not initialized:
                    yield {"type": "error", "code": "claude_process_error", "message": "Claude Code control handshake failed"}
                    return
                user_message = {"type": "user", "session_id": "", "message": {"role": "user", "content": prompt}, "parent_tool_use_id": None}
                proc.stdin.write((json.dumps(user_message, ensure_ascii=False) + "\n").encode())
                await proc.stdin.drain()
                while True:
                    event = await events.get()
                    if event["type"] == "_eof":
                        break
                    yield event
                    if event["type"] == "error":
                        return
                proc.stdin.close()
                code = await proc.wait()
                if parser.error:
                    yield {"type": "error", "code": parser.error_code, "message": parser.error}
                    return
                if code or not parser.result_seen:
                    yield {"type": "error", "code": "claude_process_error", "message": "Claude Code exited unsuccessfully or without a final result"}
                    return
                if parser.stop_reason == "max_tokens":
                    # Fail closed: the server has no incomplete-completion contract.
                    yield {"type": "error", "code": "claude_max_tokens", "message": "Claude Code reached the output token limit", "stop_reason": "max_tokens"}
                    return
                yield {"type": "completed", "text": parser.text}
        except TimeoutError:
            yield {"type": "error", "code": "claude_timeout", "message": "Claude Code request timed out"}
        except (OSError, ValueError, UnicodeError, TypeError, AttributeError):
            yield {"type": "error", "code": "claude_process_error", "message": "Unable to execute Claude Code or parse its output"}
        finally:
            if proc is not None and proc.stdin is not None:
                try:
                    proc.stdin.close()
                except (AttributeError, RuntimeError, OSError):
                    pass
            if proc is not None and proc.returncode is None:
                try:
                    proc.kill()
                except ProcessLookupError:
                    pass
                await proc.wait()
            if reader is not None:
                reader.cancel()
                await asyncio.gather(reader, return_exceptions=True)
            if stderr_task is not None:
                stderr_task.cancel()
                await asyncio.gather(stderr_task, return_exceptions=True)
