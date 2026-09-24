from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest
from aiohttp.test_utils import TestClient, TestServer

from codex_shim import claude_passthrough as claude
from codex_shim import server


@pytest.fixture
def claude_enabled(monkeypatch):
    monkeypatch.delenv("CODEX_SHIM_DISABLE_CLAUDE", raising=False)
    monkeypatch.setattr(claude, "_auth_probe_cache", None)


def test_workspace_is_stable_private_and_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(claude.Path, "home", lambda: tmp_path)
    with claude.claude_request_workspace() as first:
        pass
    with claude.claude_request_workspace() as second:
        assert first == second
        assert claude.Path(second).stat().st_mode & 0o777 == 0o700
        assert list(claude.Path(second).iterdir()) == []


def test_workspace_rejects_symlink(tmp_path, monkeypatch):
    monkeypatch.setattr(claude.Path, "home", lambda: tmp_path)
    parent = tmp_path / ".codex-shim"
    parent.mkdir()
    (parent / "claude-workspace").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(OSError):
        with claude.claude_request_workspace():
            pass


def test_auth_cache_disable_and_config(monkeypatch, claude_enabled):
    calls = []
    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        return SimpleNamespace(returncode=0, stdout='{"loggedIn":true,"authMethod":"claude.ai"}')
    monkeypatch.setattr(claude.subprocess, "run", run)
    monkeypatch.setenv("CLAUDE_CODE_BIN", "/mnt/c/Claude/claude.exe")
    assert claude.claude_passthrough_available()
    assert claude.claude_passthrough_available()
    assert len(calls) == 1
    assert calls[0][0] == ["/mnt/c/Claude/claude.exe", "auth", "status", "--json"]
    monkeypatch.setenv("CODEX_SHIM_CLAUDE_CONFIG_DIR", "C:\\trusted")
    assert claude.claude_passthrough_available()
    assert len(calls) == 2
    monkeypatch.setenv("CODEX_SHIM_DISABLE_CLAUDE", "1")
    assert not claude.claude_passthrough_available()


@pytest.mark.parametrize("status", ['{}', '{"loggedIn":false}', '{"loggedIn":true,"authMethod":"api_key"}', 'not json'])
def test_auth_unavailable(monkeypatch, claude_enabled, status):
    monkeypatch.setattr(claude.subprocess, "run", lambda *a, **kw: SimpleNamespace(returncode=0, stdout=status))
    assert not claude.claude_passthrough_available()


def test_environment(monkeypatch):
    for key in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL", "CLAUDECODE"):
        monkeypatch.setenv(key, "stub")
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", "untrusted")
    monkeypatch.setenv("WSLENV", "ANTHROPIC_API_KEY:CLAUDECODE:KEEP")
    env = claude.claude_spawn_env()
    assert not any(key.startswith("ANTHROPIC_") for key in env)
    assert "CLAUDECODE" not in env and "CLAUDE_CONFIG_DIR" not in env
    assert "ANTHROPIC" not in env["WSLENV"]


def test_aliases_overrides(monkeypatch):
    assert claude.CLAUDE_MODEL_ALIASES["cd-fable-high"][:2] == ("claude-fable-5-1", "high")
    monkeypatch.setenv("CODEX_SHIM_CLAUDE_MODEL_OVERRIDES", '{"cd-fable-high":"verified-model"}')
    assert claude.claude_upstream_model("cd-fable-high") == "verified-model"


@pytest.mark.parametrize("body", [
    {"tools": [{"type": "function"}]}, {"tool_choice": "auto"},
    {"input": [{"type": "function_call", "name": "foo"}]},
    {"messages": [{"role": "tool", "content": "result"}]},
    {"input": [{"role": "user", "content": [{"type": "input_image"}]}]},
])
def test_unsupported_requests(body):
    with pytest.raises(ValueError):
        claude.build_claude_prompt(body)


def test_parser_deduplicates():
    parser = claude.ClaudeStreamParser()
    assert parser.feed_line(json.dumps({"type": "stream_event", "event": {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "Hello"}}})) == "Hello"
    assert parser.feed_line(json.dumps({"type": "assistant", "message": {"content": [{"type": "text", "text": "Hello"}]}})) == ""
    assert parser.feed_line('{"type":"result","subtype":"success","result":"Hello"}') == ""
    assert parser.text == "Hello"


class FakeProc:
    def __init__(self, lines):
        self.returncode = None
        self.killed = False
        self.stdout = asyncio.StreamReader()
        for line in lines:
            self.stdout.feed_data(json.dumps(line).encode() + b"\n")
        self.stdout.feed_eof()
        self.stderr = asyncio.StreamReader()
        self.stderr.feed_eof()
        self.stdin = self
        self.data = None

    def write(self, data):
        self.data = data

    async def drain(self):
        pass

    def close(self):
        pass

    def kill(self):
        self.killed = True
        self.returncode = -9

    async def wait(self):
        if self.returncode is None:
            self.returncode = 0
        return self.returncode


async def test_spawn_flags_stdin_effort_and_cleanup(monkeypatch):
    proc = FakeProc([{"type": "result", "subtype": "success", "result": "Hello"}])
    captured = {}
    async def spawn(*argv, **kwargs):
        captured.update(argv=argv, kwargs=kwargs)
        return proc
    monkeypatch.setattr(claude.asyncio, "create_subprocess_exec", spawn)
    events = [event async for event in claude.iter_claude_agent_events("private prompt", "cd-fable-high")]
    argv = captured["argv"]
    assert argv[argv.index("--tools") + 1] == ""
    assert argv[argv.index("--effort") + 1] == "high"
    assert "--no-session-persistence" in argv and "--strict-mcp-config" in argv
    assert "private prompt" not in argv and proc.data == b"private prompt"
    assert "shell" not in captured["kwargs"]
    assert events[-1] == {"type": "completed", "text": "Hello"}
    assert not proc.killed


async def test_cancellation_cleans_child(monkeypatch):
    proc = FakeProc([])
    proc.stdout = asyncio.StreamReader()  # remain blocked until cancellation
    ready = asyncio.Event()
    async def spawn(*a, **kw):
        ready.set()
        return proc
    monkeypatch.setattr(claude.asyncio, "create_subprocess_exec", spawn)
    async def consume():
        return [event async for event in claude.iter_claude_agent_events("prompt", "cd-fable-high")]
    task = asyncio.create_task(consume())
    await ready.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert proc.killed


def test_catalog_contract():
    entry = claude.claude_catalog_entry("cd-fable-high")
    assert entry["default_reasoning_level"] == "high"
    assert entry["supports_parallel_tool_calls"] is False
    assert entry["input_modalities"] == ["text"]
    assert entry["available_in_plans"] == []


async def test_close_cleans_child(monkeypatch):
    proc = FakeProc([{"type": "assistant", "message": {"content": [{"type": "text", "text": "Hello"}]}}])
    async def spawn(*a, **kw):
        return proc
    monkeypatch.setattr(claude.asyncio, "create_subprocess_exec", spawn)
    events = claude.iter_claude_agent_events("prompt", "cd-fable-high")
    await anext(events)
    await events.aclose()
    assert proc.killed


@pytest.mark.parametrize("lines", [[], [{"type": "result", "is_error": True, "subtype": "error_during_execution", "result": "sensitive"}]])
async def test_process_errors(monkeypatch, lines):
    async def spawn(*a, **kw):
        return FakeProc(lines)
    monkeypatch.setattr(claude.asyncio, "create_subprocess_exec", spawn)
    events = [event async for event in claude.iter_claude_agent_events("prompt", "cd-fable-high")]
    assert events[-1]["type"] == "error"
    assert "sensitive" not in str(events)


@pytest.mark.parametrize("endpoint,field", [("/v1/responses", "input"), ("/v1/chat/completions", "messages")])
@pytest.mark.parametrize("stream", [False, True])
async def test_routing(monkeypatch, tmp_path, endpoint, field, stream):
    monkeypatch.setattr(server, "claude_passthrough_available", lambda: True)
    async def events(prompt, slug):
        assert "hello" in prompt and slug == "cd-fable-high"
        yield {"type": "text_delta", "delta": "answer"}
        yield {"type": "completed", "text": "answer"}
    monkeypatch.setattr(server, "iter_claude_agent_events", events)
    settings = tmp_path / "settings.json"
    settings.write_text('{"models":[]}')
    shim = server.ShimServer(settings)
    async with TestClient(TestServer(shim.app())) as client:
        response = await client.post(endpoint, json={"model": "cd-fable-high", field: [{"role": "user", "content": "hello"}], "stream": stream})
        assert response.status == 200
        text = await response.text()
        assert "answer" in text and "cd-fable-high" in text
        bad = await client.post(endpoint, json={"model": "cd-fable-high", field: [], "tools": [{"type": "function"}]})
        assert bad.status == 400
        models = await (await client.get("/v1/models")).json()
        assert any(row["id"] == "cd-fable-high" for row in models["data"])
        health = await (await client.get("/health")).json()
        assert health["claude_passthrough"] is True


@pytest.mark.parametrize("exception,code", [(FileNotFoundError(), "claude_process_error"), (TimeoutError(), "claude_timeout")])
async def test_spawn_failure(monkeypatch, exception, code):
    async def spawn(*a, **kw):
        raise exception
    monkeypatch.setattr(claude.asyncio, "create_subprocess_exec", spawn)
    events = [event async for event in claude.iter_claude_agent_events("prompt", "cd-fable-high")]
    assert events == [{"type": "error", "code": code, "message": events[0]["message"]}]


async def test_route_auth_and_structured_errors(monkeypatch, tmp_path):
    settings = tmp_path / "settings.json"
    settings.write_text('{"models":[]}')
    shim = server.ShimServer(settings)
    async with TestClient(TestServer(shim.app())) as client:
        monkeypatch.setattr(server, "claude_passthrough_available", lambda: False)
        response = await client.post("/v1/responses", json={"model": "cd-fable-high", "input": "hello"})
        assert response.status == 401
        assert (await response.json())["error"]["code"] == "claude_unavailable"
        monkeypatch.setattr(server, "claude_passthrough_available", lambda: True)
        async def events(*args):
            yield {"type": "error", "code": "claude_upstream_error", "message": "failure"}
        monkeypatch.setattr(server, "iter_claude_agent_events", events)
        for stream in (False, True):
            response = await client.post("/v1/responses", json={"model": "cd-fable-high", "input": "hello", "stream": stream})
            text = await response.text()
            assert "claude_upstream_error" in text
            assert '"status": "completed"' not in text
            assert response.status == (200 if stream else 502)
