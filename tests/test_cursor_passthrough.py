from __future__ import annotations

import json

import pytest

from codex_shim.cursor_passthrough import (
    CursorStreamParser,
    build_cursor_prompt,
    cursor_passthrough_display_names,
    cursor_upstream_model,
    is_cursor_passthrough_slug,
    iter_cursor_agent_events,
)


def test_is_cursor_passthrough_slug():
    assert is_cursor_passthrough_slug("composer-2-5")
    assert is_cursor_passthrough_slug("composer-2.5")
    assert not is_cursor_passthrough_slug("gpt-5.5")


def test_cursor_subscription_aliases_resolve_expected_models():
    expected = {
        "cx-auto": "auto",
        "cx-grok-4-7": "grok-4.7-high",
        "cx-fable-5-1": "claude-fable-5-1-medium",
        "cx-fable-5-1-high": "claude-fable-5-1-high",
        "cx-fable-5": "claude-fable-5-medium",
        "cx-fable-5-high": "claude-fable-5-high",
        "cx-opus-5": "claude-opus-5-thinking-high",
        "cx-opus-5-5": "claude-opus-5-5-medium",
        "cx-sol-5-6": "gpt-5.6-sol-medium",
        "cx-sol-5-6-high": "gpt-5.6-sol-high",
    }
    assert expected.keys() <= cursor_passthrough_display_names().keys()
    for alias, upstream in expected.items():
        assert is_cursor_passthrough_slug(alias)
        assert cursor_upstream_model(alias) == upstream


def test_build_cursor_prompt_from_responses_body():
    body = {
        "model": "composer-2-5",
        "instructions": "You are Codex.",
        "input": [{"role": "user", "content": "Hello"}],
    }
    prompt = build_cursor_prompt(body)
    assert "You are Codex." in prompt
    assert "Hello" in prompt


def test_cursor_stream_parser_emits_deltas():
    parser = CursorStreamParser()
    line1 = json.dumps(
        {
            "type": "assistant",
            "message": {"role": "assistant", "content": [{"type": "text", "text": "Hel"}]},
            "timestamp_ms": 1,
        }
    )
    line2 = json.dumps(
        {
            "type": "assistant",
            "message": {"role": "assistant", "content": [{"type": "text", "text": "Hello"}]},
            "timestamp_ms": 2,
        }
    )
    assert parser.feed_line(line1) == "Hel"
    assert parser.feed_line(line2) == "lo"


@pytest.mark.parametrize("workspace", [None, "workspace", "absolute"])
async def test_iter_cursor_agent_events_does_not_kill_normal_completion(monkeypatch, tmp_path, workspace):
    monkeypatch.chdir(tmp_path)
    expected_workspace = tmp_path
    monkeypatch.delenv("CODEX_SHIM_CURSOR_WORKSPACE", raising=False)
    if workspace is not None:
        expected_workspace = tmp_path / workspace
        expected_workspace.mkdir()
        override = str(expected_workspace) if workspace == "absolute" else workspace
        monkeypatch.setenv("CODEX_SHIM_CURSOR_WORKSPACE", override)
    monkeypatch.setenv("CURSOR_AGENT_BIN", "bin/cursor-agent")
    spawned = []
    class FakeStdin:
        def write(self, data):
            self.data = data

        async def drain(self):
            pass

        def close(self):
            pass

    class FakeReader:
        def __init__(self, chunks):
            self.chunks = list(chunks)

        async def read(self, _size):
            if self.chunks:
                return self.chunks.pop(0)
            return b""

    class FakeProc:
        def __init__(self):
            self.stdin = FakeStdin()
            self.stdout = FakeReader(
                [
                    json.dumps(
                        {
                            "type": "result",
                            "subtype": "success",
                            "result": "Hello",
                        }
                    ).encode()
                    + b"\n",
                    b"",
                ]
            )
            self.stderr = FakeReader([b""])
            self.returncode = None
            self.killed = False

        def kill(self):
            self.killed = True
            self.returncode = -9

        async def wait(self):
            if self.returncode is None:
                self.returncode = 0
            return self.returncode

    proc = FakeProc()

    async def fake_create_subprocess_exec(*args, **kwargs):
        spawned.append((args, kwargs))
        assert args[0] == str(tmp_path / "bin/cursor-agent")
        assert args[args.index("--workspace") + 1] == str(expected_workspace)
        assert kwargs["cwd"] == str(expected_workspace)
        return proc

    monkeypatch.setattr(
        "codex_shim.cursor_passthrough.asyncio.create_subprocess_exec",
        fake_create_subprocess_exec,
    )

    events = [event async for event in iter_cursor_agent_events("prompt", "composer-2.5")]

    assert proc.killed is False
    assert proc.returncode == 0
    assert events[-1] == {"type": "completed", "text": "Hello"}
    assert not any(event["type"] == "usage" for event in events)
    assert proc.stdin.data == b"prompt"

    # Identical requests must not add per-request IDs or change launch context.
    _ = [event async for event in iter_cursor_agent_events("prompt", "composer-2.5")]
    assert spawned[0] == spawned[1]


@pytest.mark.parametrize(
    "usage, expected",
    [
        # Observed stream-json format: the installed CLI explicitly subtracts
        # cache counters before serializing inputTokens.
        (
            {"inputTokens": 15712, "outputTokens": 34, "cacheReadTokens": 1152, "cacheWriteTokens": 0},
            {"input_tokens": 15712, "output_tokens": 34, "cache_read_input_tokens": 1152, "cache_creation_input_tokens": 0},
        ),
        (
            {"input_tokens": 10, "output_tokens": 2, "cache_read_input_tokens": 90, "cache_creation_input_tokens": 20},
            {"input_tokens": 10, "output_tokens": 2, "cache_read_input_tokens": 90, "cache_creation_input_tokens": 20},
        ),
        # Responses-style totals/details must not become cache-exclusive usage.
        (
            {"input_tokens": 100, "output_tokens": 2, "input_tokens_details": {"cached_tokens": 90}},
            {"input_tokens": 100, "output_tokens": 2, "input_tokens_details": {"cached_tokens": 90}},
        ),
        # Hook-style snake_case cache fields have different provenance; retain
        # them without guessing their semantics from the spelling alone.
        (
            {"input_tokens": 100, "cache_read_tokens": 90, "cache_write_tokens": 0},
            {"input_tokens": 100, "cache_read_tokens": 90, "cache_write_tokens": 0},
        ),
        ({"inputTokens": 10}, {"input_tokens": 10}),
        ({"inputTokens": 0, "cacheReadTokens": 0}, {"input_tokens": 0, "cache_read_input_tokens": 0}),
        ({"inputTokens": None, "cacheReadTokens": None}, {}),
        ({"input_tokens": 7, "inputTokens": 10}, {"input_tokens": 7}),
        ({}, {}),
    ],
)
def test_cursor_stream_parser_preserves_usage_without_inventing_cache_hits(usage, expected):
    parser = CursorStreamParser()
    original = json.loads(json.dumps(usage))
    parser.feed_line(json.dumps({"type": "result", "usage": usage}))
    assert parser.usage == expected
    assert usage == original


def test_cursor_prompt_is_stable_across_identical_requests():
    body = {"model": "cx-auto", "instructions": "Stable instructions", "input": "Hello"}
    assert build_cursor_prompt(body) == build_cursor_prompt(body)
