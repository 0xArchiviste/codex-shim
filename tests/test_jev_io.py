from __future__ import annotations

import json
from pathlib import Path

import pytest

from codex_shim.context_sieve import chunk_text, sieve_request
from codex_shim.context_store import ContextStore
from codex_shim.io_profiles import IOProfile, active_profiles, find_profile, load_io_config
from codex_shim.io_runtime import RECALL_TOOL_NAME, inject_recall_tool, replace_terminal_text, run_recall_loop, _rewrite_preserves_literals
from codex_shim.retrieval import colgrep_search


class FakeJudge:
    def __init__(self, score=0.0):
        self.score = score

    async def judge(self, *, task, blocks):
        return {block["id"]: self.score for block in blocks}


def profile(**kwargs):
    values = {"slug": "cs-sol-jev-io", "display_name": "IO", "rollout": "active", "min_chars": 10, "block_lines": 2}
    values.update(kwargs)
    return IOProfile(**values)


def chat_body(text: str):
    return {
        "model": "cs-sol-jev-io",
        "messages": [
            {"role": "assistant", "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "Read", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "call_1", "content": text},
            {"role": "user", "content": "find the handler"},
        ],
    }


def test_profiles_are_opt_in_and_configurable(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    path = tmp_path / "models.json"
    path.write_text("{}")
    assert active_profiles(load_io_config(path)) == []
    path.write_text(json.dumps({"jev_io": {"enabled": True, "profiles": [{"slug": "cs-sol-jev-io", "rollout": "shadow"}]}}))
    config = load_io_config(path)
    assert find_profile(config, "cs-sol-jev-io").rollout == "shadow"
    assert {p.slug for p in active_profiles(config)} == {"cs-sol-jev-io", "cs-sol-jev-io-max"}


def test_chunk_text_preserves_line_ranges():
    blocks = chunk_text("a\nb\nc\nd\ne", 2)
    assert [(b.start, b.end, b.text) for b in blocks] == [(1, 2, "a\nb"), (3, 4, "c\nd"), (5, 5, "e")]


@pytest.mark.asyncio
async def test_sieve_is_reversible_and_scope_isolated():
    original = "alpha\nbeta\ngamma\ndelta\nepsilon\nzeta"
    store = ContextStore()
    result = await sieve_request(chat_body(original), task="handler", scope="scope-a", profile=profile(), judge=FakeJudge(), store=store)
    visible = result.body["messages"][1]["content"]
    assert "[jev-io]" in visible
    key = visible.split("key=")[1].split(".")[0]
    assert store.get("scope-a", key).content == original
    assert store.get("scope-b", key) is None
    assert chat_body(original)["messages"][1]["content"] == original


@pytest.mark.asyncio
async def test_sieve_preserves_errors_and_uncertain_content():
    store = ContextStore()
    error = await sieve_request(chat_body("line\nERROR: broke\nmore\ndata"), task="fix", scope="s", profile=profile(), judge=FakeJudge(), store=store)
    assert error.hidden == 0
    uncertain = await sieve_request(chat_body("a\nb\nc\nd\ne\nf"), task="fix", scope="s", profile=profile(), judge=FakeJudge(.4), store=store)
    assert uncertain.hidden == 0


@pytest.mark.asyncio
async def test_shadow_records_artifact_without_rewriting():
    original = "a\nb\nc\nd\ne\nf"
    store = ContextStore()
    result = await sieve_request(chat_body(original), task="x", scope="s", profile=profile(rollout="shadow"), judge=FakeJudge(), store=store)
    assert result.body["messages"][1]["content"] == original
    assert store.stats()["artifacts"] == 1


@pytest.mark.asyncio
async def test_recall_loop_never_returns_internal_tool_call():
    store = ContextStore()
    key = store.put("scope", "one\ntwo\nthree")
    calls = []

    async def complete(body):
        calls.append(body)
        if len(calls) == 1:
            return {"output": [{"type": "function_call", "id": "x", "call_id": "x", "name": RECALL_TOOL_NAME, "arguments": json.dumps({"key": key, "start": 2})}]}
        assert body["input"][-1]["output"] == "two\nthree"
        return {"output": [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "done"}]}]}

    payload = await run_recall_loop({"input": []}, scope="scope", profile=profile(), store=store, complete=complete)
    assert payload["output"][0]["type"] == "message"
    assert any(tool.get("name") == RECALL_TOOL_NAME for tool in calls[0]["tools"])


def test_rewrite_preserves_tool_outputs():
    payload = {"choices": [{"message": {"content": "old", "tool_calls": [{"id": "x"}]}}]}
    assert replace_terminal_text(payload, "new") == payload
    prose = {"choices": [{"message": {"content": "old"}}]}
    assert replace_terminal_text(prose, "new")["choices"][0]["message"]["content"] == "new"


def test_rewrite_literal_validation():
    original = "Do not rename `handle_request` in /src/server.py or API_KEY."
    assert _rewrite_preserves_literals(original, original)
    assert not _rewrite_preserves_literals(original, "Improve the handler.")


@pytest.mark.asyncio
async def test_colgrep_missing_is_graceful(monkeypatch, tmp_path):
    monkeypatch.setattr("codex_shim.retrieval.shutil.which", lambda _name: None)
    assert await colgrep_search("thing", (str(tmp_path),)) == []
