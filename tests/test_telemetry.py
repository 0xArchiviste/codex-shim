from __future__ import annotations

import json
import stat
import time

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
import pytest

from codex_shim import telemetry
from codex_shim.server import ShimServer, _sse_lines

pytestmark = pytest.mark.skipif(telemetry.sqlite3 is None, reason="Python built without stdlib SQLite extension")


def test_normalize_unknown_and_cache():
    assert telemetry.normalize_usage(None) == {}
    assert telemetry.normalize_usage({"prompt_tokens": 0}) == {"input_tokens": 0}
    assert telemetry.normalize_usage({"input_tokens": 10, "output_tokens": 2, "cache_read_input_tokens": 20, "cache_creation_input_tokens": 5}) == {
        "input_tokens": 35, "output_tokens": 2, "cached_tokens": 20, "cache_write_tokens": 5}
    assert "input_tokens" not in telemetry.normalize_usage({"input_tokens": 10, "cache_read_input_tokens": 20})
    assert telemetry.normalize_usage({"prompt_tokens": True, "completion_tokens": -1}) == {}


def test_store_reload_retention_aggregation_permissions(tmp_path):
    path = tmp_path / "private" / "usage.db"
    store = telemetry.UsageStore(path, max_rows=2)
    for index in range(3):
        record = telemetry.Record("/v1/responses")
        record.row.update(started_at=time.time()+index, input_tokens=index if index != 2 else None)
        store.save(record.row)
        store.save(record.row)  # a request may not be counted twice
    restored = telemetry.UsageStore(path, max_rows=2).snapshot()
    assert restored["requests"] == 2
    assert restored["totals"]["input_tokens"] == {"known_total": 1, "known_requests": 1, "unknown_requests": 1}
    assert restored["totals"]["output_tokens"]["known_total"] is None
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
    assert b"prompt" not in path.read_bytes()


def test_sse_chunks_partial_error_and_bounded_memory():
    record = telemetry.Record("/v1/responses")
    payload = {"type": "response.completed", "response": {"usage": {"input_tokens": 12, "output_tokens": 3}}}
    data = ("data: " + json.dumps(payload) + "\r\n\r\n").encode()
    for byte in data:
        record.feed(bytes([byte]))
    record.observe(payload)  # duplicate cumulative usage is replacement, not addition
    assert record.row["input_tokens"] == 12
    assert record.completed
    record.observe({"type": "error", "error": {"message": "SECRET raw failure"}})
    assert record.row["status"] == "error"
    assert "SECRET" not in str(record.row)
    record.feed(b"data: " + b"x" * 300000)
    assert len(record.buffer) <= 262144
    record.feed(b"\n" + data)
    assert record.row["output_tokens"] == 3


async def test_real_sse_observer_does_not_modify_stream():
    class Content:
        async def iter_chunked(self, _):
            yield b'data: {"usage":{"prompt_tokens":4}}\n\ndata: [DONE]\n\n'
    class Upstream:
        content = Content()
    record = telemetry.Record("/v1/chat/completions")
    token = telemetry._current.set(record)
    try:
        lines = [line async for line in _sse_lines(Upstream())]
    finally:
        telemetry._current.reset(token)
    assert lines == ['{"usage":{"prompt_tokens":4}}', '[DONE]']
    assert record.row["input_tokens"] == 4
    assert "output_tokens" not in record.row


@pytest.mark.parametrize("mode,expected", [("success", "completed"), ("partial", "partial"), ("error", "error"), ("disconnect", "disconnected"), ("auxiliary", "partial")])
async def test_middleware_fake_paths(tmp_path, mode, expected):
    store = telemetry.UsageStore(tmp_path / "usage.db")
    async def handler(request):
        if mode == "error":
            raise web.HTTPBadGateway(text="SECRET")
        telemetry.observe({"usage": {"input_tokens": 9}})
        if mode == "success":
            telemetry.observe_line('[DONE]')
        if mode == "disconnect":
            telemetry.mark_disconnected()
        if mode == "auxiliary":
            telemetry.mark_auxiliary()
            telemetry.observe({"usage": {"input_tokens": 0}})
        return web.Response(text="fake")
    app = web.Application(middlewares=[telemetry.usage_middleware(store)])
    app.router.add_post('/v1/responses', handler)
    async with TestClient(TestServer(app)) as client:
        await client.post('/v1/responses', json={"model":"fake", "stream":True, "input":"SECRET"})
    row = store.snapshot()["recent"][0]
    assert row["status"] == expected
    assert row["output_tokens"] is None
    assert row["input_tokens"] == (None if mode in {"error", "auxiliary"} else 9)
    assert b"SECRET" not in store.path.read_bytes()


async def test_dashboard_auth_and_static_shell(monkeypatch, tmp_path):
    monkeypatch.setenv('CODEX_SHIM_USAGE_DB', str(tmp_path / 'usage.db'))
    settings = tmp_path / 'settings.json'
    settings.write_text('{"customModels":[]}')
    server = ShimServer(settings)
    server.api_key = 'test-secret'
    async with TestClient(TestServer(server.app())) as client:
        assert (await client.get('/v1/usage')).status == 401
        assert (await client.get('/v1/usage?api_key=test-secret')).status == 401
        response = await client.get('/v1/usage', headers={'Authorization':'Bearer test-secret'})
        assert response.status == 200
        assert (await response.json())['requests'] == 0
        shell = await (await client.get('/usage')).text()
        assert 'test-secret' not in shell
        assert 'localStorage' not in shell
        assert 'textContent' in shell
    server.api_key = ''
    async with TestClient(TestServer(server.app())) as client:
        assert (await client.get('/v1/usage')).status == 401
