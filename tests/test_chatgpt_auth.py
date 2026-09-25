from __future__ import annotations

import base64
import json
import time

import pytest

from codex_shim import chatgpt_auth


def _jwt(exp: float) -> str:
    payload = base64.urlsafe_b64encode(json.dumps({"exp": exp}).encode()).decode().rstrip("=")
    return f"eyJhbGciOiJub25lIn0.{payload}.sig"


def _write(path, token, account="acct"):
    path.write_text(json.dumps({"tokens": {"access_token": token, "account_id": account}}))
    return path


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    monkeypatch.setattr(chatgpt_auth, "_last_good", None)
    monkeypatch.setattr(chatgpt_auth, "_hard_failed", {})
    monkeypatch.setattr(chatgpt_auth, "TRANSIENT_RETRY_DELAY", 0)


class FakeUpstream:
    def __init__(self, status, text=""):
        self.status = status
        self._text = text
        self.released = False

    async def text(self):
        return self._text

    def release(self):
        self.released = True


def _poster(responses):
    calls = []

    async def post(url, json=None, headers=None):
        calls.append(headers["Authorization"])
        return responses.pop(0)

    return post, calls


def test_load_credentials_orders_fresh_before_expired_and_dedupes(monkeypatch, tmp_path):
    now = time.time()
    primary = _write(tmp_path / "a.json", _jwt(now - 10))
    fallback = _write(tmp_path / "b.json", _jwt(now + 3600))
    dup = _write(tmp_path / "c.json", _jwt(now + 3600))
    monkeypatch.setenv(chatgpt_auth.FALLBACKS_ENV, f"{fallback}:{dup}:{tmp_path / 'missing.json'}")
    creds = chatgpt_auth.load_credentials(primary)
    assert [c.path for c in creds] == [str(fallback), str(primary)]


async def test_transient_401_retries_same_login(tmp_path):
    creds = chatgpt_auth.load_credentials(_write(tmp_path / "a.json", "tok-a"))
    first = FakeUpstream(401, '{"code": "invalid_api_key", "message": "sk-svcac***"}')
    post, calls = _poster([first, FakeUpstream(200)])
    upstream = await chatgpt_auth.post_with_failover(post, creds, "u", {}, {})
    assert upstream.status == 200
    assert calls == ["Bearer tok-a", "Bearer tok-a"]
    assert first.released


async def test_fails_over_to_next_login_and_remembers_it(monkeypatch, tmp_path):
    primary = _write(tmp_path / "a.json", "tok-a")
    fallback = _write(tmp_path / "b.json", "tok-b")
    monkeypatch.setenv(chatgpt_auth.FALLBACKS_ENV, str(fallback))
    creds = chatgpt_auth.load_credentials(primary)
    bad = '{"code": "invalid_api_key"}'
    post, calls = _poster([FakeUpstream(401, bad), FakeUpstream(401, bad), FakeUpstream(200)])
    upstream = await chatgpt_auth.post_with_failover(post, creds, "u", {}, {})
    assert upstream.status == 200
    assert calls == ["Bearer tok-a", "Bearer tok-a", "Bearer tok-b"]
    assert [c.path for c in chatgpt_auth.load_credentials(primary)][0] == str(fallback)


async def test_hard_401_skips_retry_and_cools_down(monkeypatch, tmp_path):
    primary = _write(tmp_path / "a.json", "tok-a")
    fallback = _write(tmp_path / "b.json", "tok-b")
    monkeypatch.setenv(chatgpt_auth.FALLBACKS_ENV, str(fallback))
    creds = chatgpt_auth.load_credentials(primary)
    post, calls = _poster([FakeUpstream(401, '{"code": "token_expired"}'), FakeUpstream(401, "nope")])
    upstream = await chatgpt_auth.post_with_failover(post, creds, "u", {}, {})
    assert upstream.status == 401
    assert calls == ["Bearer tok-a", "Bearer tok-b"]
    assert set(chatgpt_auth._hard_failed) == {str(primary), str(fallback)}


async def test_non_401_error_is_returned_without_failover(tmp_path):
    creds = chatgpt_auth.load_credentials(_write(tmp_path / "a.json", "tok-a"))
    post, calls = _poster([FakeUpstream(429)])
    upstream = await chatgpt_auth.post_with_failover(post, creds, "u", {}, {})
    assert upstream.status == 429
    assert calls == ["Bearer tok-a"]
