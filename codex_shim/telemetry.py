"""Metadata-only request accounting. Never persist provider payloads or error messages."""
from __future__ import annotations

import asyncio
from contextvars import ContextVar
from contextlib import contextmanager
import json
import os
from pathlib import Path
import re
try:
    import sqlite3
except ImportError:  # Some custom Python builds omit the optional stdlib extension.
    sqlite3 = None

StorageError = sqlite3.Error if sqlite3 is not None else OSError
import time
import uuid

from aiohttp import web

FIELDS = ("input_tokens", "output_tokens", "cached_tokens", "cache_write_tokens")
ENDPOINTS = {"/v1/chat/completions", "/v1/responses", "/v1/responses/compact", "/v1/messages"}
_current = ContextVar("usage_record", default=None)


def _number(value):
    return value if isinstance(value, int) and not isinstance(value, bool) and 0 <= value <= 2**53 else None


def normalize_usage(usage):
    """Missing counters remain NULL; Anthropic input excludes cache read/write."""
    if not isinstance(usage, dict):
        return {}
    result = {
        "input_tokens": _number(usage.get("input_tokens", usage.get("prompt_tokens"))),
        "output_tokens": _number(usage.get("output_tokens", usage.get("completion_tokens"))),
        "cached_tokens": _number(usage.get("cache_read_input_tokens")),
        "cache_write_tokens": _number(usage.get("cache_creation_input_tokens")),
    }
    details = usage.get("input_tokens_details", usage.get("prompt_tokens_details"))
    if isinstance(details, dict):
        result["cached_tokens"] = _number(details.get("cached_tokens"))
        result["cache_write_tokens"] = _number(details.get("cache_creation_input_tokens", details.get("cache_write_tokens")))
    if not isinstance(details, dict) and ("cache_read_input_tokens" in usage or "cache_creation_input_tokens" in usage):
        # A missing cache component is unknown, not an invented zero.
        parts = [result[k] for k in ("input_tokens", "cached_tokens", "cache_write_tokens")]
        result["input_tokens"] = sum(parts) if all(v is not None for v in parts) else None
    return {key: value for key, value in result.items() if value is not None}


class UsageStore:
    def __init__(self, path=None, max_rows=None, retention_days=None):
        if sqlite3 is None:
            raise OSError("Python sqlite3 support unavailable")
        self.path = Path(path or os.environ.get("CODEX_SHIM_USAGE_DB", "~/.local/state/codex-shim/usage.sqlite3")).expanduser()
        self.max_rows = max(1, int(max_rows or os.environ.get("CODEX_SHIM_USAGE_MAX_ROWS", "10000")))
        self.retention_days = max(1, int(retention_days or os.environ.get("CODEX_SHIM_USAGE_RETENTION_DAYS", "30")))
        self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        if self.path.is_symlink():
            raise OSError("Usage database must not be a symlink")
        fd = os.open(self.path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        os.fchmod(fd, 0o600)
        os.close(fd)
        with self.connect() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS requests (
                request_id TEXT PRIMARY KEY, started_at REAL, duration_ms REAL,
                model TEXT, endpoint TEXT, streaming INTEGER, http_status INTEGER,
                status TEXT, coverage TEXT, input_tokens INTEGER, output_tokens INTEGER,
                cached_tokens INTEGER, cache_write_tokens INTEGER)""")
            self.prune(db)

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=5)
        try:
            with db:
                yield db
        finally:
            db.close()

    def prune(self, db):
        db.execute("DELETE FROM requests WHERE started_at < ?", (time.time() - self.retention_days * 86400,))
        db.execute("DELETE FROM requests WHERE request_id IN (SELECT request_id FROM requests ORDER BY started_at DESC LIMIT -1 OFFSET ?)", (self.max_rows,))

    def save(self, row):
        columns = ("request_id", "started_at", "duration_ms", "model", "endpoint", "streaming", "http_status", "status", "coverage", *FIELDS)
        with self.connect() as db:
            db.execute(f"INSERT OR REPLACE INTO requests ({','.join(columns)}) VALUES ({','.join('?' for _ in columns)})", [row.get(k) for k in columns])
            self.prune(db)

    def snapshot(self):
        with self.connect() as db:
            db.row_factory = sqlite3.Row
            self.prune(db)
            rows = [dict(row) for row in db.execute("SELECT * FROM requests ORDER BY started_at DESC LIMIT 200")]
            total = db.execute("SELECT count(*) FROM requests").fetchone()[0]
            totals = {}
            for field in FIELDS:
                value, known = db.execute(f"SELECT sum({field}), count({field}) FROM requests").fetchone()
                totals[field] = {"known_total": value, "known_requests": known, "unknown_requests": total - known}
            groups = [dict(row) for row in db.execute("SELECT model, count(*) AS requests, sum(input_tokens) AS input_tokens, sum(output_tokens) AS output_tokens FROM requests GROUP BY model ORDER BY requests DESC LIMIT 100")]
        return {"requests": total, "totals": totals, "models": groups, "recent": rows, "retention_days": self.retention_days, "max_rows": self.max_rows}


class Record:
    def __init__(self, endpoint):
        self.row = dict(request_id=uuid.uuid4().hex, started_at=time.time(), endpoint=endpoint,
                        status="in_progress", model="unknown", streaming=False, coverage="primary_only")
        self.started = time.monotonic()
        self.completed = False
        self.buffer = b""
        self.discard_line = False

    def observe(self, payload):
        if not isinstance(payload, dict):
            return
        kind = payload.get("type")
        if kind == "error" or payload.get("error") or kind == "response.failed":
            self.row["status"] = "error"
        if kind == "response.incomplete" or payload.get("status") == "incomplete":
            self.row["status"] = "partial"
        obj = payload.get("response", payload.get("message", payload))
        if isinstance(obj, dict) and self.row["coverage"] != "auxiliary_unaccounted":
            self.row.update(normalize_usage(obj.get("usage")))
        if kind in {"response.completed", "message_stop", "completed"}:
            self.completed = True
        if payload.get("object") == "response" and payload.get("status") == "completed":
            self.completed = True
        for choice in payload.get("choices", []) if isinstance(payload.get("choices", []), list) else []:
            if isinstance(choice, dict) and choice.get("finish_reason"):
                self.completed = True

    def line(self, line):
        if line == "[DONE]":
            self.completed = True
            return
        try:
            self.observe(json.loads(line))
        except (ValueError, TypeError):
            pass

    def feed(self, chunk):
        # Only the raw pass-through path needs framing. Bound even a malicious giant event.
        for part in chunk.splitlines(keepends=True):
            end = part.endswith(b"\n")
            if not self.discard_line:
                if len(self.buffer) + len(part) <= 262144:
                    self.buffer += part
                else:
                    self.buffer = b""
                    self.discard_line = True
            if end:
                if not self.discard_line and self.buffer.startswith(b"data:"):
                    self.line(self.buffer[5:].decode("utf-8", "replace").strip())
                self.buffer = b""
                self.discard_line = False


@contextmanager
def suspend():
    """Exclude classifier/helper calls from the primary request counters."""
    token = _current.set(None)
    try:
        yield
    finally:
        _current.reset(token)


def observe(payload):
    record = _current.get()
    if record:
        record.observe(payload)


def observe_line(line):
    record = _current.get()
    if record:
        record.line(line)


def observe_chunk(chunk):
    record = _current.get()
    if record:
        record.feed(chunk)


def mark_disconnected():
    record = _current.get()
    if record:
        record.row["status"] = "disconnected"


def mark_auxiliary():
    record = _current.get()
    if record:
        record.row["coverage"] = "auxiliary_unaccounted"
        for field in FIELDS:
            record.row.pop(field, None)


def usage_middleware(store):
    @web.middleware
    async def middleware(request, handler):
        if request.path not in ENDPOINTS or request.method != "POST":
            return await handler(request)
        record = Record(request.path)
        token = _current.set(record)
        try:
            try:
                body = await request.json()
                if isinstance(body, dict):
                    model = body.get("model")
                    if isinstance(model, str) and re.fullmatch(r"[\w./:@+ -]{1,160}", model):
                        record.row["model"] = model
                    record.row["streaming"] = bool(body.get("stream"))
            except (ValueError, UnicodeError):
                pass
            response = await handler(request)
            record.row["http_status"] = response.status
            if request.get("shim_client_disconnected"):
                record.row["status"] = "disconnected"
            if response.status >= 400:
                record.row["status"] = "error"
            elif record.row["status"] == "in_progress":
                record.row["status"] = "completed" if not record.row["streaming"] or record.completed else "partial"
            response.headers["X-Shim-Usage-ID"] = record.row["request_id"]
            return response
        except web.HTTPException as exc:
            record.row.update(status="error", http_status=exc.status)
            raise
        except (asyncio.CancelledError, ConnectionError):
            record.row["status"] = "disconnected"
            raise
        except Exception:
            record.row["status"] = "error"
            raise
        finally:
            record.row["duration_ms"] = round((time.monotonic() - record.started) * 1000, 2)
            _current.reset(token)
            try:
                await asyncio.shield(asyncio.to_thread(store.save, record.row.copy()))
            except Exception:
                # Telemetry must never take down inference or log sensitive exceptions.
                pass
    return middleware
