"""Bounded, scoped storage for exact tool-result recall."""
from __future__ import annotations

import hashlib
import json
import secrets
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ContextArtifact:
    key: str
    scope: str
    content: Any
    created_at: float
    size: int
    metadata: dict[str, Any]


class ContextStore:
    def __init__(self, max_bytes: int = 200 * 1024 * 1024, ttl_seconds: float = 30 * 86400):
        self.max_bytes = max_bytes
        self.ttl_seconds = ttl_seconds
        self._items: OrderedDict[str, ContextArtifact] = OrderedDict()
        self._bytes = 0
        self._fingerprints: dict[tuple[str, str], str] = {}

    def put(self, scope: str, content: Any, metadata: dict[str, Any] | None = None) -> str | None:
        if not scope:
            return None
        raw = json.dumps(content, ensure_ascii=False, sort_keys=True, default=str).encode()
        if len(raw) > self.max_bytes:
            return None
        digest = hashlib.sha256(raw).hexdigest()
        existing = self._fingerprints.get((scope, digest))
        if existing and existing in self._items:
            self._items.move_to_end(existing)
            return existing
        self._expire()
        while self._items and self._bytes + len(raw) > self.max_bytes:
            self._drop(next(iter(self._items)))
        key = secrets.token_urlsafe(18)
        artifact = ContextArtifact(key, scope, content, time.time(), len(raw), metadata or {})
        self._items[key] = artifact
        self._fingerprints[(scope, digest)] = key
        self._bytes += len(raw)
        return key

    def get(self, scope: str, key: str) -> ContextArtifact | None:
        self._expire()
        artifact = self._items.get(key)
        if artifact is None or not scope or artifact.scope != scope:
            return None
        self._items.move_to_end(key)
        return artifact

    def stats(self) -> dict[str, int]:
        self._expire()
        return {"artifacts": len(self._items), "bytes": self._bytes}

    def _expire(self) -> None:
        cutoff = time.time() - self.ttl_seconds
        for key, artifact in list(self._items.items()):
            if artifact.created_at < cutoff:
                self._drop(key)

    def _drop(self, key: str) -> None:
        artifact = self._items.pop(key, None)
        if artifact is None:
            return
        self._bytes -= artifact.size
        raw = json.dumps(artifact.content, ensure_ascii=False, sort_keys=True, default=str).encode()
        self._fingerprints.pop((artifact.scope, hashlib.sha256(raw).hexdigest()), None)
