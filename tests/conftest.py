from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _isolate_shim_runtime_env(monkeypatch, tmp_path_factory):
    """Keep local tunnel/auth env from leaking into unit tests."""
    monkeypatch.setenv("CODEX_SHIM_DISABLE_CLAUDE", "1")
    monkeypatch.delenv("CODEX_SHIM_API_KEY", raising=False)
    monkeypatch.delenv("CODEX_SHIM_ALLOWED_HOSTS", raising=False)
    runtime_dir = tmp_path_factory.mktemp("shim-api")
    monkeypatch.setenv("CODEX_SHIM_USAGE_DB", str(runtime_dir / "usage.sqlite3"))
    key_path = runtime_dir / "api-key"
    monkeypatch.setattr("codex_shim.settings.DEFAULT_SHIM_API_KEY_FILE", key_path)
    monkeypatch.setattr("codex_shim.server.load_shim_api_key", lambda path=None: "")


@pytest.fixture(autouse=True)
def _disable_cursor_passthrough_by_default(monkeypatch, request):
    if "cursor_present" in request.fixturenames:
        return

    def _off(**_kwargs):
        return False

    for target in (
        "codex_shim.cursor_passthrough.cursor_passthrough_available",
        "codex_shim.server.cursor_passthrough_available",
        "codex_shim.catalog.cursor_passthrough_available",
        "codex_shim.cli.cursor_passthrough_available",
    ):
        monkeypatch.setattr(target, _off, raising=False)
