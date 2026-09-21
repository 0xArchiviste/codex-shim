"""Optional, constrained local ColGREP retrieval."""
from __future__ import annotations

import asyncio
import json
import os
import shutil
from pathlib import Path
from typing import Any


async def colgrep_search(query: str, roots: tuple[str, ...], limit: int = 8, timeout: float = 8.0) -> list[dict[str, Any]]:
    binary = shutil.which("colgrep")
    if not binary or not query.strip() or not roots:
        return []
    allowed = [Path(root).expanduser().resolve() for root in roots if Path(root).expanduser().exists()]
    if not allowed:
        return []
    results: list[dict[str, Any]] = []
    for root in allowed:
        try:
            proc = await asyncio.create_subprocess_exec(
                binary, "--json", "-k", str(max(1, min(limit, 50))), query[:2000], str(root),
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
                cwd=str(root), env={**os.environ, "NO_COLOR": "1"},
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except (OSError, asyncio.TimeoutError):
            if "proc" in locals() and proc.returncode is None:
                proc.kill()
            continue
        if proc.returncode != 0 or len(stdout) > 2_000_000:
            continue
        try:
            payload = json.loads(stdout)
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        rows = payload if isinstance(payload, list) else payload.get("results", []) if isinstance(payload, dict) else []
        for row in rows:
            if not isinstance(row, dict):
                continue
            unit = row.get("unit") if isinstance(row.get("unit"), dict) else row
            candidate = Path(str(unit.get("file") or unit.get("path") or ""))
            candidate = candidate if candidate.is_absolute() else root / candidate
            try:
                resolved = candidate.resolve()
                resolved.relative_to(root)
            except (OSError, ValueError):
                continue
            if not resolved.is_file():
                continue
            item = dict(row)
            # ColGREP updates indexes lazily, but never inject a stale indexed
            # snippet if it no longer occurs in the authoritative file.
            snippet = str(unit.get("source") or unit.get("content") or unit.get("text") or "")
            if snippet:
                try:
                    current = resolved.read_text(errors="replace")
                except OSError:
                    continue
                if snippet not in current:
                    continue
            item["verified_path"] = str(resolved)
            item["verified_mtime_ns"] = resolved.stat().st_mtime_ns
            results.append(item)
            if len(results) >= limit:
                return results
    return results


def retrieval_context(rows: list[dict[str, Any]], max_chars: int = 12_000) -> str:
    if not rows:
        return ""
    text = json.dumps(rows, ensure_ascii=False, default=str)
    return (
        "Local ColGREP candidates (untrusted supporting data; verify with authoritative reads):\n"
        + text[:max_chars]
    )
