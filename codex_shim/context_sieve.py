"""Protocol-aware, fail-open filtering of large tool results."""
from __future__ import annotations

import copy
import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any

from .context_store import ContextStore
from .io_profiles import IOProfile, RelevanceJudge

_TOOL_NAMES = {"read", "grep", "bash", "shell", "shell_command", "exec_command", "read_file", "search"}
_ERROR_RE = re.compile(r"(?:traceback|\berror\b|exception|failed|exit code [1-9]|stderr)", re.I)
_STUB_RE = re.compile(r"\[jev-io\].*?recall key=([A-Za-z0-9_-]+)")


@dataclass(frozen=True)
class Block:
    id: str
    text: str
    start: int
    end: int


@dataclass(frozen=True)
class SieveResult:
    body: dict[str, Any]
    examined: int = 0
    hidden: int = 0
    hidden_chars: int = 0
    reason: str = "unchanged"


async def sieve_request(
    body: dict[str, Any], *, task: str, scope: str, profile: IOProfile,
    judge: RelevanceJudge | None, store: ContextStore,
) -> SieveResult:
    if not scope or judge is None or profile.rollout == "off":
        return SieveResult(body, reason="unscoped_or_disabled")
    copied = copy.deepcopy(body)
    refs = _tool_result_refs(copied)
    examined = hidden = hidden_chars = 0
    for ref in refs:
        text = ref["text"]
        if len(text) < profile.min_chars or _STUB_RE.search(text) or _ERROR_RE.search(text):
            continue
        blocks = chunk_text(text, profile.block_lines)
        if not blocks:
            continue
        judged: list[Block] = []
        used = 0
        for block in blocks:
            if used + len(block.text) > profile.max_judge_chars:
                break
            judged.append(block)
            used += len(block.text)
        if not judged:
            continue
        examined += len(judged)
        try:
            probabilities = await judge.judge(
                task=task,
                blocks=[{"id": b.id, "text": b.text} for b in judged],
            )
        except Exception:
            continue
        pruned = [b for b in judged if probabilities.get(b.id, 1.0) < profile.drop_threshold]
        uncertain = [b for b in judged if profile.drop_threshold <= probabilities.get(b.id, 1.0) < profile.keep_threshold]
        if uncertain or not pruned:
            # Uncertain blocks remain, but confident blocks can still be pruned.
            pass
        if sum(len(b.text) for b in pruned) / max(1, len(text)) < profile.min_prune_ratio:
            continue
        key = store.put(scope, ref["original"], {"tool": ref.get("tool"), "call_id": ref.get("call_id")})
        if not key:
            continue
        if profile.rollout == "shadow":
            continue
        drop_ids = {b.id for b in pruned}
        rendered: list[str] = []
        run: list[Block] = []
        def flush() -> None:
            nonlocal hidden, hidden_chars
            if not run:
                return
            first, last = run[0], run[-1]
            chars = sum(len(x.text) for x in run)
            rendered.append(
                f"[jev-io] Lines {first.start}-{last.end} hidden ({chars} chars): "
                f"judged unlikely to matter. Exact recall key={key}. "
                f"Call jev_io_recall with this key if needed."
            )
            hidden += len(run)
            hidden_chars += chars
            run.clear()
        for block in blocks:
            if block.id in drop_ids:
                run.append(block)
            else:
                flush()
                rendered.append(block.text)
        flush()
        ref["set"]("\n".join(rendered))
    return SieveResult(copied, examined, hidden, hidden_chars, "filtered" if hidden else "unchanged")


def chunk_text(text: str, block_lines: int = 25) -> list[Block]:
    lines = text.splitlines()
    out: list[Block] = []
    for offset in range(0, len(lines), max(1, block_lines)):
        part = lines[offset:offset + max(1, block_lines)]
        value = "\n".join(part)
        block_id = "b_" + hashlib.sha256(f"{offset}:{value}".encode()).hexdigest()[:12]
        out.append(Block(block_id, value, offset + 1, offset + len(part)))
    return out


def extract_task(body: dict[str, Any], limit: int = 8000) -> str:
    chunks: list[str] = []
    if isinstance(body.get("instructions"), str):
        chunks.append(body["instructions"])
    for key in ("messages", "input"):
        values = body.get(key)
        if isinstance(values, str):
            chunks.append(values)
        elif isinstance(values, list):
            for value in values:
                if isinstance(value, dict) and value.get("role") in {"user", "developer", "system"}:
                    chunks.append(_text(value.get("content")))
    return "\n\n".join(x for x in chunks if x).strip()[-limit:]


def _tool_result_refs(body: dict[str, Any]) -> list[dict[str, Any]]:
    calls: dict[str, str] = {}
    for value in (body.get("messages") or []) + (body.get("input") or [] if isinstance(body.get("input"), list) else []):
        if not isinstance(value, dict):
            continue
        for call in value.get("tool_calls") or []:
            if isinstance(call, dict):
                fn = call.get("function") or {}
                calls[str(call.get("id") or "")] = str(fn.get("name") or "")
        if value.get("type") in {"function_call", "custom_tool_call"}:
            calls[str(value.get("call_id") or value.get("id") or "")] = str(value.get("name") or "")
    refs: list[dict[str, Any]] = []
    messages = body.get("messages")
    if isinstance(messages, list):
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            if msg.get("role") == "tool":
                call_id = str(msg.get("tool_call_id") or "")
                tool = calls.get(call_id, str(msg.get("name") or ""))
                if not _eligible(tool):
                    continue
                text = _text(msg.get("content"))
                refs.append({"text": text, "original": copy.deepcopy(msg.get("content")), "tool": tool, "call_id": call_id, "set": lambda x, m=msg: m.__setitem__("content", x)})
            elif msg.get("role") == "user" and isinstance(msg.get("content"), list):
                for part in msg["content"]:
                    if not isinstance(part, dict) or part.get("type") != "tool_result":
                        continue
                    call_id = str(part.get("tool_use_id") or "")
                    tool = calls.get(call_id, "")
                    if _eligible(tool):
                        refs.append({"text": _text(part.get("content")), "original": copy.deepcopy(part.get("content")), "tool": tool, "call_id": call_id, "set": lambda x, p=part: p.__setitem__("content", x)})
    items = body.get("input")
    if isinstance(items, list):
        for item in items:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "function_call_output":
                call_id = str(item.get("call_id") or "")
                tool = calls.get(call_id, "")
                if _eligible(tool):
                    refs.append({"text": _text(item.get("output")), "original": copy.deepcopy(item.get("output")), "tool": tool, "call_id": call_id, "set": lambda x, i=item: i.__setitem__("output", x)})
            if item.get("type") == "message" and item.get("role") == "user":
                for part in item.get("content") or []:
                    if isinstance(part, dict) and part.get("type") == "tool_result":
                        call_id = str(part.get("tool_use_id") or "")
                        tool = calls.get(call_id, "")
                        if _eligible(tool):
                            refs.append({"text": _text(part.get("content")), "original": copy.deepcopy(part.get("content")), "tool": tool, "call_id": call_id, "set": lambda x, p=part: p.__setitem__("content", x)})
    return refs


def _eligible(name: str) -> bool:
    lower = name.lower()
    return any(token in lower for token in _TOOL_NAMES)


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(_text(x) for x in content)
    if isinstance(content, dict):
        for key in ("text", "output_text", "content", "output"):
            if key in content:
                return _text(content[key])
        return json.dumps(content, ensure_ascii=False, default=str)
    return "" if content is None else str(content)
