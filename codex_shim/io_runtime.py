"""Internal Jev IO recall loop and conservative generative refinements."""
from __future__ import annotations

import copy
import json
import re
from typing import Any, Awaitable, Callable

from .context_store import ContextStore
from .ensemble import extract_assistant_text
from .io_profiles import IOProfile, RewriteConfig

RECALL_TOOL_NAME = "jev_io_recall"
RECALL_TOOL = {
    "type": "function",
    "name": RECALL_TOOL_NAME,
    "description": "Restore exact content hidden by Jev IO. Use only a recall key shown in context.",
    "parameters": {
        "type": "object",
        "properties": {"key": {"type": "string"}, "start": {"type": "integer"}, "end": {"type": "integer"}},
        "required": ["key"],
        "additionalProperties": False,
    },
}


def inject_recall_tool(body: dict[str, Any]) -> dict[str, Any]:
    copied = copy.deepcopy(body)
    if isinstance(copied.get("input"), str):
        copied["input"] = [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": copied["input"]}]}]
    tools = list(copied.get("tools") or [])
    names = {_tool_name(tool) for tool in tools if isinstance(tool, dict)}
    if RECALL_TOOL_NAME not in names:
        tools.append(copy.deepcopy(RECALL_TOOL))
    copied["tools"] = tools
    return copied


def recall_calls(payload: dict[str, Any]) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for item in payload.get("output") or []:
        if isinstance(item, dict) and item.get("type") in {"function_call", "custom_tool_call"} and item.get("name") == RECALL_TOOL_NAME:
            calls.append(item)
    choices = payload.get("choices") or []
    if choices:
        for call in ((choices[0] or {}).get("message") or {}).get("tool_calls") or []:
            fn = call.get("function") or {}
            if fn.get("name") == RECALL_TOOL_NAME:
                calls.append({"id": call.get("id"), "call_id": call.get("id"), "name": RECALL_TOOL_NAME, "arguments": fn.get("arguments", "{}"), "type": "function_call"})
    return calls


async def run_recall_loop(
    body: dict[str, Any], *, scope: str, profile: IOProfile, store: ContextStore,
    complete: Callable[[dict[str, Any]], Awaitable[dict[str, Any]]],
) -> dict[str, Any]:
    current = inject_recall_tool(body)
    for _ in range(max(1, profile.max_internal_rounds)):
        payload = await complete(current)
        calls = recall_calls(payload)
        if not calls:
            return payload
        responses: list[dict[str, Any]] = []
        for call in calls:
            args = _arguments(call.get("arguments"))
            artifact = store.get(scope, str(args.get("key") or ""))
            value = artifact.content if artifact else {"error": "unknown or expired recall key"}
            if isinstance(value, str):
                lines = value.splitlines()
                start = max(1, int(args.get("start") or 1))
                end = min(len(lines), int(args.get("end") or len(lines)))
                value = "\n".join(lines[start - 1:end])
            responses.append({"type": "function_call_output", "call_id": str(call.get("call_id") or call.get("id") or ""), "output": value})
        current = _continue_body(current, payload, responses)
    raise RuntimeError("Jev IO recall loop exceeded its round limit")


async def rewrite_text(config: RewriteConfig, *, purpose: str, original: str, task: str, post_json) -> str:
    if not original.strip():
        return original
    system = (
        "Refine text without changing intent, facts, constraints, code, identifiers, or claims. "
        "Return only the refined text. If uncertain, return the original verbatim."
    )
    prompt = f"Purpose: {purpose}\nTask:\n{task[:6000]}\n\nOriginal:\n{original[:24000]}"
    headers = {"Authorization": f"Bearer {config.api_key}", "Content-Type": "application/json", **(config.extra_headers or {})}
    payload = await post_json(
        config.base_url.rstrip("/") + "/chat/completions",
        {"model": config.model, "stream": False, "temperature": 0, "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}]},
        headers,
        config.timeout,
    )
    text = extract_assistant_text(payload).strip()
    return text if text and _rewrite_preserves_literals(original, text) else original


def _rewrite_preserves_literals(original: str, candidate: str) -> bool:
    """Reject rewrites that drop code-ish literals or explicit constraints."""
    literal_re = re.compile(r"`[^`]+`|(?:/[^\s,;]+)+|\b[A-Z][A-Z0-9_]{2,}\b|\bcs-[a-z0-9-]+\b")
    required = {match.group(0) for match in literal_re.finditer(original)}
    if any(value not in candidate for value in required):
        return False
    negations = ("do not", "don't", "must not", "never", "without")
    return all(phrase not in original.lower() or phrase in candidate.lower() for phrase in negations)


def refine_request_input(body: dict[str, Any], brief: str) -> dict[str, Any]:
    copied = copy.deepcopy(body)
    label = "Supplementary task brief (non-authoritative; original instructions control):\n" + brief
    if isinstance(copied.get("messages"), list):
        copied["messages"].insert(0, {"role": "user", "content": label})
    elif isinstance(copied.get("input"), list):
        copied["input"].insert(0, {"type": "message", "role": "user", "content": [{"type": "input_text", "text": label}]})
    return copied


def replace_terminal_text(payload: dict[str, Any], text: str) -> dict[str, Any]:
    if recall_calls(payload):
        return payload
    copied = copy.deepcopy(payload)
    choices = copied.get("choices")
    if isinstance(choices, list) and choices:
        message = (choices[0] or {}).get("message") or {}
        if message.get("tool_calls"):
            return payload
        if isinstance(message.get("content"), str):
            message["content"] = text
            return copied
    output = copied.get("output")
    if isinstance(output, list):
        if any(isinstance(x, dict) and x.get("type") != "message" for x in output):
            return payload
        for item in output:
            for part in item.get("content") or []:
                if isinstance(part, dict) and part.get("type") in {"output_text", "text"}:
                    part["text"] = text
                    return copied
    return payload


def _continue_body(body: dict[str, Any], payload: dict[str, Any], results: list[dict[str, Any]]) -> dict[str, Any]:
    copied = copy.deepcopy(body)
    if isinstance(copied.get("input"), list):
        copied["input"].extend(copy.deepcopy(payload.get("output") or []))
        copied["input"].extend(results)
        return copied
    raise RuntimeError("internal recall currently requires Responses-normalized input")


def _arguments(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    try:
        value = json.loads(raw or "{}")
        return value if isinstance(value, dict) else {}
    except (json.JSONDecodeError, TypeError):
        return {}


def _tool_name(tool: dict[str, Any]) -> str:
    return str(tool.get("name") or (tool.get("function") or {}).get("name") or "")
