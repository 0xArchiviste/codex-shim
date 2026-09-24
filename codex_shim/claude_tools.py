"""Offline validation for the Claude text-tool protocol.

No provider calls, tool execution, or remote schema retrieval occur here.
Ordinary prose returns []; attempted malformed replies raise safe ValueErrors.
JSON Schema ``format`` is an annotation, following jsonschema's default policy.
"""
from __future__ import annotations

import json
import math
import re
from typing import Any

from jsonschema import validators
from referencing import Registry
from referencing.exceptions import NoSuchResource

_FENCE = "codex-shim-tool"
_REPLY = re.compile(r"\s*```codex-shim-tool[^\S\r\n]*\r?\n(.*?)\r?\n```\s*", re.DOTALL)


def _deny_remote(uri: str):
    raise NoSuchResource(ref=uri)


_REGISTRY = Registry(retrieve=_deny_remote)


def claude_tool_definitions(body: dict[str, Any]) -> list[dict[str, Any]]:
    """Normalize Chat/Responses function tools without modifying exact names.

    Native/freeform tools are explicitly unsupported, rather than silently
    converted into functions with misleading schemas. Safe errors omit payloads.
    """
    tools = body.get("tools", [])
    if not isinstance(tools, list):
        raise ValueError("Claude tools must be a list")
    definitions = []
    names: set[str] = set()
    for tool in tools:
        if not isinstance(tool, dict) or tool.get("type", "function") != "function":
            raise ValueError("Claude bridge supports function tools only")
        fn = tool.get("function", tool)
        if not isinstance(fn, dict):
            raise ValueError("Invalid Claude tool definition")
        name = fn.get("name")
        if not isinstance(name, str) or not name.strip() or name in names:
            raise ValueError("Claude tool names must be nonempty and unique")
        schema = fn.get("parameters", {"type": "object"})
        _validator(schema)
        names.add(name)
        definitions.append({"name": name, "description": fn.get("description") or "", "parameters": schema})
    return definitions


def render_claude_tool_definitions(body: dict[str, Any]) -> str:
    """Return one JSON definition per line, retaining client names exactly."""
    try:
        return "\n".join(json.dumps(row, ensure_ascii=False, allow_nan=False) for row in claude_tool_definitions(body))
    except (TypeError, ValueError):
        raise ValueError("Invalid Claude tool definitions") from None


def _validator(schema: Any):
    try:
        if not isinstance(schema, (dict, bool)):
            raise ValueError
        cls = validators.validator_for(schema, default=None) if isinstance(schema, dict) and "$schema" in schema else validators.Draft202012Validator
        if cls is None:
            raise ValueError
        cls.check_schema(schema)
        return cls(schema, registry=_REGISTRY)
    except Exception:
        raise ValueError("Invalid or unsupported Claude tool schema") from None


def _policy(body: dict[str, Any], names: set[str]) -> tuple[str, str | None]:
    choice = body.get("tool_choice", "auto")
    if choice is None:
        choice = "auto"
    if isinstance(choice, str) and choice in {"auto", "none", "required"}:
        return choice, None
    if isinstance(choice, dict) and choice.get("type") == "function":
        fn = choice.get("function", choice)
        name = fn.get("name") if isinstance(fn, dict) else None
        if isinstance(name, str) and name in names:
            return "required", name
    raise ValueError("Invalid or unsupported Claude tool choice")


def _reject_constant(value: str):
    raise ValueError("Nonstandard JSON constant")


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key")
        result[key] = value
    return result


def _finite(value: Any) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(_finite(child) for child in value.values())
    if isinstance(value, list):
        return all(_finite(child) for child in value)
    return True


def validate_claude_tool_reply(text: str, body: dict[str, Any]) -> list[dict[str, str]]:
    """Return atomic validated calls, [] for prose, or raise a safe ValueError.

    Returned arguments are standard JSON strings. Missing required/forced calls,
    invalid attempted fences, unknown names, and policy violations are errors.
    All errors deliberately exclude model output, arguments, and schema details.
    """
    definitions = {row["name"]: row for row in claude_tool_definitions(body)}
    choice, forced = _policy(body, set(definitions))
    parallel = body.get("parallel_tool_calls", True)
    if not isinstance(parallel, bool):
        raise ValueError("Invalid Claude parallel tool policy")
    if not isinstance(text, str):
        raise ValueError("Invalid Claude reply")
    match = _REPLY.fullmatch(text)
    if not match:
        if _FENCE in text:
            raise ValueError("Malformed Claude tool reply")
        if choice == "required":
            raise ValueError("Claude reply omitted a required tool call")
        return []
    if choice == "none":
        raise ValueError("Claude tool calls are disabled")
    try:
        payload = json.loads(match.group(1), parse_constant=_reject_constant, object_pairs_hook=_unique_object)
        if not isinstance(payload, dict) or set(payload) != {"tool_calls"} or not _finite(payload):
            raise ValueError
        rows = payload["tool_calls"]
        if not isinstance(rows, list) or not rows or (not parallel and len(rows) > 1):
            raise ValueError
        calls = []
        for row in rows:
            if not isinstance(row, dict) or set(row) != {"name", "arguments"}:
                raise ValueError
            name, arguments = row["name"], row["arguments"]
            if not isinstance(name, str) or name not in definitions or (forced is not None and name != forced):
                raise ValueError
            if not isinstance(arguments, dict):
                raise ValueError
            _validator(definitions[name]["parameters"]).validate(arguments)
            calls.append({"name": name, "arguments": json.dumps(arguments, sort_keys=True, ensure_ascii=False, allow_nan=False)})
        return calls
    except Exception:
        # Includes unresolved/remote references, invalid regex schemas and JSON
        # recursion failures. Never expose schema validation instance contents.
        raise ValueError("Invalid Claude tool reply or tool policy violation") from None
