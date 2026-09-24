from __future__ import annotations

import hashlib
import json
import re
from typing import Any

RESPONSES_ID_MAX_LEN = 64


def clamp_responses_id(value: Any, *, prefix: str = "id", require_prefix: bool = False) -> str:
    """Clamp Responses item ids to 64 chars, optionally enforcing a required prefix.

    ChatGPT's Codex backend rejects ids longer than 64 chars, and function_call
    item ``id`` values must begin with ``fc``.
    """
    text = str(value or "").strip() or prefix
    if require_prefix and not text.startswith(prefix):
        text = f"{prefix}_{text}"
    if len(text) <= RESPONSES_ID_MAX_LEN:
        return text
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:20]
    if require_prefix:
        prefix_part = f"{prefix}_"
        budget = RESPONSES_ID_MAX_LEN - len(digest) - 1 - len(prefix_part)
        if budget < 1:
            return f"{prefix}_{digest}"[:RESPONSES_ID_MAX_LEN]
        remainder = text[len(prefix) :].lstrip("_-") if text.startswith(prefix) else text
        mid = re.sub(r"[^A-Za-z0-9_-]+", "", remainder)[:budget] or "x"
        return f"{prefix_part}{mid}_{digest}"[:RESPONSES_ID_MAX_LEN]
    keep = RESPONSES_ID_MAX_LEN - len(digest) - 1
    head = re.sub(r"[^A-Za-z0-9_-]+", "", text[:keep]) or prefix
    return f"{head[:keep]}_{digest}"[:RESPONSES_ID_MAX_LEN]


def responses_function_call_ids(raw_id: Any) -> tuple[str, str]:
    """Map a chat/Cursor tool-call id to Responses ``(id, call_id)``."""
    source = str(raw_id or "").strip() or "0"
    return (
        clamp_responses_id(source, prefix="fc", require_prefix=True),
        clamp_responses_id(source, prefix="call"),
    )


THINK_RE = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)

SHIM_ENCRYPTED_CONTENT_PREFIX = "anthropic-thinking-v1:"
_THINKING_MAGIC = SHIM_ENCRYPTED_CONTENT_PREFIX


def _decode_thinking_blob(encoded: Any) -> dict[str, Any] | None:
    import base64

    if not isinstance(encoded, str) or not encoded.startswith(_THINKING_MAGIC):
        return None
    blob = encoded[len(_THINKING_MAGIC) :]
    try:
        raw = base64.urlsafe_b64decode(blob.encode("ascii"))
        data = json.loads(raw.decode("utf-8"))
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    return data


def responses_to_chat(body: dict[str, Any], upstream_model: str) -> dict[str, Any]:
    messages = []
    instructions = body.get("instructions")
    if instructions:
        messages.append({"role": "system", "content": _content_to_text(instructions)})
    pending_reasoning: str | None = None
    for m in _responses_input_to_messages(body.get("input")):
        if m.get("_reasoning_only"):
            summary = m.get("summary") or []
            text = " ".join(item.get("text", "") for item in summary if isinstance(item, dict))
            if text:
                pending_reasoning = text
            continue
        if pending_reasoning and m.get("role") == "assistant":
            m["reasoning_content"] = pending_reasoning
            pending_reasoning = None
        messages.append(m)
    messages = _sanitize_chat_messages(_merge_consecutive_messages(_normalize_chat_roles(messages)))

    chat: dict[str, Any] = {
        "model": upstream_model,
        "messages": messages or [{"role": "user", "content": ""}],
        "stream": bool(body.get("stream", False)),
    }
    _copy_if_present(body, chat, "temperature")
    _copy_if_present(body, chat, "top_p")
    _copy_if_present(body, chat, "max_output_tokens", "max_tokens")
    _copy_if_present(body, chat, "max_tokens")
    _copy_if_present(body, chat, "parallel_tool_calls")
    _copy_if_present(body, chat, "reasoning_effort")
    _copy_if_present(body, chat, "prompt_cache_key")
    _copy_if_present(body, chat, "prompt_cache_retention")

    tools = _responses_tools_to_chat_tools(body.get("tools"))
    if tools:
        chat["tools"] = tools
        tool_choice = _responses_tool_choice_to_chat(body.get("tool_choice"), body.get("tools"))
        if tool_choice is not None:
            chat["tool_choice"] = tool_choice
    return chat


def responses_to_anthropic(body: dict[str, Any], upstream_model: str, max_tokens: int | None) -> dict[str, Any]:
    system_parts: list[str] = []
    instructions = body.get("instructions")
    if instructions:
        system_parts.append(_content_to_text(instructions))

    messages: list[dict[str, Any]] = []

    def append(role: str, content: Any) -> None:
        if messages and messages[-1]["role"] == role and isinstance(messages[-1]["content"], list) and isinstance(content, list):
            messages[-1]["content"].extend(content)
        else:
            messages.append({"role": role, "content": content})

    pending_thinking: list[dict[str, Any]] = []
    for chat_msg in _responses_input_to_messages(body.get("input")):
        role = chat_msg.get("role", "user")
        if chat_msg.get("_reasoning_only"):
            decoded = _decode_thinking_blob(chat_msg.get("encrypted_content"))
            if decoded is not None:
                pending_thinking.append(decoded)
            else:
                # Summary-only fallback: emit a plain `thinking` block (no
                # signature). Anthropic requires `signature` on the original
                # session; if we lack it, skip rather than upsetting strict
                # APIs.
                for summary in chat_msg.get("summary") or []:
                    text = summary.get("text") if isinstance(summary, dict) else None
                    if text:
                        pending_thinking.append({"type": "thinking", "thinking": text, "signature": ""})
            continue
        if role in {"system", "developer"}:
            system_parts.append(_content_to_text(chat_msg.get("content", "")))
            continue
        if role == "assistant":
            blocks: list[dict[str, Any]] = []
            blocks.extend(pending_thinking)
            pending_thinking = []
            content = chat_msg.get("content")
            if content:
                blocks.extend(_chat_content_to_anthropic_blocks(content))
            for call in chat_msg.get("tool_calls") or []:
                fn = call.get("function") or {}
                args_raw = fn.get("arguments") or ""
                try:
                    args_obj = json.loads(args_raw) if args_raw else {}
                except json.JSONDecodeError:
                    args_obj = {"_raw": args_raw}
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": call.get("id") or "call_0",
                        "name": fn.get("name") or "",
                        "input": args_obj,
                    }
                )
            if blocks:
                append("assistant", blocks)
            continue
        if role == "tool":
            # Reasoning items only attach to assistant turns; drop any pending
            # thinking when a tool result interrupts (shouldn't happen in
            # normal Codex flows but defensive).
            pending_thinking = []
            append(
                "user",
                [
                    {
                        "type": "tool_result",
                        "tool_use_id": chat_msg.get("tool_call_id") or "call_0",
                        "content": _content_to_text(chat_msg.get("content", "")),
                    }
                ],
            )
            continue
        # user / anything else
        pending_thinking = []
        append(role, _chat_content_to_anthropic_content(chat_msg.get("content", "")))

    # If reasoning items appeared without a following assistant turn (e.g. the
    # final pending think after a tool_use round-trip), emit an assistant
    # message containing them so Anthropic's API accepts the followup.
    if pending_thinking:
        append("assistant", pending_thinking)

    anthropic: dict[str, Any] = {
        "model": upstream_model,
        "messages": messages or [{"role": "user", "content": ""}],
        "max_tokens": int(body.get("max_output_tokens") or body.get("max_tokens") or max_tokens or 4096),
        "stream": bool(body.get("stream", False)),
    }
    if system_parts:
        anthropic["system"] = "\n\n".join(system_parts)
    _copy_if_present(body, anthropic, "temperature")
    _copy_if_present(body, anthropic, "top_p")

    tools = _responses_tools_to_anthropic_tools(body.get("tools"))
    if tools:
        anthropic["tools"] = tools
    return anthropic


def chat_to_responses_request(body: dict[str, Any], upstream_model: str, max_tokens: int | None = None) -> dict[str, Any]:
    """Convert an OpenAI chat-completions request into a Responses request.

    Used when clients talk chat (or Anthropic-via-chat) to the shim and the
    upstream is ChatGPT/Codex Responses passthrough — the reverse of
    ``responses_to_chat``.
    """
    system_parts: list[str] = []
    input_items: list[dict[str, Any]] = []

    for msg in body.get("messages") or []:
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role") or "user")
        if role in {"system", "developer"}:
            text = _content_to_text(msg.get("content", ""))
            if text:
                system_parts.append(text)
            continue
        if role == "tool":
            input_items.append(
                {
                    "type": "function_call_output",
                    "call_id": clamp_responses_id(msg.get("tool_call_id") or "call_0", prefix="call"),
                    "output": _content_to_text(msg.get("content", "")),
                }
            )
            continue
        if role == "assistant":
            reasoning = msg.get("reasoning_content") or msg.get("reasoning")
            if reasoning:
                input_items.append(
                    {
                        "type": "reasoning",
                        "summary": [{"type": "summary_text", "text": str(reasoning)}],
                    }
                )
            content = msg.get("content")
            if content not in (None, ""):
                input_items.append(
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": _chat_content_to_responses_content(content, as_output=True),
                    }
                )
            for call in msg.get("tool_calls") or []:
                if not isinstance(call, dict):
                    continue
                fn = call.get("function") or {}
                item_id, call_id = responses_function_call_ids(call.get("id") or "call_0")
                input_items.append(
                    {
                        "type": "function_call",
                        "id": item_id,
                        "call_id": call_id,
                        "name": fn.get("name") or "",
                        "arguments": fn.get("arguments") or "",
                    }
                )
            continue
        input_items.append(
            {
                "type": "message",
                "role": "user" if role == "user" else role,
                "content": _chat_content_to_responses_content(msg.get("content", ""), as_output=False),
            }
        )

    converted: dict[str, Any] = {
        "model": upstream_model,
        "input": input_items
        or [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": ""}]}],
        "stream": bool(body.get("stream", False)),
    }
    if system_parts:
        converted["instructions"] = "\n\n".join(system_parts)
    _copy_if_present(body, converted, "temperature")
    _copy_if_present(body, converted, "top_p")
    _copy_if_present(body, converted, "max_tokens", "max_output_tokens")
    if max_tokens and "max_output_tokens" not in converted:
        converted["max_output_tokens"] = max_tokens
    _copy_if_present(body, converted, "parallel_tool_calls")
    _copy_if_present(body, converted, "prompt_cache_key")
    _copy_if_present(body, converted, "prompt_cache_retention")

    tools = _chat_tools_to_responses_tools(body.get("tools"))
    if tools:
        converted["tools"] = tools
        tool_choice = _chat_tool_choice_to_responses(body.get("tool_choice"))
        if tool_choice is not None:
            converted["tool_choice"] = tool_choice

    effort = body.get("reasoning_effort")
    if effort:
        converted["reasoning"] = {"effort": effort}
    return converted


def chat_to_anthropic(body: dict[str, Any], upstream_model: str, max_tokens: int | None) -> dict[str, Any]:
    pseudo_responses = chat_to_responses_request(body, upstream_model, max_tokens=max_tokens)
    return responses_to_anthropic(pseudo_responses, upstream_model, max_tokens)


def anthropic_messages_to_chat(body: dict[str, Any], upstream_model: str, max_tokens: int | None = None) -> dict[str, Any]:
    messages: list[dict[str, Any]] = []
    system = body.get("system")
    if system:
        messages.append({"role": "system", "content": _anthropic_content_to_text(system)})

    for raw_msg in body.get("messages") or []:
        if not isinstance(raw_msg, dict):
            continue
        role = str(raw_msg.get("role") or "user")
        content = raw_msg.get("content", "")
        if role == "assistant":
            messages.append(_anthropic_assistant_message_to_chat(content))
        elif role == "user":
            messages.extend(_anthropic_user_message_to_chat(content))
        else:
            messages.append({"role": role, "content": _anthropic_content_to_chat_content(content)})

    messages = _sanitize_chat_messages(_merge_consecutive_messages(_normalize_chat_roles(messages)))
    chat: dict[str, Any] = {
        "model": upstream_model,
        "messages": messages or [{"role": "user", "content": ""}],
        "stream": bool(body.get("stream", False)),
    }
    _copy_if_present(body, chat, "temperature")
    _copy_if_present(body, chat, "top_p")
    _copy_if_present(body, chat, "max_tokens")
    if max_tokens and "max_tokens" not in chat:
        chat["max_tokens"] = max_tokens
    _copy_if_present(body, chat, "stop_sequences", "stop")

    thinking = body.get("thinking")
    if isinstance(thinking, dict) and thinking.get("effort"):
        chat["reasoning_effort"] = thinking["effort"]
    output_config = body.get("output_config")
    if isinstance(output_config, dict) and output_config.get("effort"):
        chat["reasoning_effort"] = output_config["effort"]

    tools = _anthropic_tools_to_chat_tools(body.get("tools"))
    if tools:
        chat["tools"] = tools
        tool_choice = _anthropic_tool_choice_to_chat(body.get("tool_choice"))
        if tool_choice is not None:
            chat["tool_choice"] = tool_choice
    if chat["stream"]:
        stream_options = dict(body.get("stream_options") or {})
        stream_options["include_usage"] = True
        chat["stream_options"] = stream_options
    return chat


def chat_completion_to_anthropic_message(payload: dict[str, Any], requested_model: str) -> dict[str, Any]:
    choice = (payload.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    content: list[dict[str, Any]] = []
    reasoning = message.get("reasoning_content") or message.get("reasoning")
    if reasoning:
        content.append({"type": "thinking", "thinking": str(reasoning)})
    text = strip_think(message.get("content") or "")
    if text:
        content.append({"type": "text", "text": text})
    for call in message.get("tool_calls") or []:
        if not isinstance(call, dict):
            continue
        fn = call.get("function") or {}
        args_raw = fn.get("arguments") or ""
        try:
            args_obj = json.loads(args_raw) if args_raw else {}
        except json.JSONDecodeError:
            args_obj = {"_raw": args_raw}
        content.append(
            {
                "type": "tool_use",
                "id": call.get("id") or "call_0",
                "name": fn.get("name") or "",
                "input": args_obj,
            }
        )
    if not content:
        content.append({"type": "text", "text": ""})

    response: dict[str, Any] = {
        "id": payload.get("id") or "msg_chat",
        "type": "message",
        "role": "assistant",
        "model": requested_model,
        "content": content,
        "stop_reason": _chat_finish_to_anthropic_stop(choice.get("finish_reason")),
        "stop_sequence": None,
    }
    usage = _responses_usage_to_anthropic_usage(normalize_responses_usage(payload.get("usage")))
    if usage is not None:
        response["usage"] = usage
    return response


def anthropic_to_chat_response(payload: dict[str, Any], requested_model: str) -> dict[str, Any]:
    content = ""
    tool_calls = []
    for block in payload.get("content", []):
        if block.get("type") == "text":
            content += block.get("text", "")
        elif block.get("type") == "tool_use":
            tool_calls.append(
                {
                    "id": block.get("id"),
                    "type": "function",
                    "function": {
                        "name": block.get("name", ""),
                        "arguments": _jsonish(block.get("input", {})),
                    },
                }
            )
    message: dict[str, Any] = {"role": "assistant", "content": strip_think(content)}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {
        "id": payload.get("id", "chatcmpl-anthropic"),
        "object": "chat.completion",
        "created": 0,
        "model": requested_model,
        "choices": [{"index": 0, "message": message, "finish_reason": _anthropic_stop(payload.get("stop_reason"))}],
    }


def chat_completion_to_response(payload: dict[str, Any], requested_model: str, tool_types: dict[str, str] | None = None) -> dict[str, Any]:
    choice = (payload.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    output: list[dict[str, Any]] = []
    reasoning = message.get("reasoning_content")
    if reasoning:
        output.append(
            {
                "id": "reasoning_0",
                "type": "reasoning",
                "status": "completed",
                "summary": [{"type": "summary_text", "text": reasoning}],
            }
        )
    text = strip_think(message.get("content") or "")
    if text:
        output.append(
            {
                "id": "msg_0",
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        )
    tool_types = tool_types or {}
    for call in message.get("tool_calls") or []:
        fn = call.get("function") or {}
        name = fn.get("name", "")
        original_type = tool_types.get(name, "")
        item_type = "function_call"
        if original_type == "apply_patch":
            item_type = "custom_tool_call"
        elif original_type.startswith("web_search"):
            item_type = "web_search_call"
        output.append(
            {
                "id": call.get("id", "call_0"),
                "type": item_type,
                "status": "completed",
                "call_id": call.get("id", "call_0"),
                "name": name,
                "arguments": fn.get("arguments", ""),
            }
        )
    return {
        "id": payload.get("id", "resp_chat"),
        "object": "response",
        "created_at": payload.get("created", 0),
        "status": "completed",
        "model": requested_model,
        "output": output,
        "usage": normalize_responses_usage(payload.get("usage")),
    }


def anthropic_to_response(payload: dict[str, Any], requested_model: str, tool_types: dict[str, str] | None = None) -> dict[str, Any]:
    response = chat_completion_to_response(anthropic_to_chat_response(payload, requested_model), requested_model, tool_types)
    response["usage"] = normalize_responses_usage(payload.get("usage"))
    return response


def response_to_chat_completion(
    payload: dict[str, Any],
    requested_model: str,
    tool_name_aliases: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Convert a completed Responses object into an OpenAI chat.completion."""
    text_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    reasoning_parts: list[str] = []

    for item in payload.get("output") or []:
        if not isinstance(item, dict):
            continue
        item_type = item.get("type")
        if item_type == "message":
            for part in item.get("content") or []:
                if not isinstance(part, dict):
                    continue
                if part.get("type") in {"output_text", "text"} and part.get("text"):
                    text_parts.append(str(part["text"]))
        elif item_type in {"function_call", "custom_tool_call", "web_search_call"}:
            call_id = item.get("call_id") or item.get("id") or f"call_{len(tool_calls)}"
            name, arguments = remap_chatgpt_tool_call(item, tool_name_aliases)
            tool_calls.append(
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": arguments,
                    },
                }
            )
        elif item_type == "reasoning":
            for summary in item.get("summary") or []:
                if isinstance(summary, dict) and summary.get("text"):
                    reasoning_parts.append(str(summary["text"]))
        elif item_type in {"output_text", "text"} and item.get("text"):
            text_parts.append(str(item["text"]))

    content = strip_think("".join(text_parts))
    message: dict[str, Any] = {
        "role": "assistant",
        "content": content if content else (None if tool_calls else ""),
    }
    if tool_calls:
        message["tool_calls"] = tool_calls
    if reasoning_parts:
        message["reasoning_content"] = "\n".join(reasoning_parts)

    finish_reason = "tool_calls" if tool_calls else "stop"
    if payload.get("status") == "incomplete":
        finish_reason = "length"

    result: dict[str, Any] = {
        "id": payload.get("id") or "chatcmpl_resp",
        "object": "chat.completion",
        "created": int(payload.get("created_at") or payload.get("created") or 0),
        "model": requested_model,
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
    }
    usage = _responses_usage_to_chat_usage(normalize_responses_usage(payload.get("usage")))
    if usage is not None:
        result["usage"] = usage
    return result


def response_to_anthropic_message(payload: dict[str, Any], requested_model: str) -> dict[str, Any]:
    """Convert a completed Responses object into an Anthropic Messages response."""
    return chat_completion_to_anthropic_message(response_to_chat_completion(payload, requested_model), requested_model)


def normalize_responses_usage(usage: Any) -> dict[str, Any] | None:
    if not isinstance(usage, dict):
        return None

    input_tokens = _int_token(usage.get("input_tokens"))
    if input_tokens is None:
        input_tokens = _int_token(usage.get("prompt_tokens"))

    output_tokens = _int_token(usage.get("output_tokens"))
    if output_tokens is None:
        output_tokens = _int_token(usage.get("completion_tokens"))

    # Anthropic input_tokens excludes both cache reads and cache writes. OpenAI
    # input/prompt totals already include caches, whose counters live in details.
    # Only expand raw Anthropic usage, so normalization is also idempotent.
    cache_read = _int_token(usage.get("cache_read_input_tokens"))
    cache_created = _int_token(usage.get("cache_creation_input_tokens"))
    anthropic_usage = (
        "prompt_tokens" not in usage
        and "input_tokens_details" not in usage
        and (cache_read is not None or cache_created is not None)
    )
    if anthropic_usage:
        input_tokens = (input_tokens or 0) + (cache_read or 0) + (cache_created or 0)

    total_tokens = _int_token(usage.get("total_tokens"))
    if anthropic_usage:
        total_tokens = None
    if total_tokens is None and input_tokens is not None and output_tokens is not None:
        total_tokens = input_tokens + output_tokens

    if input_tokens is None:
        input_tokens = max(total_tokens - output_tokens, 0) if total_tokens is not None and output_tokens is not None else 0
    if output_tokens is None:
        output_tokens = max(total_tokens - input_tokens, 0) if total_tokens is not None else 0
    if total_tokens is None:
        total_tokens = input_tokens + output_tokens

    normalized: dict[str, Any] = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }

    input_details: dict[str, Any] = {}
    if isinstance(usage.get("input_tokens_details"), dict):
        input_details.update(usage["input_tokens_details"])
    if isinstance(usage.get("prompt_tokens_details"), dict):
        input_details.update(usage["prompt_tokens_details"])

    if cache_read is not None:
        input_details.setdefault("cached_tokens", cache_read)
        input_details.setdefault("cache_read_input_tokens", cache_read)
    if cache_created is not None:
        input_details.setdefault("cache_creation_input_tokens", cache_created)

    if input_details:
        normalized["input_tokens_details"] = input_details

    output_details: dict[str, Any] = {}
    if isinstance(usage.get("output_tokens_details"), dict):
        output_details.update(usage["output_tokens_details"])
    if isinstance(usage.get("completion_tokens_details"), dict):
        output_details.update(usage["completion_tokens_details"])
    if output_details:
        normalized["output_tokens_details"] = output_details

    return normalized


def _int_token(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def strip_think(text: str) -> str:
    return THINK_RE.sub("", text or "")


def _responses_input_to_messages(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, str):
        return [{"role": "user", "content": value}]
    if not isinstance(value, list):
        return [{"role": "user", "content": _responses_content_to_chat_content(value)}]
    messages: list[dict[str, Any]] = []
    pending_tool_calls: list[dict[str, Any]] = []

    def flush_pending_assistant_tool_calls():
        if pending_tool_calls:
            messages.append({"role": "assistant", "content": None, "tool_calls": list(pending_tool_calls)})
            pending_tool_calls.clear()

    for item in value:
        if isinstance(item, str):
            flush_pending_assistant_tool_calls()
            messages.append({"role": "user", "content": item})
            continue
        if not isinstance(item, dict):
            continue
        item_type = item.get("type")
        if item_type in {"message", None} and "role" in item:
            flush_pending_assistant_tool_calls()
            role = item.get("role", "user")
            if role == "developer":
                role = "system"
            messages.append({"role": role, "content": _responses_content_to_chat_content(item.get("content", ""))})
        elif item_type in {"input_text", "text", "input_image"}:
            flush_pending_assistant_tool_calls()
            messages.append({"role": "user", "content": _responses_content_to_chat_content(item)})
        elif item_type == "computer_call_output":
            flush_pending_assistant_tool_calls()
            messages.append({"role": "user", "content": _computer_output_to_chat_content(item)})
        elif item_type == "function_call":
            # Coalesce consecutive function_call items into a single assistant
            # message with multiple tool_calls so chat-completions upstreams
            # accept the subsequent tool messages.
            call_id = item.get("call_id") or item.get("id") or "call_0"
            pending_tool_calls.append(
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": item.get("name") or "",
                        "arguments": item.get("arguments") or "",
                    },
                }
            )
        elif item_type == "function_call_output":
            flush_pending_assistant_tool_calls()
            output = item.get("output", "")
            messages.append({"role": "tool", "tool_call_id": item.get("call_id"), "content": _content_to_text(output)})
            if _has_visual_content(output):
                messages.append({"role": "user", "content": _visual_feedback_chat_content(output, item.get("call_id"))})
        elif item_type == "reasoning":
            # For Chat-Completions upstreams reasoning is informational only.
            # We keep it as a marker so the Anthropic translator can reattach
            # encrypted_content as a `thinking` block on the assistant turn.
            flush_pending_assistant_tool_calls()
            messages.append(
                {
                    "role": "assistant",
                    "_reasoning_only": True,
                    "encrypted_content": item.get("encrypted_content"),
                    "summary": item.get("summary") or [],
                    "content": None,
                }
            )
    flush_pending_assistant_tool_calls()
    return messages


def _responses_content_to_chat_content(content: Any) -> str | list[dict[str, Any]]:
    parts = _chat_parts_from_content(content)
    if not parts:
        return ""
    if any(part.get("type") == "image_url" for part in parts):
        return parts
    return "\n".join(str(part.get("text", "")) for part in parts if part.get("type") == "text")


def _computer_output_to_chat_content(item: dict[str, Any]) -> str | list[dict[str, Any]]:
    call_id = item.get("call_id") or item.get("id")
    prefix = f"Computer output for {call_id}." if call_id else "Computer output."
    parts = _chat_parts_from_content(item.get("output", ""))
    if any(part.get("type") == "image_url" for part in parts):
        return [{"type": "text", "text": prefix}, *parts]
    text = "\n".join(str(part.get("text", "")) for part in parts if part.get("type") == "text")
    return f"{prefix}\n{text}" if text else prefix


def _visual_feedback_chat_content(output: Any, call_id: Any) -> list[dict[str, Any]]:
    prefix = f"Visual tool output for {call_id}." if call_id else "Visual tool output."
    parts = [part for part in _chat_parts_from_content(output) if part.get("type") == "image_url"]
    return [{"type": "text", "text": prefix}, *parts]


def _chat_parts_from_content(content: Any) -> list[dict[str, Any]]:
    if content is None:
        return []
    if isinstance(content, str):
        return [{"type": "text", "text": content}] if content else []
    if isinstance(content, list):
        parts: list[dict[str, Any]] = []
        for part in content:
            parts.extend(_chat_parts_from_content(part))
        return parts
    if isinstance(content, dict):
        content_type = str(content.get("type") or "")
        if content_type in {"input_text", "output_text", "text"}:
            text = str(content.get("text", ""))
            return [{"type": "text", "text": text}] if text else []
        if content_type in {"input_image", "image_url"} or "image_url" in content:
            image = _chat_image_part(content)
            return [image] if image else []
        if content_type == "computer_call_output":
            return _chat_parts_from_content(content.get("output"))
        if "output" in content and _has_visual_content(content.get("output")):
            return _chat_parts_from_content(content.get("output"))
        if "content" in content:
            return _chat_parts_from_content(content["content"])
        if "text" in content:
            text = str(content.get("text", ""))
            return [{"type": "text", "text": text}] if text else []
    return []


def _chat_image_part(part: dict[str, Any]) -> dict[str, Any] | None:
    url = _image_url_from_part(part)
    if not url:
        return None
    image_url: dict[str, Any] = {"url": url}
    detail = part.get("detail") or part.get("image_detail")
    if detail is None:
        nested = part.get("image_url")
        if isinstance(nested, dict):
            detail = nested.get("detail")
    if detail and detail not in ("low", "auto", "high", "xhigh"):
        # Codex Desktop sends "original" which is not a standard OpenAI Chat
        # Completions value — providers like Kimi K2.6 reject it (400).
        # Map it to "high" (the closest standard equivalent). Any unknown
        # detail value falls back to "auto".
        detail = "high" if detail == "original" else "auto"
    if detail:
        image_url["detail"] = detail
    return {"type": "image_url", "image_url": image_url}


def _image_url_from_part(part: dict[str, Any]) -> str:
    image_url = part.get("image_url")
    if isinstance(image_url, str):
        return image_url
    if isinstance(image_url, dict) and isinstance(image_url.get("url"), str):
        return image_url["url"]
    for key in ("url", "file_url"):
        value = part.get(key)
        if isinstance(value, str):
            return value
    return ""


def _has_visual_content(content: Any) -> bool:
    return any(part.get("type") == "image_url" for part in _chat_parts_from_content(content))


def _chat_content_to_anthropic_content(content: Any) -> str | list[dict[str, Any]]:
    blocks = _chat_content_to_anthropic_blocks(content)
    if not any(block.get("type") == "image" for block in blocks):
        return "\n".join(block.get("text", "") for block in blocks if block.get("type") == "text")
    return blocks


def _chat_content_to_anthropic_blocks(content: Any) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    for part in _chat_parts_from_content(content):
        if part.get("type") == "text":
            text = str(part.get("text", ""))
            if text:
                blocks.append({"type": "text", "text": text})
        elif part.get("type") == "image_url":
            block = _chat_image_part_to_anthropic(part)
            if block:
                blocks.append(block)
    return blocks or [{"type": "text", "text": ""}]


def _chat_image_part_to_anthropic(part: dict[str, Any]) -> dict[str, Any] | None:
    image_url = part.get("image_url")
    url = ""
    if isinstance(image_url, dict):
        url = str(image_url.get("url") or "")
    elif isinstance(image_url, str):
        url = image_url
    if not url:
        return None
    if url.startswith("data:"):
        match = re.match(r"data:([^;,]+);base64,(.*)", url, re.DOTALL)
        if not match:
            return None
        return {"type": "image", "source": {"type": "base64", "media_type": match.group(1), "data": match.group(2)}}
    return {"type": "image", "source": {"type": "url", "url": url}}


def _anthropic_content_to_chat_content(content: Any) -> str | list[dict[str, Any]]:
    parts = _anthropic_content_to_chat_parts(content)
    if not parts:
        return ""
    if any(part.get("type") == "image_url" for part in parts):
        return parts
    return "\n".join(str(part.get("text", "")) for part in parts if part.get("type") == "text")


def _anthropic_content_to_text(content: Any) -> str:
    chat_content = _anthropic_content_to_chat_content(content)
    return _content_to_text(chat_content)


def _anthropic_content_to_chat_parts(content: Any) -> list[dict[str, Any]]:
    if content is None:
        return []
    if isinstance(content, str):
        return [{"type": "text", "text": content}] if content else []
    if isinstance(content, list):
        parts: list[dict[str, Any]] = []
        for block in content:
            if not isinstance(block, dict):
                parts.extend(_anthropic_content_to_chat_parts(block))
                continue
            block_type = block.get("type")
            if block_type == "text":
                text = str(block.get("text", ""))
                if text:
                    parts.append({"type": "text", "text": text})
            elif block_type == "image":
                image = _anthropic_image_block_to_chat_part(block)
                if image:
                    parts.append(image)
            elif block_type == "image_url" or "image_url" in block:
                image = _chat_image_part(block)
                if image:
                    parts.append(image)
            elif "content" in block:
                parts.extend(_anthropic_content_to_chat_parts(block.get("content")))
        return parts
    if isinstance(content, dict):
        return _anthropic_content_to_chat_parts([content])
    return [{"type": "text", "text": str(content)}]


def _anthropic_image_block_to_chat_part(block: dict[str, Any]) -> dict[str, Any] | None:
    source = block.get("source")
    if not isinstance(source, dict):
        return None
    url = ""
    if source.get("type") == "base64":
        media_type = str(source.get("media_type") or "image/png")
        data = str(source.get("data") or "")
        if data:
            url = f"data:{media_type};base64,{data}"
    elif source.get("type") == "url":
        url = str(source.get("url") or "")
    if not url:
        return None
    return {"type": "image_url", "image_url": {"url": url}}


def _anthropic_assistant_message_to_chat(content: Any) -> dict[str, Any]:
    text_parts: list[Any] = []
    tool_calls: list[dict[str, Any]] = []
    reasoning_parts: list[str] = []
    blocks = content if isinstance(content, list) else [{"type": "text", "text": content}]
    for block in blocks:
        if not isinstance(block, dict):
            text_parts.append(block)
            continue
        block_type = block.get("type")
        if block_type == "tool_use":
            tool_calls.append(
                {
                    "id": block.get("id") or f"call_{len(tool_calls)}",
                    "type": "function",
                    "function": {
                        "name": block.get("name") or "",
                        "arguments": _jsonish(block.get("input", {})),
                    },
                }
            )
        elif block_type in {"thinking", "redacted_thinking"}:
            thinking = block.get("thinking") or block.get("data") or ""
            if thinking:
                reasoning_parts.append(str(thinking))
        else:
            text_parts.append(block)
    message: dict[str, Any] = {"role": "assistant", "content": _anthropic_content_to_chat_content(text_parts) if text_parts else ""}
    if tool_calls:
        message["tool_calls"] = tool_calls
    if reasoning_parts:
        message["reasoning_content"] = "\n".join(reasoning_parts)
    return message


def _anthropic_user_message_to_chat(content: Any) -> list[dict[str, Any]]:
    if not isinstance(content, list):
        return [{"role": "user", "content": _anthropic_content_to_chat_content(content)}]
    tool_messages: list[dict[str, Any]] = []
    user_parts: list[Any] = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "tool_result":
            tool_content = block.get("content", "")
            tool_use_id = block.get("tool_use_id") or "call_0"
            tool_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_use_id,
                    "content": _anthropic_content_to_text(tool_content),
                }
            )
            if _anthropic_has_visual_content(tool_content):
                user_parts.extend(_anthropic_visual_tool_result_chat_parts(tool_content, tool_use_id))
        else:
            user_parts.append(block)
    messages = list(tool_messages)
    if user_parts or not messages:
        messages.append({"role": "user", "content": _anthropic_content_to_chat_content(user_parts)})
    return messages


def _anthropic_has_visual_content(content: Any) -> bool:
    return any(part.get("type") == "image_url" for part in _anthropic_content_to_chat_parts(content))


def _anthropic_visual_tool_result_chat_parts(content: Any, tool_use_id: Any) -> list[dict[str, Any]]:
    prefix = f"Visual tool result for {tool_use_id}." if tool_use_id else "Visual tool result."
    images = [part for part in _anthropic_content_to_chat_parts(content) if part.get("type") == "image_url"]
    return [{"type": "text", "text": prefix}, *images]


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                if part.get("type") in {"input_text", "output_text", "text"}:
                    parts.append(str(part.get("text", "")))
                elif part.get("type") in {"input_image", "image_url"} or "image_url" in part:
                    parts.append("[image]")
                elif "content" in part:
                    parts.append(_content_to_text(part["content"]))
        return "\n".join(p for p in parts if p)
    if isinstance(content, dict):
        if content.get("type") in {"input_image", "image_url"} or "image_url" in content:
            return "[image]"
        if "output" in content:
            return _content_to_text(content.get("output"))
        if "text" in content:
            return str(content.get("text", ""))
        return str(content)
    return str(content)


def _responses_tools_to_chat_tools(tools: Any) -> list[dict[str, Any]]:
    if not isinstance(tools, list):
        return []
    converted = []
    for tool in tools:
        function_tool = _responses_tool_to_chat_function(tool)
        if function_tool:
            converted.append(function_tool)
    return converted


def _chat_content_to_responses_content(content: Any, *, as_output: bool) -> list[dict[str, Any]]:
    text_type = "output_text" if as_output else "input_text"
    image_type = "input_image"
    parts: list[dict[str, Any]] = []
    for part in _chat_parts_from_content(content):
        if part.get("type") == "text":
            text = str(part.get("text", ""))
            if text:
                parts.append({"type": text_type, "text": text})
        elif part.get("type") == "image_url":
            url = _image_url_from_part(part)
            if not url:
                continue
            image: dict[str, Any] = {"type": image_type, "image_url": url}
            detail = None
            image_url = part.get("image_url")
            if isinstance(image_url, dict):
                detail = image_url.get("detail")
            if detail:
                image["detail"] = detail
            parts.append(image)
    return parts or [{"type": text_type, "text": ""}]


CHATGPT_SAFE_APPLY_PATCH_NAME = "cs_ApplyPatch"

_CHATGPT_UNSUPPORTED_NATIVE_TOOL_TYPES = frozenset(
    {
        "apply_patch",
        "local_shell",
        "computer_use",
        "computer_use_preview",
    }
)


def _tool_name(tool: dict[str, Any]) -> str:
    fn = tool.get("function")
    if isinstance(fn, dict) and fn.get("name"):
        return str(fn["name"])
    if tool.get("name"):
        return str(tool["name"])
    return str(tool.get("type") or "")


def is_apply_patch_tool_name(name: Any) -> bool:
    compact = re.sub(r"[^a-z0-9]", "", str(name or "").lower())
    return compact in {"applypatch"}


def chatgpt_tool_name_aliases(tools: Any) -> dict[str, str]:
    """Map ChatGPT-safe tool names back to the client-facing names."""
    aliases: dict[str, str] = {}
    if not isinstance(tools, list):
        return aliases
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        name = _tool_name(tool)
        tool_type = str(tool.get("type") or "").strip().lower()
        if is_apply_patch_tool_name(name) or tool_type == "apply_patch":
            aliases[CHATGPT_SAFE_APPLY_PATCH_NAME] = name or "ApplyPatch"
    return aliases


def _function_tool_entry(name: str, tool: dict[str, Any], fn: dict[str, Any] | None = None) -> dict[str, Any]:
    src = fn if isinstance(fn, dict) else tool
    entry: dict[str, Any] = {
        "type": "function",
        "name": str(name),
        "description": src.get("description") or tool.get("description") or "",
    }
    params = src.get("parameters") or tool.get("parameters") or tool.get("input_schema")
    if params:
        entry["parameters"] = params
    if isinstance(src.get("strict"), bool):
        entry["strict"] = src["strict"]
    elif isinstance(tool.get("strict"), bool):
        entry["strict"] = tool["strict"]
    return entry


def sanitize_chatgpt_tools(tools: Any) -> list[dict[str, Any]]:
    """Normalize tools for ChatGPT's Codex Responses backend.

    That backend rejects native ``apply_patch`` and also rejects ``parameters``
    on a tool it has already classified as apply_patch by name. Keep Cursor's
    ApplyPatch as a plain function tool under a non-colliding name.
    """
    if not isinstance(tools, list):
        return []
    converted: list[dict[str, Any]] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        name = _tool_name(tool)
        tool_type = str(tool.get("type") or "function").strip().lower()
        fn = tool["function"] if isinstance(tool.get("function"), dict) else None
        if is_apply_patch_tool_name(name) or tool_type == "apply_patch":
            source = dict(tool)
            if not source.get("description"):
                source["description"] = (fn or {}).get("description") or "Apply a unified diff patch."
            if not source.get("parameters") and not source.get("input_schema") and not (fn or {}).get("parameters"):
                source["parameters"] = {
                    "type": "object",
                    "properties": {"input": {"type": "string"}},
                }
            converted.append(_function_tool_entry(CHATGPT_SAFE_APPLY_PATCH_NAME, source, fn))
            continue
        if tool_type in _CHATGPT_UNSUPPORTED_NATIVE_TOOL_TYPES:
            fname = name or tool_type
            if not fname:
                continue
            converted.append(_function_tool_entry(fname, tool, fn))
            continue
        fname = (fn or {}).get("name") if fn else name
        if not fname:
            continue
        if tool_type in {"function", "custom"} or fn is not None or tool.get("parameters") or tool.get("input_schema"):
            converted.append(_function_tool_entry(str(fname), tool, fn))
            continue
        # Preserve Responses-native tools that ChatGPT actually accepts (e.g. web_search).
        converted.append({"type": tool_type})
    return converted


def remap_chatgpt_tool_call(item: dict[str, Any], aliases: dict[str, str] | None = None) -> tuple[str, str]:
    """Return (client_tool_name, chat_arguments) for a Responses tool call item."""
    aliases = aliases or {}
    name = str(item.get("name") or "")
    name = aliases.get(name, name)
    args = item.get("arguments")
    if args in (None, ""):
        args = item.get("input") or ""
    if isinstance(args, dict):
        return name, json.dumps(args)
    text = str(args or "")
    if name and is_apply_patch_tool_name(name) and text.strip() and not text.lstrip().startswith("{"):
        return aliases.get(CHATGPT_SAFE_APPLY_PATCH_NAME, name), json.dumps({"input": text})
    return name, text


def summarize_chatgpt_tools(tools: Any) -> list[dict[str, Any]]:
    """Compact tool metadata for debug logs (no argument payloads)."""
    summary: list[dict[str, Any]] = []
    if not isinstance(tools, list):
        return summary
    for index, tool in enumerate(tools):
        if not isinstance(tool, dict):
            summary.append({"index": index, "invalid": True})
            continue
        summary.append(
            {
                "index": index,
                "type": tool.get("type"),
                "name": tool.get("name") or _tool_name(tool),
                "keys": sorted(str(key) for key in tool.keys()),
                "has_parameters": "parameters" in tool,
            }
        )
    return summary


def _chat_tools_to_responses_tools(tools: Any) -> list[dict[str, Any]]:
    if not isinstance(tools, list):
        return []
    converted: list[dict[str, Any]] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        if tool.get("type") == "function" and isinstance(tool.get("function"), dict):
            fn = tool["function"]
            name = fn.get("name")
            if not name:
                continue
            converted.append(_function_tool_entry(str(name), tool, fn))
            continue
        name = tool.get("name")
        if name:
            converted.append(_function_tool_entry(str(name), tool, None))
    return converted


def _chat_tool_choice_to_responses(tool_choice: Any) -> Any:
    if tool_choice is None:
        return None
    if isinstance(tool_choice, str):
        return tool_choice
    if isinstance(tool_choice, dict):
        if tool_choice.get("type") == "function":
            fn = tool_choice.get("function") or {}
            name = fn.get("name") or tool_choice.get("name")
            if name:
                return {"type": "function", "name": str(name)}
        if tool_choice.get("name"):
            return {"type": "function", "name": str(tool_choice["name"])}
    return tool_choice


def _responses_usage_to_chat_usage(usage: dict[str, Any] | None) -> dict[str, Any] | None:
    if usage is None:
        return None
    result: dict[str, Any] = {
        "prompt_tokens": int(usage.get("input_tokens") or 0),
        "completion_tokens": int(usage.get("output_tokens") or 0),
        "total_tokens": int(usage.get("total_tokens") or 0),
    }
    input_details = usage.get("input_tokens_details")
    if isinstance(input_details, dict):
        # Keep provider-specific cache-write/read counters and native OpenAI
        # details (for example audio_tokens), not just the cached-token alias.
        prompt_details: dict[str, Any] = dict(input_details)
        cached = input_details.get("cached_tokens", input_details.get("cache_read_input_tokens"))
        if isinstance(cached, int) and not isinstance(cached, bool):
            prompt_details["cached_tokens"] = cached
        if prompt_details:
            result["prompt_tokens_details"] = prompt_details
    output_details = usage.get("output_tokens_details")
    if isinstance(output_details, dict) and output_details:
        result["completion_tokens_details"] = dict(output_details)
    return result


def _responses_tool_to_chat_function(tool: Any) -> dict[str, Any] | None:
    if not isinstance(tool, dict):
        return None
    if tool.get("type") == "function" and "function" in tool:
        return tool
    name = _responses_tool_function_name(tool)
    if not name:
        return None
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": tool.get("description") or _native_tool_description(tool),
            "parameters": tool.get("parameters") or _native_tool_parameters(tool),
        },
    }


def _responses_tool_function_name(tool: dict[str, Any]) -> str:
    fn = tool.get("function")
    if isinstance(fn, dict) and fn.get("name"):
        return _sanitize_tool_name(str(fn["name"]))
    if tool.get("name"):
        return _sanitize_tool_name(str(tool["name"]))
    tool_type = str(tool.get("type") or "").strip().lower()
    aliases = {
        "web_search": "web_search",
        "web_search_preview": "web_search",
        "computer_use": "computer_use",
        "computer_use_preview": "computer_use",
        "apply_patch": "apply_patch",
        "local_shell": "local_shell",
        "shell": "local_shell",
    }
    if tool_type in aliases:
        return aliases[tool_type]
    if tool_type.startswith("mcp"):
        return _sanitize_tool_name(tool_type)
    return ""


def _sanitize_tool_name(name: str) -> str:
    clean = re.sub(r"[^a-zA-Z0-9_-]+", "_", name.strip())[:64]
    return clean.strip("_") or "tool"


def _native_tool_description(tool: dict[str, Any]) -> str:
    tool_type = str(tool.get("type") or "tool")
    if tool_type.startswith("web_search"):
        return "Search the web using Codex's web-search tool fallback."
    if tool_type.startswith("computer_use"):
        return "Request a Codex computer-use action."
    if tool_type == "apply_patch":
        return "Apply a unified diff patch to the working tree."
    if tool_type in {"local_shell", "shell"}:
        return "Run a local shell command through Codex."
    if tool_type.startswith("mcp"):
        return "Interact with Codex MCP resources."
    return f"Codex tool fallback for Responses tool type {tool_type}."


def _native_tool_parameters(tool: dict[str, Any]) -> dict[str, Any]:
    tool_type = str(tool.get("type") or "").strip().lower()
    if tool_type.startswith("web_search"):
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
            },
            "required": ["query"],
            "additionalProperties": True,
        }
    if tool_type.startswith("computer_use"):
        return {
            "type": "object",
            "properties": {
                "action": {"type": "string", "description": "Computer action to perform"},
                "x": {"type": "number", "description": "Screen x coordinate, when relevant"},
                "y": {"type": "number", "description": "Screen y coordinate, when relevant"},
                "text": {"type": "string", "description": "Text to type, when relevant"},
            },
            "required": ["action"],
            "additionalProperties": True,
        }
    if tool_type == "apply_patch":
        return {
            "type": "object",
            "properties": {"patch": {"type": "string", "description": "Unified diff patch"}},
            "required": ["patch"],
            "additionalProperties": True,
        }
    if tool_type in {"local_shell", "shell"}:
        return {
            "type": "object",
            "properties": {"command": {"type": "string", "description": "Shell command to run"}},
            "required": ["command"],
            "additionalProperties": True,
        }
    return {"type": "object", "properties": {"input": {"type": "string"}}, "additionalProperties": True}


def _responses_tool_choice_to_chat(tool_choice: Any, tools: Any) -> Any:
    if tool_choice is None:
        return None
    if isinstance(tool_choice, str):
        if tool_choice in {"auto", "none", "required"}:
            return tool_choice
        name = _tool_choice_name(tool_choice, tools)
        return {"type": "function", "function": {"name": name}} if name else tool_choice
    if isinstance(tool_choice, dict):
        if tool_choice.get("type") == "function" and "function" in tool_choice:
            return tool_choice
        name = _tool_choice_name(str(tool_choice.get("name") or tool_choice.get("type") or ""), tools)
        return {"type": "function", "function": {"name": name}} if name else tool_choice
    return tool_choice


def _tool_choice_name(choice: str, tools: Any) -> str:
    choice = choice.lower().strip()
    if isinstance(tools, list):
        for tool in tools:
            if not isinstance(tool, dict):
                continue
            names = {
                str(tool.get("type") or "").lower(),
                str(tool.get("name") or "").lower(),
            }
            fn = tool.get("function")
            if isinstance(fn, dict):
                names.add(str(fn.get("name") or "").lower())
            if choice in names:
                return _responses_tool_function_name(tool)
    return _sanitize_tool_name(choice)


def _responses_tools_to_anthropic_tools(tools: Any) -> list[dict[str, Any]]:
    chat_tools = _responses_tools_to_chat_tools(tools)
    converted = []
    for tool in chat_tools:
        fn = tool.get("function") or {}
        converted.append(
            {
                "name": fn.get("name"),
                "description": fn.get("description", ""),
                "input_schema": fn.get("parameters") or {"type": "object", "properties": {}},
            }
        )
    return [tool for tool in converted if tool.get("name")]


def _anthropic_tools_to_chat_tools(tools: Any) -> list[dict[str, Any]]:
    if not isinstance(tools, list):
        return []
    converted: list[dict[str, Any]] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        name = tool.get("name")
        if not name:
            continue
        converted.append(
            {
                "type": "function",
                "function": {
                    "name": str(name),
                    "description": tool.get("description") or "",
                    "parameters": tool.get("input_schema") or {"type": "object", "properties": {}},
                },
            }
        )
    return converted


def _anthropic_tool_choice_to_chat(tool_choice: Any) -> Any:
    if tool_choice is None:
        return None
    if isinstance(tool_choice, str):
        if tool_choice in {"auto", "none"}:
            return tool_choice
        if tool_choice == "any":
            return "required"
        return {"type": "function", "function": {"name": tool_choice}}
    if isinstance(tool_choice, dict):
        choice_type = tool_choice.get("type")
        if choice_type in {"auto", "none"}:
            return choice_type
        if choice_type == "any":
            return "required"
        if choice_type == "tool" and tool_choice.get("name"):
            return {"type": "function", "function": {"name": str(tool_choice["name"])}}
    return tool_choice


def _chat_finish_to_anthropic_stop(reason: Any) -> str:
    if reason in {"tool_calls", "function_call"}:
        return "tool_use"
    if reason == "length":
        return "max_tokens"
    if reason == "content_filter":
        return "refusal"
    return "end_turn"


def _responses_usage_to_anthropic_usage(usage: dict[str, Any] | None) -> dict[str, Any] | None:
    if usage is None:
        return None
    result = {
        "input_tokens": int(usage.get("input_tokens") or 0),
        "output_tokens": int(usage.get("output_tokens") or 0),
    }
    input_details = usage.get("input_tokens_details")
    if isinstance(input_details, dict):
        cache_read = input_details.get("cache_read_input_tokens", input_details.get("cached_tokens"))
        if isinstance(cache_read, int) and not isinstance(cache_read, bool):
            result["cache_read_input_tokens"] = cache_read
        cache_created = input_details.get("cache_creation_input_tokens")
        if isinstance(cache_created, int) and not isinstance(cache_created, bool):
            result["cache_creation_input_tokens"] = cache_created
        # Convert the inclusive Responses total back to Anthropic's uncached
        # input count; cache reads and writes are separate Anthropic buckets.
        result["input_tokens"] = max(
            result["input_tokens"]
            - result.get("cache_read_input_tokens", 0)
            - result.get("cache_creation_input_tokens", 0),
            0,
        )
    return result


def _copy_if_present(src: dict[str, Any], dst: dict[str, Any], src_key: str, dst_key: str | None = None) -> None:
    if src_key in src and src[src_key] is not None:
        dst[dst_key or src_key] = src[src_key]


def _anthropic_stop(reason: Any) -> str:
    return "tool_calls" if reason == "tool_use" else "stop"


def _jsonish(value: Any) -> str:
    import json

    if isinstance(value, str):
        return value
    return json.dumps(value, separators=(",", ":"))


def _sanitize_string(value: str) -> str:
    value = value.replace("\x00", "")
    return "".join(char for char in value if char in "\n\r\t" or ord(char) >= 0x20)


def _sanitize_chat_content_parts(parts: list[Any]) -> list[dict[str, Any]]:
    cleaned = []
    for part in parts:
        if isinstance(part, str):
            cleaned.append({"type": "text", "text": _sanitize_string(part)})
            continue
        if not isinstance(part, dict):
            continue
        current = dict(part)
        if current.get("type") == "text":
            current["text"] = _sanitize_string(str(current.get("text", "")))
        elif current.get("type") == "image_url":
            image_url = current.get("image_url")
            if isinstance(image_url, dict):
                current["image_url"] = {k: _sanitize_string(str(v)) for k, v in image_url.items() if v is not None}
            elif isinstance(image_url, str):
                current["image_url"] = {"url": _sanitize_string(image_url)}
        cleaned.append(current)
    return cleaned


def _sanitize_chat_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cleaned = []
    for message in messages:
        current = dict(message)
        current.pop("_reasoning_only", None)
        current.pop("encrypted_content", None)
        current.pop("summary", None)
        role = current.get("role", "user")
        content = current.get("content")
        if content is None:
            current["content"] = None if role == "assistant" else ""
        elif isinstance(content, list):
            current["content"] = _sanitize_chat_content_parts(content)
        elif isinstance(content, str):
            current["content"] = _sanitize_string(content)
        else:
            current["content"] = _sanitize_string(_content_to_text(content))

        if isinstance(current.get("reasoning_content"), str):
            current["reasoning_content"] = _sanitize_string(current["reasoning_content"])
        tool_calls = current.get("tool_calls")
        if tool_calls:
            copied_calls = []
            for call in tool_calls:
                if not isinstance(call, dict):
                    continue
                copied_call = dict(call)
                if isinstance(copied_call.get("id"), str):
                    copied_call["id"] = _sanitize_string(copied_call["id"])
                function = copied_call.get("function")
                if isinstance(function, dict):
                    function = dict(function)
                    arguments = function.get("arguments")
                    if isinstance(arguments, str):
                        function["arguments"] = _sanitize_string(arguments)
                    copied_call["function"] = function
                copied_calls.append(copied_call)
            current["tool_calls"] = copied_calls
        tool_call_id = current.get("tool_call_id")
        if isinstance(tool_call_id, str):
            current["tool_call_id"] = _sanitize_string(tool_call_id)
        cleaned.append(current)
    return cleaned


def _normalize_chat_roles(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = []
    for message in messages:
        current = dict(message)
        if current.get("role") == "developer":
            current["role"] = "system"
        normalized.append(current)
    return normalized


def _merge_consecutive_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for message in messages:
        current = dict(message)
        role = current.get("role")
        if merged and role == merged[-1].get("role") and role in {"system", "user", "assistant"}:
            previous = merged[-1]
            previous["content"] = _merge_chat_content(previous.get("content"), current.get("content"))
            if role == "assistant":
                if current.get("reasoning_content") and not previous.get("reasoning_content"):
                    previous["reasoning_content"] = current["reasoning_content"]
                tool_calls = list(previous.get("tool_calls") or []) + list(current.get("tool_calls") or [])
                if tool_calls:
                    previous["tool_calls"] = tool_calls
            continue
        merged.append(current)
    return merged


def _merge_chat_content(left: Any, right: Any) -> Any:
    if not left:
        return right or ""
    if not right:
        return left
    if isinstance(left, list) or isinstance(right, list):
        merged: list[Any] = []
        merged.extend(left if isinstance(left, list) else [{"type": "text", "text": str(left)}])
        if merged and right:
            merged.append({"type": "text", "text": ""})
        merged.extend(right if isinstance(right, list) else [{"type": "text", "text": str(right)}])
        return merged
    return f"{left}\n\n{right}"
