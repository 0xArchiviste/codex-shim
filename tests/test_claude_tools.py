"""Pure offline tests: no server, CLI, authentication, or provider calls."""
import json

import pytest

from codex_shim.claude_tools import (
    claude_tool_definitions,
    render_claude_tool_definitions,
    validate_claude_tool_reply,
)


SCHEMA = {
    "type": "object",
    "properties": {
        "pattern": {"type": "string", "minLength": 1},
        "output_mode": {"enum": ["content", "files_with_matches", "count"]},
        "head_limit": {"type": "integer", "minimum": 0},
    },
    "required": ["pattern"],
    "additionalProperties": False,
}


def body(name="Grep", *, chat=False, schema=SCHEMA, **policy):
    fn = {"name": name, "description": "Search", "parameters": schema}
    tool = {"type": "function", "function": fn} if chat else {"type": "function", **fn}
    return {"tools": [tool], **policy}


def reply(arguments=None, name="Grep", rows=None):
    rows = rows if rows is not None else [{"name": name, "arguments": arguments if arguments is not None else {"pattern": "x"}}]
    return "```codex-shim-tool\n" + json.dumps({"tool_calls": rows}) + "\n```"


@pytest.mark.parametrize("chat", [False, True])
@pytest.mark.parametrize("name", ["Grep", "functions.Grep", "mcp.server/search", "x" * 80])
def test_exact_names_and_regex_roundtrip(chat, name):
    request = body(name, chat=chat)
    args = {"pattern": r'foo\s+"bar"|日本語\\path', "output_mode": "content"}
    calls = validate_claude_tool_reply(reply(args, name), request)
    assert calls == [{"name": name, "arguments": json.dumps(args, sort_keys=True, ensure_ascii=False)}]
    assert json.loads(calls[0]["arguments"]) == args
    assert json.loads(render_claude_tool_definitions(request))["name"] == name
    assert claude_tool_definitions(request)[0]["parameters"] == SCHEMA


@pytest.mark.parametrize("args", [{}, {"pattern": 123}, {"pattern": ""}, {"pattern": "x", "output_mode": "bad"}, {"pattern": "x", "head_limit": -1}, {"pattern": "x", "head_limit": True}, {"pattern": "x", "unknown": True}, [], "{}", None])
def test_schema_and_object_validation(args):
    with pytest.raises(ValueError):
        validate_claude_tool_reply(reply(rows=[{"name": "Grep", "arguments": args}]), body())


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity", "1e999"])
def test_nonstandard_and_overflow_json_rejected(constant):
    text = '```codex-shim-tool\n{"tool_calls":[{"name":"Grep","arguments":{"pattern":' + constant + '}}]}\n```'
    with pytest.raises(ValueError):
        validate_claude_tool_reply(text, body(schema={}))


def test_duplicate_keys_rejected():
    text = '```codex-shim-tool\n{"tool_calls":[{"name":"Grep","arguments":{"pattern":"a","pattern":"b"}}]}\n```'
    with pytest.raises(ValueError):
        validate_claude_tool_reply(text, body())


@pytest.mark.parametrize("choice", ["auto", "none"])
def test_ordinary_text(choice):
    assert validate_claude_tool_reply("Here is the answer.", body(tool_choice=choice)) == []
    assert validate_claude_tool_reply("```python\nprint(1)\n```", {}) == []


@pytest.mark.parametrize("choice", ["required", {"type": "function", "name": "Grep"}, {"type": "function", "function": {"name": "Grep"}}])
def test_required_and_forced(choice):
    request = body(tool_choice=choice)
    assert validate_claude_tool_reply(reply(), request)
    with pytest.raises(ValueError):
        validate_claude_tool_reply("No tools needed.", request)


def test_none_forced_and_parallel_policy():
    with pytest.raises(ValueError):
        validate_claude_tool_reply(reply(), body(tool_choice="none"))
    request = body(tool_choice={"type": "function", "name": "Grep"})
    request["tools"] += body("Read")["tools"]
    with pytest.raises(ValueError):
        validate_claude_tool_reply(reply(name="Read"), request)
    rows = [{"name": "Grep", "arguments": {"pattern": p}} for p in ("a", "b")]
    assert len(validate_claude_tool_reply(reply(rows=rows), body())) == 2
    with pytest.raises(ValueError):
        validate_claude_tool_reply(reply(rows=rows), body(parallel_tool_calls=False))


@pytest.mark.parametrize("text", ["```codex-shim-tool\n{", reply(rows=[]), reply(name="Unknown"), '```codex-shim-tool\nnot json\n```'])
def test_malformed_attempt_is_not_prose(text):
    with pytest.raises(ValueError):
        validate_claude_tool_reply(text, body())


@pytest.mark.parametrize("bad", [None, {"name": "Unknown", "arguments": {}}, {"name": ["Grep"], "arguments": {}}, {"name": "Grep", "arguments": {}}, {"name": "Grep"}])
def test_atomic_batch(bad):
    rows = [{"name": "Grep", "arguments": {"pattern": "x"}}, bad]
    with pytest.raises(ValueError):
        validate_claude_tool_reply(reply(rows=rows), body())


def test_full_schema_features_and_local_refs():
    schema = {"$defs": {"term": {"type": "string", "pattern": "^ok$"}}, "type": "object", "properties": {"pattern": {"$ref": "#/$defs/term"}}, "required": ["pattern"], "oneOf": [{"required": ["a"]}, {"required": ["b"]}]}
    assert validate_claude_tool_reply(reply({"pattern": "ok", "a": 1}), body(schema=schema))
    for args in ({"pattern": "bad", "a": 1}, {"pattern": "ok", "a": 1, "b": 1}):
        with pytest.raises(ValueError):
            validate_claude_tool_reply(reply(args), body(schema=schema))


def test_remote_refs_never_retrieved(monkeypatch):
    import urllib.request
    def forbidden(*args, **kwargs):
        pytest.fail("Network retrieval attempted")
    monkeypatch.setattr(urllib.request, "urlopen", forbidden)
    with pytest.raises(ValueError):
        validate_claude_tool_reply(reply(), body(schema={"$ref": "https://example.invalid/schema"}))


@pytest.mark.parametrize("schema", [{"type": "not-a-type"}, {"$schema": "https://example.invalid/dialect"}])
def test_bad_schemas(schema):
    with pytest.raises(ValueError):
        claude_tool_definitions(body(schema=schema))


def test_safe_errors_hide_private_arguments():
    with pytest.raises(ValueError) as caught:
        validate_claude_tool_reply(reply({"pattern": "private-secret", "unknown": 1}), body())
    assert "private-secret" not in str(caught.value)
    assert "unknown" not in str(caught.value)


def test_common_model_tool_formats():
    assert validate_claude_tool_reply("I need to search.\n" + reply() + "\nDone.", body())
    encoded = reply(rows=[{"id": "call_1", "name": "Grep", "arguments": json.dumps({"pattern": "handler"})}])
    assert json.loads(validate_claude_tool_reply(encoded, body())[0]["arguments"]) == {"pattern": "handler"}


def test_duplicate_names_and_unsupported_native_tools():
    request = body()
    request["tools"] *= 2
    with pytest.raises(ValueError):
        claude_tool_definitions(request)
    with pytest.raises(ValueError):
        claude_tool_definitions({"tools": [{"type": "web_search"}]})
