"""Offline regressions for Cursor-style Grep tool turns."""
import json

import pytest
from aiohttp.test_utils import TestClient, TestServer
from codex_shim import server
from codex_shim.claude_passthrough import build_claude_prompt


def test_exact_tool_name_and_empty_result_history():
    prompt = build_claude_prompt({
        "tools": [{"type": "function", "name": "functions.Grep", "parameters": {"type": "object"}}],
        "input": [
            {"type": "function_call", "name": "functions.Grep", "call_id": "call_search", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "call_search", "output": ""},
        ],
    })
    assert '"name": "functions.Grep"' in prompt
    assert "functions_Grep" not in prompt
    assert '"call_id": "call_search"' in prompt
    assert "[TOOL call_search]\n[empty tool result]" in prompt
    assert "None" not in prompt


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_grep_calls_have_unique_ids_and_history_survives(tmp_path, monkeypatch, stream):
    monkeypatch.setattr(server, "claude_passthrough_available", lambda: True)
    prompts = []
    async def events(prompt, slug):
        prompts.append(prompt)
        yield {"type": "completed", "text": '```codex-shim-tool\n{"tool_calls":[{"name":"Grep","arguments":{"pattern":"handler","path":"src"}}]}\n```'}
    monkeypatch.setattr(server, "iter_claude_agent_events", events)
    settings = tmp_path / "models.json"
    settings.write_text('{"models":[]}')
    body = {"model": "cd-opus-5-5-medium", "stream": stream,
            "messages": [{"role": "user", "content": "Find handler"}],
            "tools": [{"type": "function", "function": {"name": "Grep", "parameters": {
                "type": "object", "properties": {"pattern": {"type": "string"}, "path": {"type": "string"}},
                "required": ["pattern"]}}}]}
    async with TestClient(TestServer(server.ShimServer(settings).app())) as client:
        ids = []
        for turn in range(2):
            response = await client.post('/v1/chat/completions', json=body)
            assert response.status == 200
            if stream:
                raw = await response.text()
                chunks = [json.loads(line[6:]) for line in raw.splitlines() if line.startswith('data: ') and line != 'data: [DONE]']
                calls = [call for chunk in chunks for choice in chunk.get('choices', [])
                         for call in choice.get('delta', {}).get('tool_calls', [])]
                assert any(choice.get('finish_reason') == 'tool_calls' for chunk in chunks for choice in chunk.get('choices', []))
                call = calls[0]
                call.pop('index', None)
            else:
                data = await response.json()
                assert data['choices'][0]['finish_reason'] == 'tool_calls'
                call = data['choices'][0]['message']['tool_calls'][0]
            assert call['function']['name'] == 'Grep'
            assert json.loads(call['function']['arguments']) == {'pattern': 'handler', 'path': 'src'}
            ids.append(call['id'])
            body['messages'].extend([{'role': 'assistant', 'content': None, 'tool_calls': [call]},
                                     {'role': 'tool', 'tool_call_id': call['id'], 'content': 'src/main.py:12: handler'}])
        assert ids[0] != ids[1]
        assert 'src/main.py:12: handler' in prompts[1]
        assert ids[0] in prompts[1]
