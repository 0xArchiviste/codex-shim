from __future__ import annotations

import json
from pathlib import Path

from codex_shim.ensemble import (
    AdjudicatorConfig,
    CandidateAnswer,
    EnsembleMix,
    build_decisions_request,
    extract_assistant_text,
    extract_task_text,
    find_mix,
    load_ensemble_config,
    parse_adjudication,
    slugify,
)


def test_slugify_basic():
    assert slugify("Grok v Astra") == "grok-v-astra"


def test_load_ensemble_config(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    path = tmp_path / "models.json"
    path.write_text(
        json.dumps(
            {
                "models": [],
                "ensembles": {
                    "enabled": True,
                    "adjudicator": {"api_key_env": "OPENROUTER_API_KEY"},
                    "mixes": [
                        {
                            "nickname": "grokvastra",
                            "display_name": "Grok v Astra",
                            "candidates": ["cs-grok-4-6", "cs-astra-light"],
                            "merge": "random",
                            "merge_model": "cs-grok-4-6",
                            "merge_probability": 0.5,
                        }
                    ],
                },
            }
        )
    )
    config = load_ensemble_config(path)
    assert config is not None
    assert config.effective_enabled
    mix = find_mix(config, "grokvastra")
    assert mix is not None
    assert mix.slug == "cs-grokvastra"
    assert mix.candidates == ("cs-grok-4-6", "cs-astra-light")
    assert find_mix(config, "cs-grokvastra") is mix


def test_extract_task_from_messages():
    body = {
        "messages": [
            {"role": "system", "content": "Be helpful"},
            {"role": "user", "content": "Fix the bug in foo()"},
        ]
    }
    text = extract_task_text(body)
    assert "Fix the bug" in text
    assert "Be helpful" in text


def test_extract_assistant_text_chat_and_responses():
    chat = {
        "choices": [{"message": {"role": "assistant", "content": "hello from chat"}}]
    }
    assert extract_assistant_text(chat) == "hello from chat"
    responses = {
        "output": [
            {
                "type": "message",
                "content": [{"type": "output_text", "text": "hello from responses"}],
            }
        ]
    }
    assert extract_assistant_text(responses) == "hello from responses"


def test_parse_adjudication_winner_and_merge_modes():
    answers = [
        CandidateAnswer(slug="cs-grok-4-6", text="A"),
        CandidateAnswer(slug="cs-astra-light", text="B"),
    ]
    label_map = {"cs-grok-4-6": "candidate_0", "cs-astra-light": "candidate_1"}
    payload = {
        "model": "typesafe/jev-1.13",
        "answers": {
            "winner": {"type": "choice", "choice": "candidate_1"},
            "ship": {"type": "noul", "noul": 0.82},
            "should_merge": {"type": "noul", "noul": 0.9},
        },
    }
    never = parse_adjudication(payload, label_map, answers, "never", 0.5)
    assert never.winner == "cs-astra-light"
    assert never.ship == 0.82
    assert never.merge is False

    always = parse_adjudication(payload, label_map, answers, "always", 0.5)
    assert always.merge is True

    class R:
        def random(self):
            return 0.1

    random_yes = parse_adjudication(payload, label_map, answers, "random", 0.5, rng=R())  # type: ignore[arg-type]
    assert random_yes.merge is True

    jev = parse_adjudication(payload, label_map, answers, "jev", 0.5)
    assert jev.merge is True


def test_build_decisions_request_shapes_choice_criteria():
    adj = AdjudicatorConfig(
        base_url="https://openrouter.ai/api/alpha",
        model="~typesafe/jev-latest",
        api_key="x",
        timeout=10,
        extra_headers={},
    )
    answers = [
        CandidateAnswer(slug="cs-grok-4-6", text="answer A"),
        CandidateAnswer(slug="cs-astra-light", text="answer B"),
    ]
    req = build_decisions_request(adj, "task text", answers, ask_merge=True)
    assert req["model"] == "~typesafe/jev-latest"
    assert "winner" in req["questions"]
    assert "should_merge" in req["questions"]
    assert set(req["questions"]["winner"]["criteria"]) == {"candidate_0", "candidate_1"}
