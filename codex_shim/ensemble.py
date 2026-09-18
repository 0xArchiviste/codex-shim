"""Jev ensemble mixes — fan out to named candidate sets, adjudicate, optionally merge.

A mix is a nicknamed proxy slug (e.g. ``cs-grokvastra`` / nickname ``grokvastra``)
that fans a request out to an arbitrary list of already-configured models, asks
Jev (OpenRouter Decisions / TypeSafe System One) which answer wins, then returns
that answer — or optionally synthesizes a merge via another generative model.

```
task ─► A, B, C … ─► Jev(winner, ship, merge?) ─► winner | merge(A,B)
```

Config lives in the top-level ``ensembles`` block of ``~/.codex-shim/models.json``.
"""

from __future__ import annotations

import json
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

DEFAULT_ADJUDICATOR_BASE = "https://openrouter.ai/api/alpha"
DEFAULT_ADJUDICATOR_MODEL = "~typesafe/jev-latest"
DEFAULT_TIMEOUT = 45.0
DEFAULT_MERGE_PROBABILITY = 0.5
VALID_MERGE_MODES = frozenset({"never", "always", "random", "jev"})

_SLUG_RE = re.compile(r"[^a-zA-Z0-9]+")


@dataclass(frozen=True)
class AdjudicatorConfig:
    base_url: str
    model: str
    api_key: str
    timeout: float
    extra_headers: dict[str, str]


@dataclass(frozen=True)
class EnsembleMix:
    slug: str
    nickname: str
    display_name: str
    candidates: tuple[str, ...]
    merge: str
    merge_model: Optional[str]
    merge_probability: float


@dataclass(frozen=True)
class EnsembleConfig:
    enabled: bool
    adjudicator: AdjudicatorConfig
    mixes: tuple[EnsembleMix, ...]

    @property
    def effective_enabled(self) -> bool:
        return self.enabled and not _env_flag("CODEX_SHIM_DISABLE_ENSEMBLE") and bool(self.mixes)


@dataclass(frozen=True)
class CandidateAnswer:
    slug: str
    text: str
    raw: dict[str, Any] | None = None
    error: str | None = None


@dataclass(frozen=True)
class AdjudicationResult:
    winner: str
    ship: float
    merge: bool
    raw_answers: dict[str, Any]
    model: str


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def ensemble_log_enabled() -> bool:
    return _env_flag("CODEX_SHIM_ENSEMBLE_LOG")


def slugify(value: str) -> str:
    slug = _SLUG_RE.sub("-", (value or "").strip().lower()).strip("-")
    return slug or "ensemble"


def load_ensemble_config(settings_path: Path | str, byok_models: list[Any] | None = None) -> Optional[EnsembleConfig]:
    """Parse the optional top-level ``ensembles`` block from settings JSON."""
    path = Path(settings_path).expanduser()
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    raw = data.get("ensembles")
    if not isinstance(raw, dict):
        return None

    adjudicator = _parse_adjudicator(raw.get("adjudicator") or {}, byok_models or [])
    if adjudicator is None or not adjudicator.api_key:
        return None

    mixes: list[EnsembleMix] = []
    for row in raw.get("mixes") or []:
        mix = _parse_mix(row)
        if mix is not None:
            mixes.append(mix)
    if not mixes:
        return None

    enabled = bool(raw.get("enabled", True))
    return EnsembleConfig(enabled=enabled, adjudicator=adjudicator, mixes=tuple(mixes))


def _parse_adjudicator(raw: dict[str, Any], byok_models: list[Any]) -> Optional[AdjudicatorConfig]:
    if not isinstance(raw, dict):
        raw = {}
    base_url = str(raw.get("base_url") or raw.get("baseUrl") or DEFAULT_ADJUDICATOR_BASE).rstrip("/")
    model = str(raw.get("model") or DEFAULT_ADJUDICATOR_MODEL).strip()
    api_key_env = str(raw.get("api_key_env") or raw.get("apiKeyEnv") or "OPENROUTER_API_KEY").strip()
    api_key = str(raw.get("api_key") or raw.get("apiKey") or "").strip()
    if not api_key and api_key_env:
        api_key = os.environ.get(api_key_env, "").strip()
    if not api_key:
        # Reuse credentials from a configured OpenRouter BYOK row (e.g. cs-jev).
        for m in byok_models:
            base = str(getattr(m, "base_url", "") or "").lower()
            if "openrouter.ai" in base and getattr(m, "api_key", ""):
                api_key = str(m.api_key).strip()
                break
    if not api_key:
        try:
            api_key = (Path.home() / ".codex-shim" / "openrouter-api-key").read_text().strip()
        except OSError:
            api_key = ""
    timeout = _as_float(raw.get("timeout"), DEFAULT_TIMEOUT)
    extra = {
        str(k): str(v)
        for k, v in (raw.get("extra_headers") or raw.get("extraHeaders") or {}).items()
        if v is not None
    }
    extra.setdefault("HTTP-Referer", "https://localhost")
    extra.setdefault("X-Title", "codex-shim-ensemble")
    return AdjudicatorConfig(
        base_url=base_url,
        model=model,
        api_key=api_key,
        timeout=timeout,
        extra_headers=extra,
    )


def _parse_mix(row: Any) -> Optional[EnsembleMix]:
    if not isinstance(row, dict):
        return None
    candidates_raw = row.get("candidates") or row.get("models") or []
    if isinstance(candidates_raw, str):
        candidates = [c.strip() for c in candidates_raw.split(",") if c.strip()]
    elif isinstance(candidates_raw, list):
        candidates = [str(c).strip() for c in candidates_raw if str(c).strip()]
    else:
        return None
    if len(candidates) < 2:
        return None

    nickname = str(row.get("nickname") or row.get("name") or "").strip()
    slug = str(row.get("slug") or "").strip()
    if not slug and nickname:
        slug = nickname if nickname.startswith("cs-") else f"cs-{slugify(nickname)}"
    if not slug:
        slug = "cs-" + slugify("-".join(candidates[:2]))
    if not nickname:
        nickname = slug.removeprefix("cs-") if slug.startswith("cs-") else slug
    display = str(row.get("display_name") or row.get("displayName") or nickname).strip()
    merge = str(row.get("merge") or "random").strip().lower()
    if merge not in VALID_MERGE_MODES:
        merge = "random"
    merge_model = str(row.get("merge_model") or row.get("mergeModel") or "").strip() or None
    merge_probability = _as_float(row.get("merge_probability") or row.get("mergeProbability"), DEFAULT_MERGE_PROBABILITY)
    merge_probability = min(1.0, max(0.0, merge_probability))
    return EnsembleMix(
        slug=slugify(slug) if not slug.startswith("cs-") else slug,
        nickname=nickname,
        display_name=display,
        candidates=tuple(candidates),
        merge=merge,
        merge_model=merge_model,
        merge_probability=merge_probability,
    )


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def find_mix(config: EnsembleConfig, requested: str) -> Optional[EnsembleMix]:
    needle = (requested or "").strip()
    if not needle:
        return None
    for mix in config.mixes:
        if needle in {mix.slug, mix.nickname}:
            return mix
    return None


def active_mixes(config: EnsembleConfig, available: set[str]) -> list[EnsembleMix]:
    """Mixes whose candidate set intersects available models (≥2 usable)."""
    if not config.effective_enabled:
        return []
    out: list[EnsembleMix] = []
    for mix in config.mixes:
        usable = [c for c in mix.candidates if c in available]
        if len(usable) >= 2:
            out.append(mix)
    return out


def filter_candidates(mix: EnsembleMix, available: set[str]) -> list[str]:
    return [c for c in mix.candidates if c in available]


def extract_task_text(body: dict[str, Any], limit: int = 6000) -> str:
    """Best-effort plain-text task summary for Jev ``state``."""
    chunks: list[str] = []
    instructions = body.get("instructions")
    if isinstance(instructions, str) and instructions.strip():
        chunks.append(instructions.strip())

    messages = body.get("messages")
    if isinstance(messages, list):
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            role = str(msg.get("role") or "")
            text = _content_to_text(msg.get("content"))
            if text:
                chunks.append(f"{role}: {text}" if role else text)

    items = body.get("input")
    if isinstance(items, list):
        for item in items:
            if isinstance(item, str) and item.strip():
                chunks.append(item.strip())
            elif isinstance(item, dict):
                text = _content_to_text(item.get("content") or item.get("text") or item.get("input_text"))
                if not text and item.get("type") == "message":
                    text = _content_to_text(item.get("content"))
                if text:
                    role = str(item.get("role") or item.get("type") or "input")
                    chunks.append(f"{role}: {text}")

    if not chunks:
        prompt = body.get("prompt")
        if isinstance(prompt, str) and prompt.strip():
            chunks.append(prompt.strip())

    text = "\n\n".join(chunks).strip() or "(empty task)"
    return text[:limit]


def _content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                for key in ("text", "input_text", "output_text", "content"):
                    val = part.get(key)
                    if isinstance(val, str) and val.strip():
                        parts.append(val)
                        break
        return "\n".join(p.strip() for p in parts if p and p.strip())
    if isinstance(content, dict):
        return _content_to_text(content.get("text") or content.get("content"))
    return str(content).strip()


def extract_assistant_text(payload: dict[str, Any]) -> str:
    """Pull assistant-visible text from chat, Responses, or Anthropic shapes."""
    if not isinstance(payload, dict):
        return ""
    # OpenAI chat
    choices = payload.get("choices")
    if isinstance(choices, list) and choices:
        message = (choices[0] or {}).get("message") or {}
        text = _content_to_text(message.get("content"))
        if text:
            return text
        tool_calls = message.get("tool_calls") or []
        if tool_calls:
            return json.dumps({"tool_calls": tool_calls}, indent=2)[:4000]
    # Responses API
    output = payload.get("output")
    if isinstance(output, list):
        parts: list[str] = []
        for item in output:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "message":
                parts.append(_content_to_text(item.get("content")))
            elif item.get("type") in {"function_call", "custom_tool_call", "web_search_call"}:
                parts.append(json.dumps(item, indent=2)[:1500])
        text = "\n".join(p for p in parts if p)
        if text:
            return text
    if isinstance(payload.get("output_text"), str):
        return payload["output_text"].strip()
    # Anthropic
    content = payload.get("content")
    if isinstance(content, list):
        text = _content_to_text(content)
        if text:
            return text
    return ""


def build_decisions_request(
    adjudicator: AdjudicatorConfig,
    task: str,
    answers: list[CandidateAnswer],
    ask_merge: bool,
) -> dict[str, Any]:
    usable = [a for a in answers if a.text and not a.error]
    labels = {a.slug: f"candidate_{i}" for i, a in enumerate(usable)}
    state = {
        "task": task[:4000],
        "candidates": {
            labels[a.slug]: {"slug": a.slug, "answer": a.text[:3500]} for a in usable
        },
    }
    criteria = {labels[a.slug]: f"Prefer the answer from {a.slug}" for a in usable}
    questions: dict[str, Any] = {
        "winner": {
            "type": "choice",
            "instructions": (
                "Which candidate best solves the task? Prefer correctness, completeness, "
                "and actionable detail over verbosity."
            ),
            "criteria": criteria,
        },
        "ship": {
            "type": "noul",
            "instructions": "Is the winning answer good enough to ship without further editing?",
        },
    }
    if ask_merge and len(usable) >= 2:
        questions["should_merge"] = {
            "type": "noul",
            "instructions": (
                "Would combining the top answers produce a meaningfully better result than "
                "the single winner alone?"
            ),
        }
    return {"model": adjudicator.model, "state": state, "questions": questions, "_label_map": labels}


def parse_adjudication(
    decisions_payload: dict[str, Any],
    label_map: dict[str, str],
    answers: list[CandidateAnswer],
    merge_mode: str,
    merge_probability: float,
    rng: random.Random | None = None,
) -> AdjudicationResult:
    """Map Jev Decisions answers onto a concrete winner slug + merge flag."""
    rng = rng or random.Random()
    usable = [a for a in answers if a.text and not a.error]
    reverse = {label: slug for slug, label in label_map.items()}
    raw_answers = decisions_payload.get("answers") if isinstance(decisions_payload, dict) else {}
    if not isinstance(raw_answers, dict):
        raw_answers = {}

    winner_label = ""
    winner_block = raw_answers.get("winner") or {}
    if isinstance(winner_block, dict):
        winner_label = str(winner_block.get("choice") or "").strip()
    winner = reverse.get(winner_label, "")
    if winner not in {a.slug for a in usable}:
        winner = usable[0].slug if usable else (answers[0].slug if answers else "")

    ship = 0.0
    ship_block = raw_answers.get("ship") or {}
    if isinstance(ship_block, dict):
        try:
            ship = float(ship_block.get("noul") or 0.0)
        except (TypeError, ValueError):
            ship = 0.0

    merge = False
    if merge_mode == "always":
        merge = len(usable) >= 2
    elif merge_mode == "never":
        merge = False
    elif merge_mode == "random":
        merge = len(usable) >= 2 and rng.random() < merge_probability
    elif merge_mode == "jev":
        merge_block = raw_answers.get("should_merge") or {}
        try:
            merge = float(merge_block.get("noul") or 0.0) >= 0.5 and len(usable) >= 2
        except (TypeError, ValueError):
            merge = False

    return AdjudicationResult(
        winner=winner,
        ship=ship,
        merge=merge,
        raw_answers=raw_answers,
        model=str(decisions_payload.get("model") or ""),
    )


def decisions_url(adjudicator: AdjudicatorConfig) -> str:
    base = adjudicator.base_url.rstrip("/")
    if base.endswith("/decisions"):
        return base
    # Accept either https://openrouter.ai/api/alpha or .../alpha/decisions
    parsed = urlparse(base)
    if parsed.path.rstrip("/").endswith("decisions"):
        return base
    return f"{base}/decisions"


def merge_prompt(task: str, primary: CandidateAnswer, secondary: CandidateAnswer) -> str:
    return (
        "Merge the two candidate solutions into one better answer for the task. "
        "Keep the strongest parts of each, remove contradictions, and do not mention "
        "that you are merging.\n\n"
        f"## Task\n{task[:4000]}\n\n"
        f"## Candidate {primary.slug}\n{primary.text[:5000]}\n\n"
        f"## Candidate {secondary.slug}\n{secondary.text[:5000]}\n"
    )


def pick_secondary(answers: list[CandidateAnswer], winner: str) -> Optional[CandidateAnswer]:
    usable = [a for a in answers if a.text and not a.error and a.slug != winner]
    return usable[0] if usable else None


def catalog_entry(mix: EnsembleMix) -> dict[str, Any]:
    return {
        "slug": mix.slug,
        "display_name": mix.display_name,
        "description": (
            f"Jev ensemble mix ({mix.nickname}): "
            + ", ".join(mix.candidates)
            + f"; merge={mix.merge}."
        ),
        "context_window": 200_000,
        "max_context_window": 200_000,
        "auto_compact_token_limit": 160_000,
        "truncation_policy": {"mode": "tokens", "limit": 64_000},
        "default_reasoning_level": "medium",
        "supported_reasoning_levels": [
            {"effort": "low", "description": "Faster, lighter reasoning"},
            {"effort": "medium", "description": "Balanced speed and reasoning"},
            {"effort": "high", "description": "Deeper reasoning"},
        ],
        "default_reasoning_summary": "none",
        "reasoning_summary_format": "none",
        "supports_reasoning_summaries": False,
        "default_verbosity": "low",
        "support_verbosity": False,
        "apply_patch_tool_type": "freeform",
        "web_search_tool_type": "text_and_image",
        "supports_search_tool": False,
        "supports_parallel_tool_calls": True,
        "experimental_supported_tools": [],
        "input_modalities": ["text", "image"],
        "supports_image_detail_original": True,
        "shell_type": "shell_command",
        "visibility": "list",
        "minimal_client_version": "0.0.1",
        "supported_in_api": True,
        "availability_nux": None,
        "upgrade": None,
        "priority": 8500,
        "prefer_websockets": False,
        "available_in_plans": ["free", "plus", "pro", "team", "business", "enterprise"],
        "base_instructions": (
            f"You are Codex routed through the {mix.display_name} Jev ensemble mix."
        ),
        "model_messages": {
            "instructions_template": (
                "You are Codex running through the {model_name} ensemble "
                "(multi-model fan-out + Jev adjudication)."
            ),
            "instructions_variables": {"model_name": mix.display_name},
        },
        "ensemble": {
            "nickname": mix.nickname,
            "candidates": list(mix.candidates),
            "merge": mix.merge,
        },
    }


def models_entry(mix: EnsembleMix, created: int) -> dict[str, Any]:
    return {
        "id": mix.slug,
        "object": "model",
        "created": created,
        "owned_by": "codex-shim-ensemble",
        "nickname": mix.nickname,
        "candidates": list(mix.candidates),
    }
