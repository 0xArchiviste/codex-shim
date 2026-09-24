"""Opt-in Jev IO profiles and auxiliary provider configuration."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .ensemble import AdjudicatorConfig, _parse_adjudicator, decisions_url

IO_PROFILE_SLUG = "cs-sol-jev-io"
IO_MAX_PROFILE_SLUG = "cs-sol-jev-io-max"
VALID_ROLLOUTS = frozenset({"off", "shadow", "active"})

# (public base alias, display name). Each receives ``-jev-io`` and
# ``-jev-io-max`` profiles. The aliases themselves are defined by their
# transport modules; this table only layers IO behavior over them.
CURSOR_IO_BASES: tuple[tuple[str, str], ...] = (
    ("cx-auto", "Cursor Auto"),
    ("cx-grok-4-7", "Cursor Grok 4.7 High"),
    ("cx-fable-5-1", "Cursor Fable 5.1 Medium"),
    ("cx-fable-5-1-high", "Cursor Fable 5.1 High"),
    ("cx-fable-5", "Cursor Fable 5 Medium"),
    ("cx-fable-5-high", "Cursor Fable 5 High"),
    ("cx-opus-5", "Cursor Opus 5 High Thinking"),
    ("cx-opus-5-5", "Cursor Opus 5.5"),
    ("cx-sol-5-6", "Cursor GPT-5.6 Sol Medium"),
    ("cx-sol-5-6-high", "Cursor GPT-5.6 Sol High"),
)


@dataclass(frozen=True)
class RewriteConfig:
    model: str
    base_url: str
    api_key: str
    timeout: float = 45.0
    extra_headers: dict[str, str] | None = None


@dataclass(frozen=True)
class IOProfile:
    slug: str
    display_name: str
    base_model: str = "cs-sol-medium"
    rollout: str = "shadow"
    rewrite_input: bool = False
    rewrite_output: bool = False
    min_chars: int = 1500
    block_lines: int = 25
    drop_threshold: float = 0.10
    keep_threshold: float = 0.70
    min_prune_ratio: float = 0.10
    max_judge_chars: int = 48_000
    max_internal_rounds: int = 3
    retrieval_roots: tuple[str, ...] = ()
    retrieval_results: int = 8
    candidates: tuple[str, ...] = ()


@dataclass(frozen=True)
class IOConfig:
    enabled: bool
    adjudicator: AdjudicatorConfig | None
    rewriter: RewriteConfig | None
    profiles: tuple[IOProfile, ...]

    @property
    def effective_enabled(self) -> bool:
        return self.enabled and not _env_flag("CODEX_SHIM_DISABLE_JEV_IO")


class RelevanceJudge(Protocol):
    async def judge(self, *, task: str, blocks: list[dict[str, Any]]) -> dict[str, float]: ...


class OpenRouterDecisionsJudge:
    """Small adapter around OpenRouter's alpha Decisions endpoint."""

    def __init__(self, config: AdjudicatorConfig, post_json):
        self.config = config
        self._post_json = post_json

    async def judge(self, *, task: str, blocks: list[dict[str, Any]]) -> dict[str, float]:
        state = {"task": task, "blocks": {b["id"]: b["text"] for b in blocks}}
        questions = {
            b["id"]: {
                "type": "noul",
                "instructions": (
                    "Will this block be needed to correctly complete the task? "
                    "Treat errors, constraints, definitions, and evidence as needed."
                ),
            }
            for b in blocks
        }
        payload = await self._post_json(
            decisions_url(self.config),
            {"model": self.config.model, "state": state, "questions": questions},
            self.config,
        )
        answers = payload.get("answers") if isinstance(payload, dict) else {}
        result: dict[str, float] = {}
        if isinstance(answers, dict):
            for block_id, answer in answers.items():
                if not isinstance(answer, dict):
                    continue
                try:
                    result[str(block_id)] = float(answer.get("noul"))
                except (TypeError, ValueError):
                    continue
        return result


def load_io_config(settings_path: Path | str, byok_models: list[Any] | None = None) -> IOConfig:
    raw: dict[str, Any] = {}
    path = Path(settings_path).expanduser()
    try:
        data = json.loads(path.read_text())
        if isinstance(data, dict) and isinstance(data.get("jev_io"), dict):
            raw = data["jev_io"]
    except (OSError, json.JSONDecodeError):
        pass

    adjudicator = _parse_adjudicator(raw.get("adjudicator") or {}, byok_models or [])
    rewriter = _parse_rewriter(raw.get("rewriter") or {}, byok_models or [])
    defaults = raw.get("defaults") if isinstance(raw.get("defaults"), dict) else {}
    rows = raw.get("profiles") if isinstance(raw.get("profiles"), list) else []
    configured = {str(row.get("slug") or ""): row for row in rows if isinstance(row, dict)}
    specs: list[tuple[str, str, str, bool, tuple[str, ...], str]] = [
        (IO_PROFILE_SLUG, "Codex Sol Jev IO", "cs-sol-medium", False, (), "shadow"),
        (IO_MAX_PROFILE_SLUG, "Codex Sol Jev IO Max", "cs-sol-medium", True, (), "shadow"),
    ]
    for base_slug, display_name in CURSOR_IO_BASES:
        specs.extend(
            (
                (
                    f"{base_slug}-jev-io",
                    f"{display_name} Jev IO",
                    base_slug,
                    False,
                    (),
                    "shadow",
                ),
                (
                    f"{base_slug}-jev-io-max",
                    f"{display_name} Jev IO Max",
                    base_slug,
                    True,
                    (),
                    "shadow",
                ),
            )
        )
    specs.append(
        (
            "cx-autogrok",
            "Cursor Auto × Grok 4.7 (Jev IO)",
            "",
            False,
            ("cx-auto", "cx-grok-4-7"),
            "active",
        )
    )
    profiles = [
        _parse_profile(
            slug,
            display,
            base_model,
            maximum,
            configured.get(slug),
            defaults,
            candidates,
            default_rollout,
        )
        for slug, display, base_model, maximum, candidates, default_rollout in specs
    ]
    # Permit additional explicit profiles without requiring a code change.
    known = {profile.slug for profile in profiles}
    for slug, row in configured.items():
        if slug and slug not in known:
            profiles.append(
                _parse_profile(
                    slug,
                    str(row.get("display_name") or slug),
                    str(row.get("base_model") or ""),
                    bool(row.get("rewrite_input") or row.get("rewrite_output")),
                    row,
                    defaults,
                    (),
                    "shadow",
                )
            )
    # Profiles are opt-in: an absent ``jev_io`` block must not change discovery
    # or model counts in existing installations.
    return IOConfig(bool(raw.get("enabled", bool(raw))), adjudicator, rewriter, tuple(profiles))


def find_profile(config: IOConfig, slug: str) -> IOProfile | None:
    if not config.effective_enabled:
        return None
    return next((p for p in config.profiles if p.slug == slug and p.rollout != "off"), None)


def active_profiles(
    config: IOConfig, available: set[str] | None = None
) -> list[IOProfile]:
    profiles = [
        p for p in config.profiles if config.effective_enabled and p.rollout != "off"
    ]
    if available is None:
        return profiles
    return [
        profile
        for profile in profiles
        if (
            profile.base_model in available
            if profile.base_model
            else bool(profile.candidates)
            and all(candidate in available for candidate in profile.candidates)
        )
    ]


def models_entry(profile: IOProfile, created: int) -> dict[str, Any]:
    return {"id": profile.slug, "object": "model", "created": created, "owned_by": "codex-shim-jev-io"}


def catalog_entry(profile: IOProfile) -> dict[str, Any]:
    context_window = 1_000_000 if profile.slug.startswith("cx-") else 400_000
    return {
        "slug": profile.slug,
        "display_name": profile.display_name,
        "description": (
            f"{profile.display_name} with reversible Jev IO processing "
            f"({profile.rollout})."
        ),
        "context_window": context_window,
        "max_context_window": context_window,
        "auto_compact_token_limit": int(context_window * 0.8),
        "truncation_policy": {"mode": "tokens", "limit": 64_000},
        "default_reasoning_level": "medium",
        "supported_reasoning_levels": [{"effort": x, "description": x} for x in ("low", "medium", "high", "xhigh")],
        "supports_parallel_tool_calls": True,
        "input_modalities": ["text", "image"],
        "visibility": "list",
        "supported_in_api": True,
        "available_in_plans": ["free", "plus", "pro", "team", "business", "enterprise"],
    }


def _parse_profile(
    slug: str,
    display: str,
    base_model: str,
    maximum: bool,
    row: Any,
    defaults: dict[str, Any],
    candidates: tuple[str, ...],
    default_rollout: str,
) -> IOProfile:
    values = dict(defaults)
    if isinstance(row, dict):
        values.update(row)
    rollout = str(
        values.get("rollout")
        or os.environ.get("CODEX_SHIM_JEV_IO_ROLLOUT")
        or default_rollout
    ).lower()
    if rollout not in VALID_ROLLOUTS:
        rollout = "off"
    roots = values.get("retrieval_roots") or values.get("retrievalRoots") or []
    if isinstance(roots, str):
        roots = [roots]
    raw_candidates = values.get("candidates", candidates)
    if isinstance(raw_candidates, str):
        raw_candidates = [x.strip() for x in raw_candidates.split(",") if x.strip()]
    if not isinstance(raw_candidates, (list, tuple)):
        raw_candidates = candidates
    return IOProfile(
        slug=slug,
        display_name=str(values.get("display_name") or display),
        base_model=str(values.get("base_model") or base_model),
        rollout=rollout,
        rewrite_input=bool(values.get("rewrite_input", maximum)),
        rewrite_output=bool(values.get("rewrite_output", maximum)),
        min_chars=_integer(values.get("min_chars"), 1500),
        block_lines=_integer(values.get("block_lines"), 25),
        drop_threshold=_number(values.get("drop_threshold"), 0.10),
        keep_threshold=_number(values.get("keep_threshold"), 0.70),
        min_prune_ratio=_number(values.get("min_prune_ratio"), 0.10),
        max_judge_chars=_integer(values.get("max_judge_chars"), 48_000),
        max_internal_rounds=_integer(values.get("max_internal_rounds"), 3),
        retrieval_roots=tuple(str(Path(x).expanduser()) for x in roots if str(x).strip()),
        retrieval_results=_integer(values.get("retrieval_results"), 8),
        candidates=tuple(str(x) for x in raw_candidates if str(x).strip()),
    )


def _parse_rewriter(raw: Any, models: list[Any]) -> RewriteConfig | None:
    if not isinstance(raw, dict):
        return None
    model = str(raw.get("model") or "").strip()
    if not model:
        return None
    base_url = str(raw.get("base_url") or raw.get("baseUrl") or "https://openrouter.ai/api/v1").rstrip("/")
    key = str(raw.get("api_key") or "").strip()
    env = str(raw.get("api_key_env") or "OPENROUTER_API_KEY")
    if not key:
        key = os.environ.get(env, "").strip()
    if not key:
        for item in models:
            if "openrouter.ai" in str(getattr(item, "base_url", "")) and getattr(item, "api_key", ""):
                key = str(item.api_key)
                break
    if not key:
        return None
    return RewriteConfig(model, base_url, key, _number(raw.get("timeout"), 45.0), dict(raw.get("extra_headers") or {}))


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _integer(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _number(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
