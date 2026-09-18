# Jev ensembles — parked / later work

Status: **earmarked for a later pass**. The current ensemble path works for
small prompts; large Codex/agent transcripts exceed Jev’s context budget and
need a dedicated judge strategy before we lean on this for production mixes.

## What exists today

- Config: top-level `ensembles` block in `~/.codex-shim/models.json`
- Code: `codex_shim/ensemble.py` + server fan-out / OpenRouter Decisions
- Example mix: `cs-grokvastra` / nickname `grokvastra`
  (`cs-grok-4-6` + `cs-astra-light`, `merge: random`)
- Adjudicator: OpenRouter `POST /api/alpha/decisions` with
  `~typesafe/jev-latest` (not chat/completions — that path 500s for Jev)
- Env: `CODEX_SHIM_DISABLE_ENSEMBLE`, `CODEX_SHIM_ENSEMBLE_LOG`

Jev limits (approx.): **32k tokens for state + longest question**, **64k for
the whole request**. Today we blunt-truncate (`task[:4000]`, each answer
`[:3500]` chars), which is lossy on real agent turns.

## Parked: large-input adjudication

Problem: fan-out answers + task can be far larger than 32k, so a single Choice
over full transcripts is not viable.

Recommended options to implement later (low cost first):

1. **`score_each`** — one Jev call per candidate
   (`state = {task_digest, answer_digest}` → score/ship), then pick the max
   (optional final Choice over top 2).
2. **Compress then Choice** — cheap generative digests (≤~800–1200 tokens each)
   via a configured `compressor` slug, then one Jev Choice over digests.
3. **Pairwise tournament** — A vs B, winner vs C… when N is small.
4. **Structured cards** — diffs/file lists/tool names/test exits instead of
   full prose; return the winner’s full answer as-is.
5. **Two-stage filter** — drop empty/failed heuristically, Jev only on survivors.

Suggested future mix config shape (not implemented):

```json
"judge": {
  "mode": "score_each",
  "compressor": "cs-astra-light",
  "task_tokens": 1500,
  "answer_tokens": 1200,
  "max_candidates": 4
}
```

Default direction when we pick this up: **`score_each` + optional light
compress** when packed size would exceed ~24k tokens; keep today’s single
Choice for small prompts.

## Do not expand until then

- No new judge modes, compressors, or tournament logic without revisiting this
  doc.
- Mix nicknames in `models.json` remain fine to add; they still use the current
  blunt-truncation judge.
