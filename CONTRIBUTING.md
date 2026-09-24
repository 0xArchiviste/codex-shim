# Contributing to codex-shim

Thanks for hacking on the shim. Issues and PRs welcome.

## Dev loop

```bash
git clone https://github.com/0xSero/codex-shim
cd codex-shim
python3 -m pip install -e ".[dev]"

python3 -m pytest tests/ -q
python3 -m compileall codex_shim/ -q
```

CI runs the same commands on Python 3.11 and 3.12 via
`.github/workflows/ci.yml`. Match it locally before opening a PR.

## Agent deployment directive

When a task changes runtime behavior and asks to update the live reverse-BYOK
shim, agents should preserve the current endpoint and credentials:

1. Read the live process command line, working directory, PID file, configured
   settings path, listening port, and service manager ownership before acting.
   Check `systemctl --user status codex-shim.service` on Linux.
2. Regenerate and gracefully restart the daemon on that same port with
   `codex-shim --settings <existing-settings> --port <existing-port> restart`.
   The CLI delegates to an active `codex-shim.service`; never stop/start a
   second process against a `Restart=always` unit. Repair a stale/missing PID
   file only after verifying the actual listener or service `MainPID`.
3. Do not restart ngrok when it already forwards to that port; this preserves
   its public URL. Do not rotate `CODEX_SHIM_API_KEY`.
4. Verify local `/health`, authenticated local `/v1/models`, the ngrok target,
   and authenticated public `/v1/models` after the handoff.
5. Run tests and `git diff --check` before committing repository changes. Never
   commit files under `~/.codex-shim`, API keys, tunnel credentials, or runtime
   PID/log files.

The current reverse-BYOK convention uses port `8766`, but always inspect the
running deployment instead of assuming it.

## Claude Code transport safety

- API requests must start isolated headless CLI sessions, never attach to or
  resume the user's interactive Claude session.
- Use an explicit executable/profile when crossing WSL/Windows boundaries.
  Claude desktop and Claude Code CLI are different executables; a desktop
  process is not evidence of CLI login or model entitlement.
- Never extract desktop credentials or copy OAuth tokens into repository files.
- Keep local tools disabled for the text-only adapter. Reject unsupported
  client tool requests explicitly; do not turn them into host-side execution.
- Verify authentication and a live model response before claiming a `cd-*`
  deployment works. Unit tests with fake CLI output are not entitlement checks.
- UltraCode workflow support requires separate tool protocol validation; adding
  model aliases alone does not provide Claude Code's orchestration runtime.

## What kinds of changes are useful

- Translation fixes for tricky tool-call / reasoning streams, with a
  captured fixture under `tests/` proving the bug and the fix.
- New provider translations (e.g. a new chat-completions or
  Anthropic-shaped upstream). Add a test that exercises the new shape end
  to end through `ShimServer`, the way `test_server.py` does.
- Compatibility notes / safer detection for new Codex Desktop builds,
  especially around the ASAR picker patch needle in
  `codex_shim/cli.py::patch_codex_app`.
- Doc patches that name a specific build / version. "I tested on Codex
  Desktop 0.x.y on macOS arm64 and it did Z" is more useful than a
  generic warning.

## Code style

- Match the surrounding file. No new dependencies without a reason.
- Keep `codex_shim/server.py` translation behavior covered by tests in
  `tests/test_server.py` or `tests/test_translate.py` — tool-call shape
  bugs are easy to miss by eyeballing streams.
- Don't include API keys, ChatGPT access tokens, or `auth.json` contents
  in fixtures, logs, or test data. Use synthetic tokens (`"stub"`,
  `"secret"`) like the existing tests.

## Reporting bugs

Please include:

- Codex Desktop / CLI version (`codex --version` and the Desktop About
  panel).
- OS (macOS arm64 / x86_64 / Linux distro / WSL).
- Output of `codex-shim status` and the last ~80 lines of
  `.codex-shim/shim.log` with API keys redacted.
- Whether the model is a configured BYOK/upstream entry or the `gpt-5.5`
  ChatGPT passthrough.
- Minimal repro: the exact `codex-shim …` invocation and what you
  expected vs. what happened.

## Security

Don't open public issues for security problems. Email or DM the
maintainer with details and a repro and we'll coordinate a fix.
