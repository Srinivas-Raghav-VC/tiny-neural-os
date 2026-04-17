---
allowed-tools: Bash(source .venv/bin/activate && marimo check --fix:*), Read, Edit, Write
---

## Context

This is the output of:

!`cd /mnt/d/alpha-proj && source .venv/bin/activate && marimo check --fix $ARGUMENTS || true`

## Your task

Only if the output above shows warnings or errors, read the notebook file in `$ARGUMENTS` and fix the issues.

After edits, re-run:

!`cd /mnt/d/alpha-proj && source .venv/bin/activate && marimo check $ARGUMENTS || true`

Do not make unrelated edits.
