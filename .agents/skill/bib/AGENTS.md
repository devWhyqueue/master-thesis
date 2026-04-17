# bib Agent Instructions

When working on the `bib` skill:

- Keep the package layout within the 7-direct-children limit per package directory.
- After every code change to `bib`, run this command from the repo root before making the next change:

```bash
uv run python /mnt/c/Users/Yannik/.codex/skills/clean-code/run.py --minimal --scope bib
```

- Treat a failing clean-code run as a blocker. Fix the reported issue, then rerun the same command immediately.
- Prefer feature-oriented modules under `bib/` over `internal`, `_impl`, or alias-based layouts.
