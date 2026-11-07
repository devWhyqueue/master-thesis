# Repository Guidelines

**Status:** Active
**Last Updated:** 2026-03-13

**IMPORTANT:** Follow documentation rules in [CONTRIBUTING.md](CONTRIBUTING.md) - especially the file creation and naming conventions.

## Project Overview

`notebooklm-py` is an unofficial Python client for Google NotebookLM that uses undocumented RPC APIs. The library enables programmatic automation of NotebookLM features including notebook management, source integration, AI querying, and studio artifact generation (podcasts, videos, quizzes, etc.).

**Critical constraint**: This uses Google's internal `batchexecute` RPC protocol with obfuscated method IDs that Google can change at any time. All RPC method IDs in `src/notebooklm/rpc/types.py` are undocumented and subject to breakage.

## Project Structure & Module Organization

`src/notebooklm/` contains the async client and typed APIs. Internal feature modules use `_` prefixes such as `_sources.py` and `_artifacts.py`; `src/notebooklm/cli/` holds Click commands, and `src/notebooklm/rpc/` handles protocol encoding and decoding. Tests are split by scope: `tests/unit/`, `tests/integration/`, and `tests/e2e/`. Recorded HTTP fixtures live in `tests/cassettes/`. Examples are in `docs/examples/`, and diagnostics live in `scripts/`.

## Build, Test, and Development Commands

Use `uv` for local work:

```bash
uv sync --extra dev --extra browser
uv run pytest
uv run ruff check src/ tests/
uv run ruff format src/ tests/
uv run mypy src/notebooklm
uv run pre-commit run --all-files
```

Run `uv run pytest tests/e2e -m readonly` only after `notebooklm login` and setting test notebook env vars.

Alternative pip-style setup (relocatable venv) and extra commands:

```bash
uv venv .venv
uv pip install -e ".[all]"
playwright install chromium   # browser binary for `notebooklm login`
source .venv/bin/activate
pytest --cov                  # tests with coverage (e2e excluded by default)
pytest tests/e2e -m e2e       # e2e tests (requires authentication)
notebooklm --help             # CLI smoke test
```

## Coding Style & Naming Conventions

Target Python 3.10+, 4-space indentation, and double quotes. Ruff enforces formatting and import order with a 100-character line length. Keep module and test file names in `snake_case`; prefer descriptive Click command names that match existing groups such as `source`, `artifact`, and `research`. Preserve the internal/public split: `_*.py` for implementation, exported types in `src/notebooklm/__init__.py`.

## Testing Guidelines

Put pure logic in `tests/unit/`, VCR-backed flows in `tests/integration/`, and authenticated NotebookLM coverage in `tests/e2e/`. Name tests `test_<behavior>.py` and record cassettes with `NOTEBOOKLM_VCR_RECORD=1 uv run pytest tests/integration/test_vcr_*.py -v`. Coverage is expected to stay at or above the configured 90% threshold.

## Commit, PR, and Agent Notes

Follow the existing commit style: `feat(cli): ...`, `fix(cli): ...`, `refactor(test): ...`, `style: ...`. PRs should include a short summary, linked issue when relevant, and the commands run locally. For Codex or other parallel agents, prefer `--json`, pass explicit notebook IDs instead of relying on `notebooklm use`, and isolate runs with `NOTEBOOKLM_HOME=/tmp/<agent-id>` when multiple agents share one machine.

## Pre-Commit Checks (REQUIRED before committing)

**IMPORTANT:** Always run these checks before committing to avoid CI failures:

```bash
# Format code with ruff
ruff format src/ tests/

# Check for linting issues
ruff check src/ tests/

# Type checking with mypy
mypy src/notebooklm --ignore-missing-imports

# Run tests
pytest
```

Or use this one-liner:
```bash
ruff format src/ tests/ && ruff check src/ tests/ && mypy src/notebooklm --ignore-missing-imports && pytest
```

## Architecture

### Layered Design

```
CLI Layer (cli/)
    ↓
Client Layer (client.py, _*.py APIs)
    ↓
Core Layer (_core.py)
    ↓
RPC Layer (rpc/)
```

1. **RPC Layer** (`src/notebooklm/rpc/`):
   - `types.py`: All RPC method IDs and enums (source of truth)
   - `encoder.py`: Request encoding
   - `decoder.py`: Response parsing

2. **Core Layer** (`src/notebooklm/_core.py`):
   - HTTP client management
   - RPC call abstraction
   - Request counter handling

3. **Client Layer** (`src/notebooklm/client.py`, `_*.py`):
   - `NotebookLMClient`: Main async client with namespaced APIs
   - `_notebooks.py`, `_sources.py`, `_artifacts.py`, etc.: Domain APIs

4. **CLI Layer** (`src/notebooklm/cli/`):
   - Modular Click commands
   - `session.py`, `notebook.py`, `source.py`, `generate.py`, etc.

### Key Files

| File | Purpose |
|------|---------|
| `client.py` | Main `NotebookLMClient` class |
| `_core.py` | HTTP and RPC infrastructure |
| `_notebooks.py` | `client.notebooks` API |
| `_sources.py` | `client.sources` API |
| `_artifacts.py` | `client.artifacts` API |
| `_chat.py` | `client.chat` API |
| `rpc/types.py` | RPC method IDs (source of truth) |
| `auth.py` | Authentication handling |
| `cli/` | CLI command modules |

### Repository Structure

```
src/notebooklm/
├── __init__.py          # Public exports
├── client.py            # NotebookLMClient
├── auth.py              # Authentication
├── types.py             # Dataclasses
├── _core.py             # Core infrastructure
├── _notebooks.py        # NotebooksAPI
├── _sources.py          # SourcesAPI
├── _artifacts.py        # ArtifactsAPI
├── _chat.py             # ChatAPI
├── _research.py         # ResearchAPI
├── _notes.py            # NotesAPI
├── rpc/                 # RPC protocol layer
│   ├── types.py         # Method IDs and enums
│   ├── encoder.py       # Request encoding
│   └── decoder.py       # Response parsing
└── cli/                 # CLI implementation
    ├── __init__.py
    ├── helpers.py       # Shared utilities
    ├── session.py       # login, use, status, clear
    ├── notebook.py      # list, create, delete, rename
    ├── source.py        # source add, list, delete
    ├── artifact.py      # artifact commands
    ├── generate.py      # generate audio, video, etc.
    ├── download.py      # download commands
    ├── chat.py          # ask, configure, history
    └── note.py          # note commands
```

## API Patterns

### Client Usage

```python
# Correct pattern - uses namespaced APIs
async with await NotebookLMClient.from_storage() as client:
    notebooks = await client.notebooks.list()
    await client.sources.add_url(nb_id, url)
    result = await client.chat.ask(nb_id, question)
    status = await client.artifacts.generate_audio(nb_id)
```

### CLI Structure

Commands are organized as:
- **Top-level**: `login`, `use`, `status`, `clear`, `list`, `create`, `ask`
- **Grouped**: `source add`, `artifact list`, `generate audio`, `download video`, `note create`

## Testing Strategy

- **Unit tests** (`tests/unit/`): Test encoding/decoding, no network
- **Integration tests** (`tests/integration/`): Mock HTTP responses
- **E2E tests** (`tests/e2e/`): Real API, require auth, marked `@pytest.mark.e2e`

### E2E Test Status

- ✅ Notebook operations (list, create, rename, delete)
- ✅ Source operations (add URL/text/YouTube, rename)
- ✅ Download operations (audio, video, infographic, slides)
- ⚠️ Artifact generation may fail due to rate limiting

## Common Pitfalls

1. **RPC method IDs change**: Check network traffic and update `rpc/types.py`
2. **Nested list structures**: Params are position-sensitive. Check existing implementations.
3. **Source ID nesting**: Different methods need `[id]`, `[[id]]`, `[[[id]]]`, or `[[[[id]]]]`
4. **CSRF tokens expire**: Use `client.refresh_auth()` or re-run `notebooklm login`
5. **Rate limiting**: Add delays between bulk operations

## Documentation

All docs use lowercase-kebab naming in `docs/`:
- `docs/cli-reference.md` - CLI commands
- `docs/python-api.md` - Python API reference
- `docs/configuration.md` - Storage and settings
- `docs/troubleshooting.md` - Known issues
- `docs/development.md` - Architecture, testing, releasing
- `docs/rpc-development.md` - RPC capture and debugging
- `docs/rpc-reference.md` - RPC payload structures

## When to Suggest CLI vs API

- **CLI**: Quick tasks, shell scripts, LLM agent automation
- **Python API**: Application integration, complex workflows, async operations

## Pull Request Workflow (REQUIRED)

After creating a PR, you MUST monitor and address feedback:

### 1. Monitor CI Status
```bash
# Check CI status (repeat until all pass)
gh pr checks <PR_NUMBER>
```

Wait for all checks to pass. If any fail, investigate and fix.

### 2. Check for Review Comments
```bash
# Get review comments
gh api repos/teng-lin/notebooklm-py/pulls/<PR_NUMBER>/comments \
  --jq '.[] | "File: \(.path):\(.line)\nComment: \(.body)\n---"'
```

### 3. Address Feedback
For each review comment (especially from `gemini-code-assist`):
1. Read and understand the feedback
2. Make the suggested fix if it improves the code
3. Commit with a descriptive message referencing the feedback
4. Push and re-check CI
5. **Reply to the review thread** confirming the fix:
   ```bash
   gh api repos/teng-lin/notebooklm-py/pulls/<PR>/comments/<COMMENT_ID>/replies \
     -f body="Addressed in commit <SHA>: <brief description>"
   ```

### 4. Verify Final State
```bash
# Ensure PR is ready to merge
gh pr view <PR_NUMBER> --json state,mergeStateStatus,mergeable
```

**Important**: Do NOT consider a PR complete until:
- All CI checks pass
- All review comments are addressed
- `mergeStateStatus` is `CLEAN`
