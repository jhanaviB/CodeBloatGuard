# CodeBloatGuard

# CodeBloatGuard

[![Tests](https://github.com/jhanaviB/CodeBloatGuard/actions/workflows/bloatguard.yml/badge.svg)](https://github.com/jhanaviB/CodeBloatGuard/actions)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Finds duplicated logic in a Python repo and flags it during PR review.


Finds duplicated logic in a Python repo and flags it during PR review. Retrieval finds candidates by meaning (not shared words), a judge model decides.

## The Problem This Solves

**Before CodeBloatGuard:**
- Developers add duplicate functions without knowing similar code exists
- Code reviews catch some duplicates, miss others
- `grep` finds exact matches, misses semantic duplicates
- Lots of time spent by team to review code

**After CodeBloatGuard:**
- Automatic duplicate detection on every PR
- Semantic search finds similar logic, not just matching words
- LLM judge explains *why* code is duplicate
- Fast enough for CI (processes 9,790 functions in 0.63s)

**Real example:**
```python
# These are duplicates, but grep wouldn't find them
def factorial(n):
    r = 1
    for i in range(2, n + 1):
        r *= i
    return r

def fact(x):
    return 1 if x <= 1 else x * fact(x - 1)


## Setup

```bash
pip install -e .
echo 'GEMINI_API_KEY=your-key-here' > .env
```

Tests need neither:

```bash
pip install -e '.[dev]'
pytest
```

## Usage

```bash
cbg index .                                  # build the index
cbg check src/yourmodule/thing.py --repo .   # fast path, fixed threshold
cbg review src/yourmodule/thing.py --repo .  # agentic path, model picks the radius
```

Both exit 1 when duplicates are found.

| Command | |
| --- | --- |
| `cbg index <repo>` | Reconcile the store with disk. Idempotent. |
| `cbg check <file>` | Flag code that exists elsewhere. Fixed threshold. |
| `cbg review <file>` | Same, but the model decides search width. |
| `cbg check-pr` | Files a PR changed. Fast path, escalating near misses. |
| `cbg chunk <repo>` | Show the function split. No API calls. |
| `cbg search <code>` | Nearest indexed functions for a snippet. |
| `cbg judge <a> <b>` | Verdict on two snippets, no retrieval. |
| `cbg eval` | Score the judge on 14 labeled pairs. |
| `cbg benchmark <repo>...` | Chunker throughput. No API calls. |
| `cbg stats` | Chunks indexed. |

`--no-judge` prints retrieval distances without calling any model. Use it to calibrate the threshold on a new repo.

## How it works

`chunker` (tree-sitter) splits files into functions → `embedder` (Gemini) embeds them → `store` (ChromaDB) does cosine retrieval. Chunk ids hash content, so unrelated edits don't re-embed.

**`check`**: everything under `DUP_DISTANCE` goes to the judge (DUPLICATE / SIMILAR / DIFFERENT + fix). Fast, deterministic — good for CI.

**`review`**: a LangGraph loop — retrieve → triage (judge / widen / stop) → judge (parallel, 4 at a time) → conventions. Triage widens retrieval when the fixed threshold doesn't fit; conventions checks naming/style precedent against neighbors, skipped only when a DUPLICATE is being replaced outright.

**`check-pr`** runs both: fast path first, escalates to the graph only when a match falls in the ambiguous band between `DUP_DISTANCE` and `ESCALATE_DISTANCE`.

API errors always return ERROR, never a false pass. Daily quota errors abort the run instead of retrying (retrying can't restore a quota that resets tomorrow).

## Models

Default is `gemini-2.5-flash` (free tier: 20 generation calls/day). Change per-stage in `config.py` if you have billing.

Optional local alternative via Claude Agent SDK (uses your Claude Pro login, no API cost):

```bash
pip install claude-agent-sdk
CBG_PROVIDER=claude cbg eval
```

Not usable in CI (no local Claude session there) — CI stays on a Gemini API key.

## Observability and evals

```bash
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your-key
```

`cbg eval` scores the judge against 14 hard labeled pairs in `evalset.py`. `--upload` pushes to LangSmith.

## Tests

```bash
pytest   # 75 tests, no key, no network, ~1s
```

Embedder is faked as a bag-of-tokens vector; Chroma runs for real against a temp dir.

## CI

`.github/workflows/bloatguard.yml`: `tests` runs pytest on every push/PR. `check` runs `cbg check-pr` on changed files. `reindex` runs on merge to main to refresh the shared store/cache.

Add `GEMINI_API_KEY` as a repo secret and merge to main once before opening a PR — without a baseline, `check` skips with a notice instead of failing.

## Measured on real repos

Chunking throughput (free, no API calls):

| repo | files | lines | functions | seconds |
| --- | ---: | ---: | ---: | ---: |
| [fastapi](https://github.com/fastapi/fastapi) | 1,130 | 111,711 | 4,897 | 0.37 |
| [click](https://github.com/pallets/click) | 78 | 28,104 | 1,705 | 0.10 |
| [flask](https://github.com/pallets/flask) | 83 | 18,337 | 1,437 | 0.05 |
| [httpie/cli](https://github.com/httpie/cli) | 133 | 19,002 | 1,063 | 0.06 |
| [requests](https://github.com/psf/requests) | 37 | 12,032 | 688 | 0.04 |
| **total** | **1,461** | **189,186** | **9,790** | **0.63** |

~2,300 files/sec — matters because a re-index only embeds what changed, so the "what changed" walk must be cheap enough to run every commit.

On `psf/requests`, retrieval alone doesn't separate signal: real code has no clean distance gap like the toy fixture does, so every reported finding goes through the judge, and the threshold needs per-repo calibration (`--no-judge`).

## Distributing

Not yet published.

```bash
python -m pip install build twine
python -m build
python -m twine upload dist/*
```

Blocker: `PROJECT_ROOT` in `config.py` resolves relative to the installed package, so a non-editable install puts the store inside site-packages. Needs to derive from the indexed repo instead.

## Known limits

- Python only, functions only.
- `DUP_DISTANCE` (0.30) / `ESCALATE_DISTANCE` (0.45) calibrated on a small fixture — verify on real repos.
- Escalation makes PR cost non-deterministic.
- Two identically-named, identical-body functions in one file collide on chunk id; only one is stored.
- `cbg search` embeds a bare snippet vs. indexed documents' `# file:`/`# symbol:` header — distances aren't directly comparable to `cbg check`.
- Judging isn't prioritized; a PR that exhausts quota may not have judged its closest pairs.
- Local and CI indexes are separate; local results are advisory.
- Store is a local Chroma directory (swap for a Chroma server in `store.py`).

## Technical Skills Demonstrated

This project showcases:

**AI/ML Engineering:**
- Semantic code search using embeddings
- LLM integration (Gemini, Claude) with structured outputs
- Agentic workflows with LangGraph
- Prompt engineering for code analysis

**Software Engineering:**
- CLI tool development (tree-sitter, argparse)
- Vector database integration (ChromaDB)
- Idempotent data pipelines (chunk ID hashing)
- Concurrent processing (ThreadPoolExecutor)

**DevOps/MLOps:**
- GitHub Actions CI/CD
- Workflow caching strategies
- API quota management
- LangSmith observability integration

**Testing & Validation:**
- 75 unit tests with pytest
- Labeled evaluation dataset (14 hard cases)
- Performance benchmarking on real repos
- No mocking - uses real ChromaDB in tests

**Product Thinking:**
- Cost optimization (free tier: 20 calls/day)
- Failure modes handled (quota exceeded, network errors)
- Progressive enhancement (fast path + escalation)
- User-friendly error messages