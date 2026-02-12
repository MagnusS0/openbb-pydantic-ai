# Test Conventions

This repository uses pytest with a refactored unit test layout:

- `tests/unit/adapter`
- `tests/unit/event_stream`
- `tests/unit/pdf`
- `tests/unit/mcp`
- `tests/unit/core`
- `tests/unit/widgets`
- `tests/smoke`

## Core Rules

- Keep tests atomic: each test should verify one behavior.
- Keep imports at module scope; no in-function imports.
- Async tests rely on `asyncio_mode = "auto"`; do not use `@pytest.mark.anyio`.
- Use snapshots only for complex structured payloads where they improve readability.
- Convert models/dataclasses to builtins before snapshotting (`tests/helpers/snapshot_utils.py`).
- Use `dirty-equals` only for dynamic fields (IDs, timestamps, similar non-deterministic values).

## Regression Contract

The adapter and event-stream tests are marked with `regression_contract` and run as a CI gate before the full suite.

Run locally:

```bash
uv run pytest -q -m regression_contract tests/unit/adapter tests/unit/event_stream
```

## Common Commands

Run all unit tests:

```bash
uv run pytest -q tests --ignore=tests/smoke
```

Run snapshot lifecycle commands:

```bash
uv run pytest --inline-snapshot=create tests --ignore=tests/smoke
uv run pytest --inline-snapshot=fix tests --ignore=tests/smoke
```

Run style guardrails for tests:

```bash
uv run python scripts/check_test_style.py
```
