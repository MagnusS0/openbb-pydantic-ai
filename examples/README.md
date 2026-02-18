# Examples

Minimal starters showing how to connect a [Pydantic AI](https://ai.pydantic.dev/) agent to [OpenBB Workspace](https://pro.openbb.co/app/workspace).

## Prerequisites

- Python 3.10+
- An API key for your model provider (e.g. `OPENAI_API_KEY` or `OPENROUTER_API_KEY`)
- Access to [OpenBB Workspace](https://pro.openbb.co/app/workspace)

## Simple example

A bare-bones agent with no extra toolsets.

```bash
uv sync
uv run uvicorn examples.simple_example:app --port 8003
```

Then in OpenBB Workspace, add a new agent with URL `http://localhost:8003`.

## Skills example

Extends the simple example with [pydantic-ai-skills](https://github.com/pydantic/pydantic-ai-skills) for progressive tool discovery. The agent can discover, load, and execute domain-specific skills at runtime.

Two bundled skills are included under `skills/`:

- **dcf-valuation** — Performs discounted cash flow analysis to estimate intrinsic value per share (adapted from [virattt/dexter](https://github.com/virattt/dexter/tree/main/src/skills)).
- **dbs-report** — Generates a DBS Group Research-style equity research report in HTML (adapted from [DidierRLopes/dbs-report-skill](https://github.com/DidierRLopes/dbs-report-skill/tree/main/.claude/skills/dbs-report))

```bash
uv sync
uv run uvicorn examples.skills_example:app --port 8003
```

Skills are loaded from the `examples/skills/` directory. To add your own, create a new subdirectory with a `SKILL.md` file — see the [pydantic-ai-skills docs](https://github.com/pydantic/pydantic-ai-skills) for the format.

## Configuration

Both examples read these environment variables:

| Variable | Default | Description |
|---|---|---|
| `AGENT_MODEL` | `openrouter:moonshotai/kimi-k2.5` | Pydantic AI model string (`provider:model`) |
| `AGENT_BASE_URL` | `http://localhost:8003` | Base URL advertised in `/agents.json` |

## Observability (optional)

To enable [Logfire](https://logfire.pydantic.dev/) tracing:

```bash
uv sync --extra observability
```

Then uncomment the two lines at the top of the example file:

```python
from examples.instrumentation import setup_observability
setup_observability()
```

To send traces to a custom OTEL backend (e.g. Langfuse, Jaeger) instead of Logfire:

```bash
export OTEL_EXPORTER_OTLP_ENDPOINT="http://localhost:4318"
```

```python
from examples.instrumentation import setup_observability
setup_observability(send_to_logfire=False)
```
