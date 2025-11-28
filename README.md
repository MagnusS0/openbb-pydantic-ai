[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![ty](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ty/main/assets/badge/v0.json)](https://github.com/astral-sh/ty)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/MagnusS0/openbb-pydantic-ai)


# OpenBB Pydantic AI Adapter

`openbb-pydantic-ai` lets any [Pydantic AI](https://ai.pydantic.dev/) agent
run behind OpenBB Workspace by translating `QueryRequest` payloads into a Pydantic
AI run, exposing Workspace widgets as deferred tools, and streaming native
OpenBB SSE events back to the UI.

- **Stateless by design**: each `QueryRequest` already carries the full
  conversation history, widgets, context, and URLs, so the adapter can process
  requests independently.
- **First-class widget tools**: every widget becomes a deferred Pydantic AI tool;
  when the model calls one, the adapter emits `copilotFunctionCall` events via
  `get_widget_data` and waits for the Workspace to return data before resuming.
- **Rich event stream**: reasoning steps, “Thinking“ traces, tables, charts, and
  citations are streamed as OpenBB SSE payloads so the Workspace UI can group
  them into dropdowns automatically.
- **Output helpers included**: structured model outputs (dicts/lists) are
  auto-detected and turned into tables or charts, with chart parameter
  normalization to ensure consistent rendering.

To learn more about the underlying SDK types, see the
[OpenBB Custom Agent SDK repo](https://github.com/OpenBB-finance/openbb-ai)
and the [Pydantic AI UI adapter docs](https://ai.pydantic.dev/ui/overview/).

## Installation

The adapter is published as a lightweight package, install it wherever you build
custom agents:

```bash
pip install openbb-pydantic-ai
# or with uv
uv add openbb-pydantic-ai
```

## Quick Start (FastAPI)

```python
from fastapi import FastAPI, Request
from pydantic_ai import Agent
from openbb_pydantic_ai import OpenBBAIAdapter, OpenBBDeps

agent = Agent(
    "openai:gpt-5",
    instructions="Be concise and helpful. Only use widget tools for data lookups.",
    deps_type=OpenBBDeps,
)
app = FastAPI()

@app.post("/query")
async def query(request: Request):
    return await OpenBBAIAdapter.dispatch_request(request, agent=agent)
```

### How It Works

#### 1. Request Handling

- OpenBB Workspace calls the `/query` endpoint with a `QueryRequest` body
- `OpenBBAIAdapter` validates it and builds the Pydantic AI message stack
- Workspace context and URLs are injected as system prompts

#### 2. Widget Tool Conversion

- Widgets in the request become deferred tools
- Each call emits a `copilotFunctionCall` event (via `get_widget_data`)
- The adapter pauses until Workspace responds with data

#### 3. Event Streaming

Pydantic AI events are wrapped into OpenBB SSE events:

- **Text chunks** → stream via `copilotMessageChunk`
- **Reasoning steps** → appear under the "Step-by-step reasoning" dropdown (including Thinking sections)
- **Tables/charts** → emitted as `copilotMessageArtifact` events with correct chart parameters for consistent rendering
- **Citations** → fire at the end of the run for every widget tool used

### Advanced Usage

Need more control? Instantiate the adapter manually:

```python
from openbb_pydantic_ai import OpenBBAIAdapter

run_input = OpenBBAIAdapter.build_run_input(body_bytes)
adapter = OpenBBAIAdapter(agent=agent, run_input=run_input)
async for event in adapter.run_stream():
    yield event  # Already encoded as OpenBB SSE payloads
```

You can also supply `message_history`, `deferred_tool_results`, or `on_complete`
callbacks—any option supported by `Agent.run_stream_events()` is accepted.

**Runtime deps & prompts**

- `OpenBBDeps` bundles widgets (grouped by priority), context rows, relevant
  URLs, workspace state, timezone, and a serialized `state` dict you can pass to
  toolsets or output validators.
- The adapter merges dashboard context and current widget parameter values into
  the runtime instructions automatically; append your own instructions without
  re-supplying that context.

## Features

### Widget Toolsets

- Widgets are grouped by priority (`primary`, `secondary`, `extra`) and exposed
  through dedicated toolsets so you can gate access if needed.
- Tool names start with `openbb_widget_` plus the widget identifier; any
  redundant `openbb_` prefix from the origin is trimmed so names stay concise
  (e.g., `openbb_widget_sandbox_financial_statements`). Use
  `build_widget_tool_name` to reproduce the routing string exactly.

### MCP Tools

- Any tools listed in `QueryRequest.tools` are exposed as a external
  MCP toolset, so the model can call the same names the Workspace UI presents.
- Deferred `execute_agent_tool` results replay on the next request just like
  widget results, keeping multi-turn streaming consistent.

### Deferred Results & Citations

- Pending widget responses provided in the request are replayed before the run
  starts, making multi-turn workflows seamless.
- Every widget call records a citation via `openbb_ai.helpers.cite`, emitted as a
  `copilotCitationCollection` at the end of the run.

### Structured Output Detection

The adapter provides built-in helpers and automatic detection for charts and tables:

- **Markdown tables** - Stream tabular data as Markdown; Workspace renders them as tables and lets users promote them to widgets.
- **`openbb_create_chart`** - Create chart artifacts (line, bar, scatter, pie, donut) with validation. Insert `{{place_chart_here}}` in the response where the chart should appear; the adapter swaps that placeholder for the rendered artifact while streaming.
- **Auto-detection** - Dict/list outputs shaped like `{"type": "table", "data": [...]}` (or just a list of dicts) automatically become tables.
- **Flexible chart parameters** - Chart outputs tolerate different field spellings (`y_keys`, `yKeys`, etc.) and validate required axes before emitting.
- **`openbb_create_table`** - Explicitly create a table artifact from structured data when you want predictable column ordering and metadata.

## Local Development

This repo ships a UV-based workflow:

```bash
uv sync --dev         # install dependencies
uv run pytest      # run the focused test suite
uv run ty check    # type checking (ty)
```
