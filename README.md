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
from openbb_pydantic_ai import OpenBBAIAdapter

agent = Agent(
    "openai:gpt-4o",
    instructions="Be concise and helpful. Only use widget tools for data lookups.",
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

## Features

### Widget Toolsets

- Widgets are grouped by priority (`primary`, `secondary`, `extra`) and exposed
  through dedicated toolsets so you can gate access if needed.
- Tool names follow `openbb_widget_{origin}_{widget_id}`; the helper
  `build_widget_tool_name` reproduces the exact string for routing.

### Deferred Results & Citations

- Pending widget responses provided in the request are replayed before the run
  starts, making multi-turn workflows seamless.
- Every widget call records a citation via `openbb_ai.helpers.cite`, emitted as a
  `copilotCitationCollection` at the end of the run.

### Structured Output Detection

The adapter provides built-in tools and automatic detection for tables and charts:

- **`openbb_create_table`** - Explicitly create table artifacts from structured data
- **`openbb_create_chart`** - Create chart artifacts (line, bar, scatter, pie, donut) with validation
- **Auto-detection** - Dict/list outputs shaped like `{"type": "table", "data": [...]}` (or just a list of dicts) automatically become tables
- **Flexible chart parameters** - Chart outputs tolerate different field spellings (`y_keys`, `yKeys`, etc.) and validate required axes before emitting

These tools are always available through the `VisualizationToolset`, allowing agents to explicitly create well-formatted visualizations.

## Local Development

This repo ships a UV-based workflow:

```bash
uv sync --dev         # install dependencies
uv run pytest      # run the focused test suite
uv run ty check    # static type checking (ty)
```
