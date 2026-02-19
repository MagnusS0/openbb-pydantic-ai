[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![ty](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ty/main/assets/badge/v0.json)](https://github.com/astral-sh/ty)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/MagnusS0/openbb-pydantic-ai)


# OpenBB Pydantic AI Adapter

`openbb-pydantic-ai` lets any [Pydantic AI](https://ai.pydantic.dev/) agent
run behind OpenBB Workspace by translating `QueryRequest` payloads into a Pydantic
AI run, exposing Workspace widgets as deferred tools, and streaming native
OpenBB SSE events back to the UI.

- **Stateless by design**: each `QueryRequest` carries the full conversation history, widgets, context, and URLs so requests are processed independently.
- **First-class widget tools**: every widget becomes a deferred Pydantic AI tool; when the model calls one, the adapter emits `copilotFunctionCall` events and waits for the Workspace to return data before resuming.
- **Rich event stream**: reasoning steps, thinking traces, tables, charts, HTML artifacts, and citations are streamed as native OpenBB SSE payloads.
- **PDF context**: install the `[pdf]` extra and the agent can extract and query PDF widgets on the active dashboard, with citations linked to specific pages.
- **Output helpers included**: structured outputs (dicts/lists) are auto-detected and converted to tables or charts; chart parameters are normalized for consistent rendering.

See the [OpenBB Custom Agent SDK](https://github.com/OpenBB-finance/openbb-ai) and
[Pydantic AI UI adapter docs](https://ai.pydantic.dev/ui/overview/) for the underlying types.

## Installation

```bash
pip install openbb-pydantic-ai
# or with uv
uv add openbb-pydantic-ai
```

For PDF context support (requires [docling](https://github.com/docling-project/docling)):

```bash
uv add "openbb-pydantic-ai[pdf]"
# GPU variant (CUDA 12.8)
uv add "openbb-pydantic-ai[pdf-cu128]"
```

## Quick Start (FastAPI)

```python
from anyio import BrokenResourceError
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic_ai import Agent

from openbb_pydantic_ai import OpenBBAIAdapter, OpenBBDeps

agent = Agent(
    "openrouter:minimax/minimax-m2.5",
    instructions="Be concise and helpful. Only use widget tools for data lookups.",
    deps_type=OpenBBDeps,
)

app = FastAPI()
AGENT_BASE_URL = "http://localhost:8003"


@app.get("/agents.json")
async def agents_json():
    return JSONResponse(
        content={
            "<agent-id>": {
                "name": "My Custom Agent",
                "description": "This is my custom agent",
                "image": f"{AGENT_BASE_URL}/my-custom-agent/logo.png",
                "endpoints": {"query": f"{AGENT_BASE_URL}/query"},
                "features": {
                    "streaming": True,
                    "widget-dashboard-select": True,  # primary & secondary widgets
                    "widget-dashboard-search": True,  # extra widgets
                    "mcp-tools": True,
                },
            }
        }
    )


@app.post("/query")
async def query(request: Request):
    try:
        return await OpenBBAIAdapter.dispatch_request(request, agent=agent)
    except BrokenResourceError:
        pass  # client disconnected


app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://pro.openbb.co"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### How It Works

#### 1. Request Handling

- OpenBB Workspace POST's a `QueryRequest` to `/query`
- `OpenBBAIAdapter` validates it, builds the Pydantic AI message stack, and injects workspace context and URLs as system prompts

#### 2. Widget Tool Conversion

- Widgets in the request become deferred Pydantic AI tools
- Each call emits a `copilotFunctionCall` event (via `get_widget_data`)
- The adapter pauses until Workspace responds with data, then resumes the run

#### 3. Event Streaming

| Pydantic AI event | OpenBB SSE event |
|---|---|
| Text chunk | `copilotMessageChunk` |
| Reasoning / thinking block | Collapsed under "Step-by-step reasoning" dropdown |
| Table / chart / HTML artifact | `copilotMessageArtifact` |
| Widget citations | `copilotCitationCollection` (batched at end of run) |

## Features

### Widget Toolsets

Widgets are grouped by priority (`primary`, `secondary`, `extra`) and exposed through dedicated toolsets. Tool names follow the `openbb_widget_<identifier>` convention with any redundant `openbb_` prefix trimmed (e.g. `openbb_widget_financial_statements`).

Control access via the `agents.json` feature flags:

```json
"features": {
    "widget-dashboard-select": true,
    "widget-dashboard-search": true
}
```

### Visualization: Charts, Tables & HTML

Three built-in tools handle structured output. The model can call any of them directly; the adapter handles serialization and streaming.

#### `openbb_create_chart`

Creates chart artifacts inline in the response. Supported types: `line`, `bar`, `scatter`, `pie`, `donut`.

Insert `{{place_chart_here}}` in the model's text where the chart should appear — the adapter swaps the placeholder with the rendered artifact while streaming:

```
Here is the revenue breakdown: {{place_chart_here}}
```

Required axes:
- `line` / `bar` / `scatter`: `x_key` + `y_keys`
- `pie` / `donut`: `angle_key` + `callout_label_key`

Different field spellings (`y_keys`, `yKeys`, etc.) are accepted and normalized before emitting.

#### `openbb_create_table`

Creates a table artifact from structured data with explicit column ordering and metadata. Use this when you want predictable output over auto-detection.

#### `openbb_create_html`

Renders a self-contained HTML artifact, useful for custom layouts, formatted reports, or SVG-based plots when Markdown isn't enough.

> **Constraint**: limited to HTML + CSS + inline SVG. No JavaScript. This is an OpenBB Workspace restriction on non-Enterprise plans.

**Auto-detection**: dict/list outputs shaped like `{"type": "table", "data": [...]}` or a plain list of dicts are automatically converted to table artifacts without calling any tool explicitly.

**Markdown tables** are also supported: stream tabular data as Markdown and Workspace renders it as an interactive table users can promote to a widget.

### MCP Tools

Tools listed in `QueryRequest.tools` are exposed as an external MCP toolset. The model sees the same tool names the Workspace UI presents. Deferred `execute_agent_tool` results replay on the next request just like widget results.

Enable in `agents.json`:

```json
"features": { "mcp-tools": true }
```

### PDF Context

Install the `[pdf]` extra to enable PDF support. When a PDF widget is on the active dashboard, the agent can extract and query it with citations linked to specific pages.

```bash
uv add "openbb-pydantic-ai[pdf]"
```

> **Performance**: GPU extraction is significantly faster. CPU works, but expect slowdowns on documents over ~50 pages.

### Deferred Results & Citations

- Pending widget responses in the request are replayed before the run starts, keeping multi-turn workflows seamless.
- Every widget call records a citation via `openbb_ai.helpers.cite`, emitted as a `copilotCitationCollection` at the end of the run.

## Progressive Tool Discovery (Default)

Instead of dumping every tool schema into the context upfront, the adapter wraps toolsets with four meta-tools:

| Meta-tool | Purpose |
|---|---|
| `list_tools` | List available tools by group |
| `search_tools` | Keyword search across tool descriptions |
| `get_tool_schema` | Fetch the full schema for a specific tool |
| `call_tools` | Invoke a tool by name |

The model fetches schemas only when it needs them, keeping the initial context window small. Deferred flows (widget data, MCP) continue to emit `get_widget_data` and `execute_agent_tool` events as before.

To disable and expose all schemas upfront:

```python
adapter = OpenBBAIAdapter(
    agent=agent,
    run_input=run_input,
    enable_progressive_tool_discovery=False,
)
```

## Adding Custom Toolsets

Pass custom or third-party toolsets to the adapter at request time rather than mounting them on `Agent`. They are merged into the progressive discovery wrapper automatically.

> **Important**: do **not** also pass these toolsets to `Agent(toolsets=[...])` when using the OpenBB adapter — they would appear as both direct and progressive tools.

Tag a toolset with `add_to_progressive(...)`:

```python
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai.tools import RunContext

from openbb_pydantic_ai import OpenBBDeps
from openbb_pydantic_ai.tool_discovery import add_to_progressive

custom_tools = FunctionToolset[OpenBBDeps](id="custom_agent_tools")


@custom_tools.tool
def earnings_note(ctx: RunContext[OpenBBDeps], symbol: str) -> str:
    _ = ctx
    return f"Custom note for {symbol}"


add_to_progressive(
    custom_tools,
    group="custom_agent_tools",
    description="Custom user tools",
)

# Pass at request time
return await OpenBBAIAdapter.dispatch_request(request, agent=agent, toolsets=[custom_tools])
```

Or use the `@progressive(...)` decorator directly on the tool function:

```python
from openbb_pydantic_ai.tool_discovery import progressive


@progressive(toolset=custom_tools, group="custom_agent_tools", description="Custom user tools")
@custom_tools.tool
def earnings_note(ctx: RunContext[OpenBBDeps], symbol: str) -> str:
    _ = ctx
    return f"Custom note for {symbol}"
```

Untagged toolsets passed at request time are forwarded as standalone toolsets without being merged into the progressive wrapper.

## Advanced Usage

Instantiate the adapter manually for full control:

```python
from openbb_pydantic_ai import OpenBBAIAdapter

run_input = OpenBBAIAdapter.build_run_input(body_bytes)
adapter = OpenBBAIAdapter(agent=agent, run_input=run_input)

async for event in adapter.run_stream():
    yield event  # already encoded as OpenBB SSE payloads
```

`message_history`, `deferred_tool_results`, and `on_complete` callbacks are forwarded directly to `Agent.run_stream_events()`.

**Runtime deps & prompts**: `OpenBBDeps` bundles widgets (by priority group), context rows, relevant URLs, workspace state, timezone, and a `state` dict you can pass to toolsets or output validators. The adapter merges dashboard context and current widget parameter values into the runtime instructions automatically — append your own instructions without re-supplying that context.

## Local Development

```bash
uv sync --dev
uv run pytest
uv run pre-commit run --all-files  # lint + format
```
