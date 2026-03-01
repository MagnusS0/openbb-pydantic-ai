# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`openbb-pydantic-ai` is an adapter that bridges Pydantic AI agents with OpenBB Workspace. It translates `QueryRequest` payloads into Pydantic AI runs, exposes Workspace widgets as deferred tools, and streams native OpenBB SSE events back to the UI.

## Development Commands

```bash
# Install dependencies
uv sync --dev

# Run tests
uv run pytest

# Run single test with output
uv run pytest -s tests/test_adapter.py

# Run linting and type checking (auto-formats imports)
uv run prek run --all-files

# Type checking only
uv run ty check
```

Always run both `uv run prek run --all-files` and `uv run pytest` before considering any task complete. Use `uv` for all commands, never direct `pip` or `python`.

## Architecture

### Core Flow

1. **Request Entry**: `OpenBBAIAdapter.dispatch_request()` receives FastAPI/Starlette requests, parses the body into `QueryRequest`
2. **Message Transformation**: `MessageTransformer` converts OpenBB `LlmMessage` history to Pydantic AI `ModelMessage` format
3. **Widget Toolsets**: Widgets from the request become deferred Pydantic AI tools via `build_widget_toolsets()`
4. **Event Streaming**: `OpenBBAIEventStream` transforms Pydantic AI events into OpenBB SSE format

### Key Components

- **`OpenBBAIAdapter`** (`_adapter.py`): Main entry point, implements `UIAdapter`. Handles request parsing, builds message history, creates toolsets, and manages deferred tool results
- **`OpenBBAIEventStream`** (`_event_stream.py`): Transforms Pydantic AI events (text deltas, tool calls, thinking) into OpenBB SSE events (`copilotMessageChunk`, `copilotFunctionCall`, reasoning steps)
- **`WidgetRegistry`** (`_widget_registry.py`): Lookup service for widgets by UUID, origin/id, or tool name
- **`MessageTransformer`** (`_message_transformer.py`): Bidirectional conversion between OpenBB and Pydantic AI message formats
- **`StreamParser`** (`_stream_parser.py`): Parses streamed text for chart placeholders (`{{place_chart_here}}`) and injects artifacts
- **`_viz_toolsets.py`**: Built-in tools (`openbb_create_chart`, `openbb_create_table`, `openbb_create_html`) with Pydantic validation for chart/table params
- **`_config.py`**: Centralized constants — tool names, event types, placeholder tokens
- **`_mcp_toolsets.py`**: Wraps external MCP tools from `QueryRequest.tools` into Pydantic AI toolsets
- **`pdf/`**: Optional subpackage (requires `docling`) for PDF text extraction with OCR and citation bounding boxes

### Dependencies

- **`OpenBBDeps`** (`_dependencies.py`): Runtime context injected into agents containing workspace state, widgets, context, URLs, and timezone
- **`openbb-ai`**: Provides SDK types (`QueryRequest`, `Widget`, `SSE`, etc.) and helpers (`cite`, `get_widget_data`, `reasoning_step`)
- **`pydantic-ai-slim`**: Core agent framework with `UIAdapter`, `UIEventStream`, and toolset abstractions

### Tool Handling

Widget tools follow the naming convention `openbb_widget_<identifier>`. The adapter supports:
- **Widget toolsets**: Generated from `QueryRequest.widgets`, emit `copilotFunctionCall` events
- **Viz toolsets**: `openbb_create_chart`, `openbb_create_table`, `openbb_create_html` for structured output
- **MCP toolsets**: External tools from `QueryRequest.tools`, routed via `execute_mcp_tool`
- **Deferred results**: Pending tool results from previous turns are replayed before the run starts

### SSE Event Types

The adapter produces these OpenBB SSE events:
- `copilotMessageChunk`: Streamed text content
- `copilotFunctionCall`: Widget data requests (via `get_widget_data`)
- `copilotMessageArtifact`: Tables and charts
- `copilotCitationCollection`: Citations for widget data used
- Reasoning steps: "Thinking" traces, tool calls, warnings, errors

## Code Style

See `.github/instructions/python.instructions.md` for detailed Python coding conventions used in this repo (imports, types, testing, docstrings, dataclasses vs BaseModel, etc.). Read it before writing code.
