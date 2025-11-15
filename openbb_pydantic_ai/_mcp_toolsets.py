"""Helpers for constructing deferred MCP toolsets from QueryRequest payloads."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from openbb_ai.models import AgentTool
from pydantic_ai import CallDeferred, Tool
from pydantic_ai.tools import RunContext
from pydantic_ai.toolsets import FunctionToolset

from ._dependencies import OpenBBDeps

MCP_WORKSPACE_OPTION = "mcp-tools"


def _default_schema() -> dict[str, Any]:
    return {"type": "object", "properties": {}, "additionalProperties": True}


def _tool_schema(tool: AgentTool) -> dict[str, Any]:
    schema = tool.input_schema
    if isinstance(schema, Mapping):
        return dict(schema)
    return _default_schema()


def _tool_description(tool: AgentTool) -> str:
    if tool.description:
        return tool.description
    if tool.endpoint:
        return f"Invoke '{tool.endpoint}' on the MCP server."
    return "Workspace-provided MCP tool"


def _function_name(tool: AgentTool) -> str:
    base = tool.name or tool.endpoint or tool.server_id or tool.url or "mcp_tool"
    slug = "".join(ch if ch.isalnum() else "_" for ch in base).strip("_")
    return f"call_{slug or 'mcp_tool'}"


def _resolved_tool_name(tool: AgentTool) -> str:
    if tool.name:
        return tool.name
    if tool.endpoint:
        return tool.endpoint
    return tool.server_id or tool.url or "mcp_tool"


def _build_deferred_tool(tool: AgentTool) -> Tool:
    schema = _tool_schema(tool)
    description = _tool_description(tool)
    tool_name = _resolved_tool_name(tool)

    async def _call(ctx: RunContext[OpenBBDeps], **input_arguments: Any) -> None:
        if ctx.tool_call_id is None:
            raise RuntimeError("Deferred MCP tools require a tool call id.")
        raise CallDeferred

    _call.__name__ = _function_name(tool)

    return Tool.from_schema(
        function=_call,
        name=tool_name,
        description=description,
        json_schema=schema,
        takes_ctx=True,
    )


class MCPToolset(FunctionToolset[OpenBBDeps]):
    """Toolset exposing workspace-selected MCP tools via deferred calls."""

    def __init__(self, tools: Sequence[AgentTool]):
        super().__init__()
        self._tools_by_name: dict[str, AgentTool] = {}
        self._registered_tools: dict[str, Tool] = {}

        for tool in tools:
            built = _build_deferred_tool(tool)
            self.add_tool(built)
            self._tools_by_name[built.name] = tool
            self._registered_tools[built.name] = built

    @property
    def tools_by_name(self) -> Mapping[str, AgentTool]:
        return self._tools_by_name

    @property
    def registered_tools(self) -> Mapping[str, Tool]:
        return self._registered_tools


def build_mcp_toolsets(
    tools: Sequence[AgentTool] | None,
    workspace_options: Sequence[str] | None,
) -> tuple[MCPToolset, ...]:
    """Create MCP toolsets mirroring only the tools selected in the UI."""

    if not tools:
        return ()

    if workspace_options is not None and MCP_WORKSPACE_OPTION not in workspace_options:
        return ()

    return (MCPToolset(tools),)
