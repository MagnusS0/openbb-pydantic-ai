"""Helpers for constructing deferred MCP toolsets from QueryRequest payloads."""

from __future__ import annotations

from collections.abc import Sequence

from openbb_ai.models import AgentTool
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.toolsets import ExternalToolset

from openbb_pydantic_ai._dependencies import OpenBBDeps


class MCPToolset(ExternalToolset[OpenBBDeps]):
    """External toolset exposing workspace-selected MCP tools."""

    def __init__(self, tools: Sequence[AgentTool]):
        self.tools_by_name: dict[str, AgentTool] = {}

        tool_defs: list[ToolDefinition] = []
        for tool in tools:
            tool_def = ToolDefinition(
                name=tool.name,
                parameters_json_schema=tool.input_schema
                or {"type": "object", "properties": {}, "additionalProperties": True},
                description=tool.description or "",
            )
            tool_defs.append(tool_def)
            self.tools_by_name[tool_def.name] = tool

        super().__init__(tool_defs)


def build_mcp_toolsets(
    tools: Sequence[AgentTool] | None,
) -> tuple[MCPToolset, ...]:
    """Create MCP toolsets mirroring only the tools selected in the UI."""

    if not tools:
        return ()

    return (MCPToolset(tools),)
