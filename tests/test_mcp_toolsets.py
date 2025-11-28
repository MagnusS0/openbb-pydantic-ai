from __future__ import annotations

import pytest
from openbb_ai.models import AgentTool
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel
from pydantic_ai.toolsets import ExternalToolset

from openbb_pydantic_ai._dependencies import OpenBBDeps
from openbb_pydantic_ai._mcp_toolsets import build_mcp_toolsets

pytestmark = pytest.mark.anyio


def _tool(**overrides) -> AgentTool:
    defaults = {
        "server_id": "1761162970409",
        "name": "database-connector_get_databases",
        "url": "http://localhost:8001/mcp/",
        "description": None,
        "endpoint": None,
        "input_schema": None,
        "auth_token": None,
    }
    defaults.update(overrides)
    return AgentTool(**defaults)


def test_build_mcp_toolsets_allows_empty_options_with_tools() -> None:
    """Workspace options should not gate MCP toolsets when tools are present."""

    tools = [_tool(description="List databases")]

    first = build_mcp_toolsets(tools)
    second = build_mcp_toolsets(tools)
    assert len(first) == len(second) == 1


def test_build_mcp_toolsets_creates_deferred_tools_for_each_agent_tool() -> None:
    schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
        },
        "required": ["query"],
    }

    tools = [
        _tool(description="List databases", input_schema=schema),
        _tool(
            name="database-connector_show_tables",
            description="List tables",
            input_schema=None,
        ),
    ]

    toolsets = build_mcp_toolsets(tools)
    assert len(toolsets) == 1
    toolset = toolsets[0]

    assert isinstance(toolset, ExternalToolset)

    names = {tool_def.name for tool_def in toolset.tool_defs}
    assert names == {
        "database-connector_get_databases",
        "database-connector_show_tables",
    }

    db_def = next(
        td for td in toolset.tool_defs if td.name == "database-connector_get_databases"
    )
    assert db_def.description == "List databases"
    assert db_def.parameters_json_schema == schema

    show_def = next(
        td for td in toolset.tool_defs if td.name == "database-connector_show_tables"
    )
    assert show_def.parameters_json_schema == {
        "type": "object",
        "properties": {},
        "additionalProperties": True,
    }


async def test_mcp_toolset_registers_tools_with_agent() -> None:
    tools = [
        _tool(description="List databases"),
        _tool(
            name="database-connector_show_tables",
            description="List tables",
        ),
    ]
    toolsets = build_mcp_toolsets(tools)

    model = TestModel(call_tools=[], custom_output_text="done")
    agent = Agent(
        model,
        deps_type=OpenBBDeps,
        toolsets=toolsets,
    )

    await agent.run("hello", deps=OpenBBDeps())

    params = model.last_model_request_parameters
    assert params is not None
    names = {tool.name for tool in params.function_tools}
    assert names == {
        "database-connector_get_databases",
        "database-connector_show_tables",
    }
