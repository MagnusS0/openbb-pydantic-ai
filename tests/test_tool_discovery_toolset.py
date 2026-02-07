from __future__ import annotations

from typing import Any, cast
from unittest.mock import MagicMock

import pytest
from pydantic_ai import DeferredToolRequests
from pydantic_ai.messages import ToolCallPart
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.toolsets import AbstractToolset

from openbb_pydantic_ai.tool_discovery import ToolDiscoveryToolset

pytestmark = pytest.mark.anyio


class _FakeToolHandle:
    def __init__(self, definition: ToolDefinition):
        self.definition = definition


class _FakeToolset:
    def __init__(
        self,
        *,
        tool_name: str,
        description: str,
        schema: dict[str, Any],
        result: Any,
    ) -> None:
        self._definition = ToolDefinition(
            name=tool_name,
            description=description,
            parameters_json_schema=schema,
        )
        self._result = result
        self.calls: list[tuple[str, dict[str, Any], _FakeToolHandle]] = []

    async def get_tools(self, ctx: Any) -> dict[str, _FakeToolHandle]:
        _ = ctx
        return {self._definition.name: _FakeToolHandle(self._definition)}

    async def call_tool(
        self,
        name: str,
        tool_args: dict[str, Any],
        ctx: Any,
        tool: _FakeToolHandle,
    ) -> Any:
        _ = ctx
        self.calls.append((name, tool_args, tool))
        if callable(self._result):
            return self._result(tool_args)
        return self._result


async def test_tool_discovery_lists_and_calls_wrapped_function_tool() -> None:
    wrapped = _FakeToolset(
        tool_name="add",
        description="Add two numbers",
        schema={
            "type": "object",
            "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
            "required": ["a", "b"],
        },
        result=lambda args: args["a"] + args["b"],
    )
    discovery = ToolDiscoveryToolset(
        toolsets=[("math", cast(AbstractToolset[Any], wrapped))]
    )
    ctx = MagicMock()

    listed = await discovery._list_tools_impl(ctx)
    assert listed == {"add": "Add two numbers"}

    schema_json = await discovery._get_tool_schema_impl(ctx, "add")
    assert '"name": "add"' in schema_json
    assert '"group": "math"' in schema_json

    result = await discovery._call_tool_impl(ctx, "add", {"a": 2, "b": 3})
    assert result == 5
    assert wrapped.calls and wrapped.calls[0][0] == "add"


async def test_tool_discovery_delegates_external_tool_calls() -> None:
    deferred = DeferredToolRequests()
    deferred.calls.append(
        ToolCallPart(
            tool_name="remote_lookup",
            tool_call_id="call-1",
            args={"query": "AAPL"},
        )
    )
    external = _FakeToolset(
        tool_name="remote_lookup",
        description="Remote lookup",
        schema={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
        result=deferred,
    )
    discovery = ToolDiscoveryToolset(
        toolsets=[("external", cast(AbstractToolset[Any], external))]
    )
    ctx = MagicMock()

    result = await discovery._call_tool_impl(ctx, "remote_lookup", {"query": "AAPL"})
    assert isinstance(result, DeferredToolRequests)
    assert result.calls and result.calls[0].tool_name == "remote_lookup"
    assert external.calls and external.calls[0][0] == "remote_lookup"


async def test_tool_discovery_rejects_duplicate_tool_names() -> None:
    first = _FakeToolset(
        tool_name="duplicate",
        description="First tool",
        schema={"type": "object", "properties": {}},
        result="ok",
    )
    second = _FakeToolset(
        tool_name="duplicate",
        description="Second tool",
        schema={"type": "object", "properties": {}},
        result="ok",
    )

    discovery = ToolDiscoveryToolset(
        toolsets=[
            ("first_group", cast(AbstractToolset[Any], first)),
            ("second_group", cast(AbstractToolset[Any], second)),
        ]
    )
    with pytest.raises(ValueError, match="Duplicate tool name 'duplicate'"):
        await discovery._resolve_pending(MagicMock())
