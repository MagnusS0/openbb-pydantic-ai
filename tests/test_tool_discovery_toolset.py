from __future__ import annotations

import json
from typing import Any, cast
from unittest.mock import MagicMock

import pytest
from pydantic_ai import DeferredToolRequests
from pydantic_ai.messages import ToolCallPart
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import RunContext, ToolDefinition
from pydantic_ai.toolsets import AbstractToolset, ExternalToolset, FunctionToolset
from pydantic_ai.usage import RunUsage

from openbb_pydantic_ai._viz_toolsets import build_viz_toolsets
from openbb_pydantic_ai.tool_discovery import ToolDiscoveryToolset

pytestmark = pytest.mark.anyio


class _FakeToolHandle:
    def __init__(self, tool_def: ToolDefinition):
        self.tool_def = tool_def


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


def _build_run_context() -> RunContext[Any]:
    return RunContext(
        deps=MagicMock(),
        model=TestModel(call_tools=[]),
        usage=RunUsage(),
    )


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

    schema_json = await discovery._get_tool_schema_impl(ctx, ["add"])
    parsed = json.loads(schema_json)
    assert parsed["count"] == 1
    assert parsed["tools"][0]["name"] == "add"
    assert parsed["tools"][0]["group"] == "math"

    result = await discovery._call_tool_impl(ctx, "add", {"a": 2, "b": 3})
    assert result == 5
    assert wrapped.calls and wrapped.calls[0][0] == "add"


async def test_tool_discovery_get_tool_schema_supports_batch_lookup() -> None:
    first = _FakeToolset(
        tool_name="add",
        description="Add two numbers",
        schema={
            "type": "object",
            "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
            "required": ["a", "b"],
        },
        result="ok",
    )
    second = _FakeToolset(
        tool_name="subtract",
        description="Subtract two numbers",
        schema={
            "type": "object",
            "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
            "required": ["a", "b"],
        },
        result="ok",
    )
    discovery = ToolDiscoveryToolset(
        toolsets=[
            ("math_a", cast(AbstractToolset[Any], first)),
            ("math_b", cast(AbstractToolset[Any], second)),
        ]
    )
    ctx = MagicMock()

    payload = await discovery._get_tool_schema_impl(ctx, ["add", "subtract", "add"])
    parsed = json.loads(payload)
    assert parsed["count"] == 2
    assert [tool["name"] for tool in parsed["tools"]] == ["add", "subtract"]


async def test_tool_discovery_get_tool_schema_requires_name_input() -> None:
    discovery = ToolDiscoveryToolset()
    with pytest.raises(
        ValueError, match="`tool_names` must contain at least one tool name"
    ):
        await discovery._get_tool_schema_impl(MagicMock(), [])


async def test_tool_discovery_search_tools_matches_multi_token_query() -> None:
    wrapped = _FakeToolset(
        tool_name="openbb_widget_price_performance",
        description="Interactive chart for asset historical price data.",
        schema={"type": "object", "properties": {}},
        result="ok",
    )
    discovery = ToolDiscoveryToolset(
        toolsets=[("widgets", cast(AbstractToolset[Any], wrapped))]
    )

    result = await discovery._search_tools_impl(MagicMock(), "price performance chart")
    assert "openbb_widget_price_performance" in result


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


async def test_tool_discovery_resolves_real_toolsettool_shape() -> None:
    toolset = FunctionToolset[Any]()

    def echo(ctx: RunContext[Any], value: str) -> str:
        _ = ctx
        return value

    toolset.add_function(echo, name="echo")

    discovery = ToolDiscoveryToolset(
        toolsets=[("real_function_toolset", cast(AbstractToolset[Any], toolset))]
    )
    ctx = _build_run_context()

    listed = await discovery._list_tools_impl(ctx)
    assert "echo" in listed

    result = await discovery._call_tool_impl(ctx, "echo", {"value": "ok"})
    assert result == "ok"


async def test_tool_discovery_call_tool_impl_rejects_non_object_arguments() -> None:
    wrapped = _FakeToolset(
        tool_name="echo_symbol",
        description="Echo symbol",
        schema={"type": "object", "properties": {"symbol": {"type": "string"}}},
        result=lambda args: args["symbol"],
    )
    discovery = ToolDiscoveryToolset(
        toolsets=[("symbols", cast(AbstractToolset[Any], wrapped))]
    )

    with pytest.raises(ValueError, match="must be an object/dictionary"):
        await discovery._call_tool_impl(MagicMock(), "echo_symbol", cast(Any, ["SILJ"]))


async def test_tool_discovery_external_toolset_falls_back_to_deferred_requests() -> (
    None
):
    external = ExternalToolset[Any](
        [
            ToolDefinition(
                name="openbb_widget_price_performance",
                description="Widget tool",
                parameters_json_schema={
                    "type": "object",
                    "properties": {"symbol": {"type": "string"}},
                    "required": ["symbol"],
                },
            )
        ]
    )
    discovery = ToolDiscoveryToolset(
        toolsets=[("widgets", cast(AbstractToolset[Any], external))]
    )

    ctx = _build_run_context()
    result = await discovery._call_tool_impl(
        ctx,
        "openbb_widget_price_performance",
        {"symbol": "SILJ"},
    )

    assert isinstance(result, DeferredToolRequests)
    assert len(result.calls) == 1
    assert result.calls[0].tool_name == "openbb_widget_price_performance"
    assert result.calls[0].args == {"symbol": "SILJ"}


async def test_call_tools_batch_executes_all() -> None:
    adder = _FakeToolset(
        tool_name="add",
        description="Add",
        schema={
            "type": "object",
            "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
        },
        result=lambda args: args["a"] + args["b"],
    )
    multiplier = _FakeToolset(
        tool_name="mul",
        description="Multiply",
        schema={
            "type": "object",
            "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
        },
        result=lambda args: args["a"] * args["b"],
    )
    discovery = ToolDiscoveryToolset(
        toolsets=[
            ("math", cast(AbstractToolset[Any], adder)),
            ("math", cast(AbstractToolset[Any], multiplier)),
        ]
    )

    ctx = MagicMock()
    result = await discovery._call_tools_impl(
        ctx,
        [
            {"tool_name": "add", "arguments": {"a": 1, "b": 2}},
            {"tool_name": "mul", "arguments": {"a": 3, "b": 4}},
        ],
    )
    assert result == [
        {"tool_name": "add", "result": 3},
        {"tool_name": "mul", "result": 12},
    ]
    assert len(adder.calls) == 1
    assert len(multiplier.calls) == 1


async def test_call_tools_single_immediate_returns_raw_result() -> None:
    adder = _FakeToolset(
        tool_name="add",
        description="Add",
        schema={
            "type": "object",
            "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
        },
        result=lambda args: args["a"] + args["b"],
    )
    discovery = ToolDiscoveryToolset(
        toolsets=[("math", cast(AbstractToolset[Any], adder))]
    )

    result = await discovery._call_tools_impl(
        MagicMock(),
        [{"tool_name": "add", "arguments": {"a": 2, "b": 3}}],
    )
    assert result == 5


async def test_call_tools_batch_merges_deferred_requests() -> None:
    ext_a = ExternalToolset[Any](
        [
            ToolDefinition(
                name="widget_a",
                description="Widget A",
                parameters_json_schema={
                    "type": "object",
                    "properties": {"symbol": {"type": "string"}},
                },
            )
        ]
    )
    ext_b = ExternalToolset[Any](
        [
            ToolDefinition(
                name="widget_b",
                description="Widget B",
                parameters_json_schema={
                    "type": "object",
                    "properties": {"symbol": {"type": "string"}},
                },
            )
        ]
    )
    discovery = ToolDiscoveryToolset(
        toolsets=[
            ("widgets_a", cast(AbstractToolset[Any], ext_a)),
            ("widgets_b", cast(AbstractToolset[Any], ext_b)),
        ]
    )

    ctx = _build_run_context()
    result = await discovery._call_tools_impl(
        ctx,
        [
            {"tool_name": "widget_a", "arguments": {"symbol": "AAPL"}},
            {"tool_name": "widget_b", "arguments": {"symbol": "MSFT"}},
        ],
    )
    assert isinstance(result, DeferredToolRequests)
    assert len(result.calls) == 2
    names = {c.tool_name for c in result.calls}
    assert names == {"widget_a", "widget_b"}
    call_ids = {c.tool_call_id for c in result.calls}
    assert len(call_ids) == 2


async def test_call_tools_rejects_mixed_deferred_and_immediate_results() -> None:
    adder = _FakeToolset(
        tool_name="add",
        description="Add",
        schema={
            "type": "object",
            "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
        },
        result=lambda args: args["a"] + args["b"],
    )
    deferred_toolset = ExternalToolset[Any](
        [
            ToolDefinition(
                name="widget_a",
                description="Widget A",
                parameters_json_schema={
                    "type": "object",
                    "properties": {"symbol": {"type": "string"}},
                },
            )
        ]
    )
    discovery = ToolDiscoveryToolset(
        toolsets=[
            ("math", cast(AbstractToolset[Any], adder)),
            ("widgets", cast(AbstractToolset[Any], deferred_toolset)),
        ]
    )

    with pytest.raises(ValueError, match="mixed deferred tools and immediate tools"):
        await discovery._call_tools_impl(
            _build_run_context(),
            [
                {"tool_name": "add", "arguments": {"a": 1, "b": 2}},
                {"tool_name": "widget_a", "arguments": {"symbol": "AAPL"}},
            ],
        )


async def test_call_tools_rejects_empty_calls() -> None:
    discovery = ToolDiscoveryToolset()
    with pytest.raises(ValueError, match="`calls` must be a non-empty list"):
        await discovery._call_tools_impl(MagicMock(), [])


async def test_call_tools_rejects_missing_tool_name() -> None:
    discovery = ToolDiscoveryToolset()
    with pytest.raises(ValueError, match="must be an object with a `tool_name` key"):
        await discovery._call_tools_impl(MagicMock(), [{"arguments": {"a": 1}}])


async def test_tool_discovery_executes_html_tool_without_attribute_error() -> None:
    viz_toolset = build_viz_toolsets()
    discovery = ToolDiscoveryToolset(
        toolsets=[("viz", cast(AbstractToolset[Any], viz_toolset))]
    )
    result = await discovery._call_tool_impl(
        _build_run_context(),
        "openbb_create_html",
        {"content": "<div>Hello</div>", "name": "Greeting"},
    )

    assert (
        getattr(result, "return_value", None) == "HTML artifact created successfully."
    )
    metadata = getattr(result, "metadata", None)
    assert isinstance(metadata, dict)
    assert "html" in metadata


async def test_tool_discovery_executes_chart_tool_without_attribute_error() -> None:
    viz_toolset = build_viz_toolsets()
    discovery = ToolDiscoveryToolset(
        toolsets=[("viz", cast(AbstractToolset[Any], viz_toolset))]
    )
    result = await discovery._call_tool_impl(
        _build_run_context(),
        "openbb_create_chart",
        {
            "type": "line",
            "data": [{"period": "Q1", "value": 1}],
            "x_key": "period",
            "y_keys": ["value"],
            "name": "Sample",
        },
    )

    assert getattr(result, "return_value", None) == "Chart created successfully."
    metadata = getattr(result, "metadata", None)
    assert isinstance(metadata, dict)
    assert "chart" in metadata
