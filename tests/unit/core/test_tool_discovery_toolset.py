from __future__ import annotations

from typing import Any, cast
from unittest.mock import MagicMock

import pytest
from pydantic_ai.exceptions import CallDeferred
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import RunContext, ToolDefinition
from pydantic_ai.toolsets import AbstractToolset, ExternalToolset
from pydantic_ai.usage import RunUsage

from openbb_pydantic_ai._viz_toolsets import build_viz_toolsets
from openbb_pydantic_ai.tool_discovery import ToolDiscoveryToolset


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


def _toolset(
    *,
    name: str,
    description: str,
    result: Any,
    schema: dict[str, Any] | None = None,
) -> _FakeToolset:
    return _FakeToolset(
        tool_name=name,
        description=description,
        schema=schema
        or {
            "type": "object",
            "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
        },
        result=result,
    )


async def test_list_tools_returns_grouped_entries() -> None:
    first = _toolset(name="add", description="Add numbers", result="ok")
    second = _toolset(
        name="price_chart", description="Price performance chart", result="ok"
    )
    discovery = ToolDiscoveryToolset(
        toolsets=[
            ("math", cast(AbstractToolset[Any], first)),
            ("widgets", cast(AbstractToolset[Any], second)),
        ]
    )

    listed = await discovery._list_tools_impl(MagicMock())
    assert "# math" in listed
    assert "# widgets" in listed
    assert "- add: Add numbers" in listed


async def test_search_tools_matches_tokens() -> None:
    first = _toolset(name="add", description="Add numbers", result="ok")
    second = _toolset(
        name="price_chart", description="Price performance chart", result="ok"
    )
    discovery = ToolDiscoveryToolset(
        toolsets=[
            ("math", cast(AbstractToolset[Any], first)),
            ("widgets", cast(AbstractToolset[Any], second)),
        ]
    )

    searched = await discovery._search_tools_impl(MagicMock(), "performance chart")
    assert "price_chart" in searched
    assert "add" not in searched


async def test_get_tool_schema_deduplicates_requested_names() -> None:
    add = _toolset(name="add", description="Add", result="ok")
    sub = _toolset(name="sub", description="Subtract", result="ok")
    discovery = ToolDiscoveryToolset(
        toolsets=[
            ("math_a", cast(AbstractToolset[Any], add)),
            ("math_b", cast(AbstractToolset[Any], sub)),
        ]
    )

    schema_text = await discovery._get_tool_schema_impl(
        MagicMock(), ["add", "sub", "add"]
    )
    assert schema_text.count("<add>") == 1
    assert schema_text.count("<sub>") == 1


async def test_get_tool_schema_requires_non_empty_tool_names() -> None:
    discovery = ToolDiscoveryToolset()
    with pytest.raises(
        ValueError,
        match="`tool_names` must contain at least one tool name",
    ):
        await discovery._get_tool_schema_impl(MagicMock(), [])


async def test_call_tool_rejects_non_object_arguments() -> None:
    echo = _toolset(
        name="echo",
        description="Echo",
        result=lambda args: args["symbol"],
        schema={"type": "object", "properties": {"symbol": {"type": "string"}}},
    )
    discovery = ToolDiscoveryToolset(
        toolsets=[("symbols", cast(AbstractToolset[Any], echo))]
    )

    with pytest.raises(ValueError, match="must be an object/dictionary"):
        await discovery._call_tool_impl(MagicMock(), "echo", cast(Any, ["SILJ"]))


async def test_call_tool_external_tool_raises_call_deferred() -> None:
    external = ExternalToolset[Any](
        [
            ToolDefinition(
                name="widget_lookup",
                description="Lookup",
                parameters_json_schema={
                    "type": "object",
                    "properties": {"symbol": {"type": "string"}},
                },
            )
        ]
    )
    discovery = ToolDiscoveryToolset(
        toolsets=[("widgets", cast(AbstractToolset[Any], external))]
    )

    with pytest.raises(CallDeferred) as exc_info:
        await discovery._call_tool_impl(
            _build_run_context(),
            "widget_lookup",
            {"symbol": "SILJ"},
        )

    assert exc_info.value.metadata == {
        "tool_name": "widget_lookup",
        "arguments": {"symbol": "SILJ"},
    }


@pytest.mark.parametrize(
    ("calls", "expected_snippets"),
    [
        (
            {"tool_name": "add", "arguments": {"a": 2, "b": 3}},
            ["# Results", "## add", "\n5"],
        ),
        (
            [
                {"tool_name": "add", "arguments": {"a": 1, "b": 2}},
                {"tool_name": "mul", "arguments": {"a": 3, "b": 4}},
            ],
            ["# Results", "## add", "\n3\n", "## mul", "\n12"],
        ),
    ],
    ids=["single", "batch"],
)
async def test_call_tools_returns_markdown_for_immediate_calls(
    calls: list[dict[str, Any]] | dict[str, Any],
    expected_snippets: list[str],
) -> None:
    add = _toolset(
        name="add", description="Add", result=lambda args: args["a"] + args["b"]
    )
    mul = _toolset(
        name="mul", description="Multiply", result=lambda args: args["a"] * args["b"]
    )
    discovery = ToolDiscoveryToolset(
        toolsets=[
            ("math", cast(AbstractToolset[Any], add)),
            ("math", cast(AbstractToolset[Any], mul)),
        ]
    )

    result = await discovery._call_tools_impl(MagicMock(), calls)
    assert isinstance(result, str)
    for snippet in expected_snippets:
        assert snippet in result


async def test_call_tools_raises_call_deferred_for_deferred_batch() -> None:
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

    with pytest.raises(CallDeferred) as exc_info:
        await discovery._call_tools_impl(
            _build_run_context(),
            [
                {"tool_name": "widget_a", "arguments": {"symbol": "AAPL"}},
                {"tool_name": "widget_b", "arguments": {"symbol": "MSFT"}},
            ],
        )

    metadata = exc_info.value.metadata
    assert isinstance(metadata, dict)
    deferred_calls = metadata.get("deferred_calls")
    assert isinstance(deferred_calls, list)
    assert len(deferred_calls) == 2


async def test_call_tools_rejects_mixed_deferred_and_immediate_results() -> None:
    add = _toolset(
        name="add", description="Add", result=lambda args: args["a"] + args["b"]
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
            ("math", cast(AbstractToolset[Any], add)),
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


@pytest.mark.parametrize(
    "calls, error_text",
    [
        (cast(Any, "invalid"), "must be an object or a list of objects"),
        ([], "must contain at least one call"),
        ([{"arguments": {"a": 1}}], "must be an object with a `tool_name` key"),
    ],
    ids=["invalid_type", "empty", "missing_tool_name"],
)
async def test_call_tools_validates_input_shape(
    calls: Any,
    error_text: str,
) -> None:
    discovery = ToolDiscoveryToolset()
    with pytest.raises(ValueError, match=error_text):
        await discovery._call_tools_impl(MagicMock(), calls)


async def test_duplicate_tool_names_raise() -> None:
    first = _toolset(name="duplicate", description="First", result="ok")
    second = _toolset(name="duplicate", description="Second", result="ok")
    discovery = ToolDiscoveryToolset(
        toolsets=[
            ("first_group", cast(AbstractToolset[Any], first)),
            ("second_group", cast(AbstractToolset[Any], second)),
        ]
    )

    with pytest.warns(
        UserWarning, match="Duplicate tool name 'duplicate' in source 'second_group'"
    ):
        await discovery._resolve_pending(MagicMock())

    # Verify first registration was kept, second was skipped
    assert "duplicate" in discovery._registry
    assert discovery._registry["duplicate"].source_id == "first_group"


@pytest.mark.parametrize(
    ("tool_name", "args", "expected_result", "metadata_key"),
    [
        (
            "openbb_create_html",
            {"content": "<div>Hello</div>", "name": "Greeting"},
            "HTML artifact created successfully.",
            "html",
        ),
        (
            "openbb_create_chart",
            {
                "type": "line",
                "data": [{"period": "Q1", "value": 1}],
                "x_key": "period",
                "y_keys": ["value"],
                "name": "Sample",
            },
            "Chart created successfully.",
            "chart",
        ),
    ],
    ids=["html", "chart"],
)
async def test_tool_discovery_executes_viz_tools(
    tool_name: str,
    args: dict[str, Any],
    expected_result: str,
    metadata_key: str,
) -> None:
    viz_toolset = build_viz_toolsets()
    discovery = ToolDiscoveryToolset(
        toolsets=[("viz", cast(AbstractToolset[Any], viz_toolset))]
    )

    result = await discovery._call_tool_impl(_build_run_context(), tool_name, args)

    assert getattr(result, "return_value", None) == expected_result
    metadata = getattr(result, "metadata", None)
    assert isinstance(metadata, dict)
    assert metadata_key in metadata
