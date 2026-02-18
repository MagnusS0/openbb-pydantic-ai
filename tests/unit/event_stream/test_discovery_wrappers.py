from __future__ import annotations

import pytest
from openbb_ai.helpers import chart
from openbb_ai.models import (
    AgentTool,
    LlmClientMessage,
    MessageArtifactSSE,
    MessageChunkSSE,
    RoleEnum,
    StatusUpdateSSE,
)
from pydantic_ai import DeferredToolRequests
from pydantic_ai.messages import (
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.run import AgentRunResult, AgentRunResultEvent

from openbb_pydantic_ai._config import (
    CHART_PLACEHOLDER_TOKEN,
    CHART_TOOL_NAME,
    EXECUTE_MCP_TOOL_NAME,
    GET_WIDGET_DATA_TOOL_NAME,
)
from openbb_pydantic_ai._event_stream import OpenBBAIEventStream
from openbb_pydantic_ai._widget_registry import WidgetRegistry
from openbb_pydantic_ai._widget_toolsets import build_widget_tool_name
from tests.helpers.event_stream_assertions import collect_events

pytestmark = pytest.mark.regression_contract


def _chart_artifact() -> MessageArtifactSSE:
    return chart(
        type="line",
        data=[{"period": "Q1", "value": 1}],
        x_key="period",
        y_keys=["value"],
        name="Sample",
    )


def _build_widget_stream(widget_collection, make_request):
    widget = widget_collection.primary[0]
    tool_name = build_widget_tool_name(widget)
    request = make_request([LlmClientMessage(role=RoleEnum.human, content="Hi")])
    registry = WidgetRegistry()
    registry._by_tool_name[tool_name] = widget
    return (
        OpenBBAIEventStream(run_input=request, widget_registry=registry),
        widget,
        tool_name,
    )


def test_widget_deferred_call_includes_tool_name_in_extra_state(
    widget_collection,
    make_request,
) -> None:
    stream, _widget, tool_name = _build_widget_stream(widget_collection, make_request)

    deferred = DeferredToolRequests()
    deferred.calls.append(
        ToolCallPart(
            tool_name=tool_name, tool_call_id="call-extra", args={"symbol": "AAPL"}
        )
    )

    events = collect_events(
        stream.handle_run_result(
            AgentRunResultEvent(result=AgentRunResult(output=deferred))
        )
    )
    function_calls = [e for e in events if e.event == "copilotFunctionCall"]
    assert len(function_calls) == 1

    extra_state = function_calls[0].data.extra_state
    assert extra_state is not None
    assert extra_state["tool_calls"][0]["tool_name"] == tool_name


@pytest.mark.parametrize(
    "use_data_source_envelope",
    [False, True],
    ids=["direct_widget_args", "wrapped_data_sources"],
)
def test_call_tools_widget_request_is_unwrapped(
    widget_collection,
    make_request,
    use_data_source_envelope: bool,
) -> None:
    stream, widget, tool_name = _build_widget_stream(widget_collection, make_request)

    if use_data_source_envelope:
        arguments = {
            "data_sources": [
                {
                    "widget_uuid": str(widget.uuid),
                    "origin": widget.origin,
                    "id": widget.widget_id,
                    "input_args": {"symbol": "AAPL"},
                }
            ]
        }
    else:
        arguments = {"symbol": "AAPL"}

    deferred = DeferredToolRequests()
    deferred.calls.append(
        ToolCallPart(
            tool_name="call_tools",
            tool_call_id="call-meta-widget",
            args={"calls": [{"tool_name": tool_name, "arguments": arguments}]},
        )
    )

    events = collect_events(
        stream.handle_run_result(
            AgentRunResultEvent(result=AgentRunResult(output=deferred))
        )
    )

    function_calls = [e for e in events if e.event == "copilotFunctionCall"]
    assert len(function_calls) == 1
    function_call = function_calls[0]
    assert function_call.data.function == GET_WIDGET_DATA_TOOL_NAME

    data_sources = function_call.data.input_arguments.get("data_sources", [])
    assert data_sources
    first_source = data_sources[0]
    input_args = (
        first_source["input_args"]
        if isinstance(first_source, dict)
        else getattr(first_source, "input_args", None)
    )
    assert input_args == {"symbol": "AAPL"}


@pytest.mark.parametrize(
    "use_execute_agent_envelope",
    [False, True],
    ids=["direct_mcp_args", "wrapped_execute_agent_args"],
)
def test_call_tools_mcp_request_is_unwrapped(
    make_request,
    use_execute_agent_envelope: bool,
) -> None:
    request = make_request([LlmClientMessage(role=RoleEnum.human, content="Hi")])
    agent_tool = AgentTool(
        server_id="1761162970409",
        name="equity_fundamental_income",
        url="http://localhost:8001/mcp/",
        description="Income statement",
    )
    stream = OpenBBAIEventStream(
        run_input=request, mcp_tools={agent_tool.name: agent_tool}
    )

    expected_parameters = {"provider": "fmp", "symbol": "AG"}
    if use_execute_agent_envelope:
        arguments = {
            "server_id": agent_tool.server_id,
            "tool_name": agent_tool.name,
            "parameters": expected_parameters,
        }
    else:
        arguments = expected_parameters

    deferred = DeferredToolRequests()
    deferred.calls.append(
        ToolCallPart(
            tool_name="call_tools",
            tool_call_id="call-meta-mcp",
            args={"calls": [{"tool_name": agent_tool.name, "arguments": arguments}]},
        )
    )

    events = collect_events(
        stream.handle_run_result(
            AgentRunResultEvent(result=AgentRunResult(output=deferred))
        )
    )

    function_calls = [e for e in events if e.event == "copilotFunctionCall"]
    assert len(function_calls) == 1
    function_call = function_calls[0]
    assert function_call.data.function == EXECUTE_MCP_TOOL_NAME
    assert function_call.data.input_arguments["tool_name"] == agent_tool.name
    assert function_call.data.input_arguments["parameters"] == expected_parameters


def test_call_tools_uses_nested_tool_name_for_reasoning(make_request) -> None:
    request = make_request([LlmClientMessage(role=RoleEnum.human, content="Hi")])
    stream = OpenBBAIEventStream(run_input=request)

    call_event = FunctionToolCallEvent(
        part=ToolCallPart(
            tool_name="call_tools",
            tool_call_id="meta-42",
            args={
                "calls": [
                    {
                        "tool_name": "internal_tool",
                        "arguments": {"query": "AAPL", "data": [{"value": 1}]},
                    }
                ]
            },
        )
    )

    call_events = collect_events(stream.handle_function_tool_call(call_event))
    assert call_events
    assert "internal_tool" in call_events[0].data.message
    assert "call_tools" not in call_events[0].data.message

    result_event = FunctionToolResultEvent(
        result=ToolReturnPart(
            tool_name="call_tools",
            tool_call_id="meta-42",
            content=[{"name": "address", "description": "Address"}],
        )
    )
    result_events = collect_events(stream.handle_function_tool_result(result_event))
    assert result_events
    assert isinstance(result_events[0], StatusUpdateSSE)
    assert "internal_tool" in result_events[0].data.message


def test_call_tools_multi_call_formats_reasoning_details(make_request) -> None:
    request = make_request([LlmClientMessage(role=RoleEnum.human, content="Hi")])
    stream = OpenBBAIEventStream(run_input=request)

    call_event = FunctionToolCallEvent(
        part=ToolCallPart(
            tool_name="call_tools",
            tool_call_id="meta-multi",
            args={
                "calls": [
                    {"tool_name": "tool_a", "arguments": {"symbol": "AAPL"}},
                    {"tool_name": "tool_b", "arguments": {"symbol": "MSFT"}},
                ]
            },
        )
    )

    call_events = collect_events(stream.handle_function_tool_call(call_event))
    assert call_events
    details = call_events[0].data.details
    assert details
    detail_row = details[0]
    assert detail_row["Tool count"] == "2"
    assert "tool_a" in detail_row["Tools"]
    assert "tool_b" in detail_row["Tools"]


def test_wrapped_chart_call_replaces_placeholder_inline(make_request) -> None:
    request = make_request([LlmClientMessage(role=RoleEnum.human, content="Chart")])
    stream = OpenBBAIEventStream(run_input=request)

    text_events = collect_events(
        stream.handle_text_start(
            TextPart(content=f"Intro {CHART_PLACEHOLDER_TOKEN} Outro")
        )
    )
    assert len(text_events) == 1
    assert isinstance(text_events[0], MessageChunkSSE)
    assert text_events[0].data.delta == "Intro "

    call_event = FunctionToolCallEvent(
        part=ToolCallPart(
            tool_name="call_tools",
            tool_call_id="chart-meta-1",
            args={
                "calls": [
                    {
                        "tool_name": CHART_TOOL_NAME,
                        "arguments": {
                            "type": "line",
                            "data": [{"period": "Q1", "value": 1}],
                            "x_key": "period",
                            "y_keys": ["value"],
                        },
                    }
                ]
            },
        )
    )
    collect_events(stream.handle_function_tool_call(call_event))

    chart_events = collect_events(
        stream.handle_function_tool_result(
            FunctionToolResultEvent(
                result=ToolReturnPart(
                    tool_name="call_tools",
                    tool_call_id="chart-meta-1",
                    content=None,
                    metadata={"chart": _chart_artifact()},
                )
            )
        )
    )

    assert len(chart_events) == 2
    assert isinstance(chart_events[0], MessageArtifactSSE)
    assert isinstance(chart_events[1], MessageChunkSSE)
    assert chart_events[1].data.delta == " Outro"


def test_expanded_deferred_calls_have_stable_tool_call_ids(
    widget_collection, make_request
) -> None:
    """Verify expanded deferred calls generate stable tool_call_ids.

    When call_tools wraps multiple nested tool calls, _expand_deferred_calls
    should create ToolCallPart instances with stable tool_call_ids derived
    from the parent call. This ensures results can be matched and processed
    correctly.
    """
    stream, _widget, tool_name = _build_widget_stream(widget_collection, make_request)

    # Create a deferred call with two nested widget calls
    deferred = DeferredToolRequests()
    deferred.calls.append(
        ToolCallPart(
            tool_name="call_tools",
            tool_call_id="call-parent-123",
            args={
                "calls": [
                    {"tool_name": tool_name, "arguments": {"symbol": "AAPL"}},
                    {"tool_name": tool_name, "arguments": {"symbol": "TSLA"}},
                ]
            },
        )
    )

    # Process the deferred request
    events = collect_events(
        stream.handle_run_result(
            AgentRunResultEvent(result=AgentRunResult(output=deferred))
        )
    )

    # Verify get_widget_data was called
    function_calls = [e for e in events if e.event == "copilotFunctionCall"]
    assert len(function_calls) == 1

    # Extract tool_call_ids from the extra_state
    extra_state = function_calls[0].data.extra_state
    assert extra_state is not None
    tool_calls = extra_state.get("tool_calls", [])
    assert len(tool_calls) == 2

    # Verify tool_call_ids follow the derived pattern
    assert tool_calls[0]["tool_call_id"] == "call-parent-123-0"
    assert tool_calls[1]["tool_call_id"] == "call-parent-123-1"

    # Now simulate results coming back with those tool_call_ids
    # and verify handle_function_tool_result processes them correctly
    result_event_1 = FunctionToolResultEvent(
        result=ToolReturnPart(
            tool_name=tool_name,
            tool_call_id="call-parent-123-0",
            content={"data": [{"symbol": "AAPL", "price": 150}]},
        )
    )

    result_events = collect_events(stream.handle_function_tool_result(result_event_1))
    # Should emit a status update (not be dropped)
    assert len(result_events) > 0
    status_updates = [e for e in result_events if e.event == "copilotStatusUpdate"]
    assert len(status_updates) > 0

    # Verify second result is also processed
    result_event_2 = FunctionToolResultEvent(
        result=ToolReturnPart(
            tool_name=tool_name,
            tool_call_id="call-parent-123-1",
            content={"data": [{"symbol": "TSLA", "price": 200}]},
        )
    )

    result_events_2 = collect_events(stream.handle_function_tool_result(result_event_2))
    assert len(result_events_2) > 0
    status_updates_2 = [e for e in result_events_2 if e.event == "copilotStatusUpdate"]
    assert len(status_updates_2) > 0
