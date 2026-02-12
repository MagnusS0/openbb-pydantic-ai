from __future__ import annotations

import pytest
from openbb_ai.models import (
    LlmClientMessage,
    MessageChunkSSE,
    RoleEnum,
    StatusUpdateSSE,
)
from pydantic_ai.messages import (
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    ThinkingPart,
    ThinkingPartDelta,
    ToolCallPart,
    ToolReturnPart,
)

from openbb_pydantic_ai._event_stream import OpenBBAIEventStream
from openbb_pydantic_ai._widget_registry import WidgetRegistry
from openbb_pydantic_ai._widget_toolsets import build_widget_tool_name
from tests.helpers.event_stream_assertions import collect_events

pytestmark = pytest.mark.regression_contract


def test_non_widget_tool_calls_show_in_reasoning(make_request) -> None:
    request = make_request([LlmClientMessage(role=RoleEnum.human, content="Hi")])
    stream = OpenBBAIEventStream(run_input=request)

    call_event = FunctionToolCallEvent(
        part=ToolCallPart(
            tool_name="internal_tool",
            tool_call_id="tool-42",
            args={
                "query": "AAPL",
                "data": [{"value": 1}, {"value": 2}, {"value": 3}],
            },
        )
    )

    call_events = collect_events(stream.handle_function_tool_call(call_event))
    assert call_events
    assert call_events[0].event == "copilotStatusUpdate"
    assert "internal_tool" in call_events[0].data.message
    details = call_events[0].data.details
    assert details
    detail_entry = details[0]
    assert isinstance(detail_entry, dict)
    assert detail_entry["query"] == "AAPL"
    assert "list(len=3" in detail_entry["data"]

    result_event = FunctionToolResultEvent(
        result=ToolReturnPart(
            tool_name="internal_tool",
            tool_call_id="tool-42",
            content=[{"name": "address", "description": "Address"}],
        )
    )

    result_events = collect_events(stream.handle_function_tool_result(result_event))
    assert result_events
    assert len(result_events) == 1
    artifact_step = result_events[0]
    assert artifact_step.event == "copilotStatusUpdate"
    assert isinstance(artifact_step, StatusUpdateSSE)
    assert "internal_tool" in artifact_step.data.message
    assert artifact_step.data.artifacts


def test_thinking_events_emit_reasoning_step(make_request) -> None:
    request = make_request([LlmClientMessage(role=RoleEnum.human, content="Hi")])
    stream = OpenBBAIEventStream(run_input=request)

    start_part = ThinkingPart(content="We must respond")
    assert collect_events(stream.handle_thinking_start(start_part)) == []

    delta = ThinkingPartDelta(content_delta=" quickly.")
    assert collect_events(stream.handle_thinking_delta(delta)) == []

    end_part = ThinkingPart(content="We must respond quickly.")
    end_events = collect_events(stream.handle_thinking_end(end_part))

    assert end_events[0].event == "copilotStatusUpdate"
    assert end_events[0].data.message == "Thinking"
    assert end_events[0].data.details
    assert "Thinking" in end_events[0].data.details[0]
    assert "respond quickly" in end_events[0].data.details[0]["Thinking"]


def test_widget_result_without_structured_data_falls_back_to_reasoning(
    widget_collection, make_request
) -> None:
    widget = widget_collection.primary[0]
    tool_name = build_widget_tool_name(widget)
    request = make_request([LlmClientMessage(role=RoleEnum.human, content="Hi")])
    registry = WidgetRegistry()
    registry._by_tool_name[tool_name] = widget
    stream = OpenBBAIEventStream(
        run_input=request,
        widget_registry=registry,
    )

    stream._state.register_tool_call(
        tool_call_id="call-42",
        tool_name=tool_name,
        args={"symbol": "NFLX"},
        widget=widget,
    )

    result_event = FunctionToolResultEvent(
        result=ToolReturnPart(
            tool_name=tool_name,
            tool_call_id="call-42",
            content={"unexpected": "value"},
        )
    )

    events = collect_events(stream.handle_function_tool_result(result_event))
    assert events
    assert events[0].event == "copilotStatusUpdate"


def test_error_strings_from_tool_results_emit_status_updates(make_request) -> None:
    request = make_request([LlmClientMessage(role=RoleEnum.human, content="Hi")])
    stream = OpenBBAIEventStream(run_input=request)

    stream._state.register_tool_call(
        tool_call_id="call-error",
        tool_name="execute_agent_tool",
        args={},
    )

    error_content = {
        "input_arguments": {"tool_name": "database-connector_query_database"},
        "data": [
            {
                "items": [
                    {
                        "name": "error",
                        "content": "[\"Error calling tool 'query_database': boom\"]",
                        "data_format": {"parse_as": "json"},
                    }
                ]
            }
        ],
    }

    result_event = FunctionToolResultEvent(
        result=ToolReturnPart(
            tool_name="execute_agent_tool",
            tool_call_id="call-error",
            content=error_content,
        )
    )

    events = collect_events(stream.handle_function_tool_result(result_event))

    error_updates = [
        e
        for e in events
        if isinstance(e, StatusUpdateSSE) and e.data.eventType == "ERROR"
    ]
    assert error_updates
    assert not any(isinstance(e, MessageChunkSSE) for e in events)
