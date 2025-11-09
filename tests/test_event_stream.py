from __future__ import annotations

import asyncio
import json

import pytest
from openbb_ai.models import (
    DataContent,
    LlmClientFunctionCallResultMessage,
    LlmClientMessage,
    MessageChunkSSE,
    RoleEnum,
    SingleDataContent,
)
from pydantic_ai import DeferredToolRequests
from pydantic_ai.messages import (
    FunctionToolResultEvent,
    TextPart,
    ThinkingPart,
    ThinkingPartDelta,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.run import AgentRunResult, AgentRunResultEvent

from openbb_pydantic_ai import (
    GET_WIDGET_DATA_TOOL_NAME,
    OpenBBAIEventStream,
    build_widget_tool_name,
)


def test_event_stream_emits_widget_requests_and_citations(
    widget_collection, make_request
) -> None:
    widget = widget_collection.primary[0]
    tool_name = build_widget_tool_name(widget)
    request = make_request([LlmClientMessage(role=RoleEnum.human, content="Hi")])
    stream = OpenBBAIEventStream(
        run_input=request,
        widget_lookup={tool_name: widget},
    )

    deferred = DeferredToolRequests()
    deferred.calls.append(
        ToolCallPart(
            tool_name=tool_name, tool_call_id="call-1", args={"symbol": "AAPL"}
        )
    )
    run_result_event = AgentRunResultEvent(result=AgentRunResult(output=deferred))

    async def _run_tool_request():
        return [event async for event in stream.handle_run_result(run_result_event)]

    events = asyncio.run(_run_tool_request())

    assert events[0].event == "copilotStatusUpdate"
    assert "Sample Widget" in events[0].data.message
    assert events[1].event == "copilotFunctionCall"
    assert "call-1" in stream._pending_tool_calls  # type: ignore[attr-defined]

    tool_result_event = FunctionToolResultEvent(
        result=ToolReturnPart(
            tool_name=tool_name,
            tool_call_id="call-1",
            content={
                "data": [{"items": [{"content": json.dumps([{"col": 1}, {"col": 2}])}]}]
            },
        )
    )

    async def _run_tool_result():
        return [
            event
            async for event in stream.handle_function_tool_result(tool_result_event)
        ]

    tool_events = asyncio.run(_run_tool_result())

    assert tool_events[0].event == "copilotStatusUpdate"
    assert tool_events[0].data.artifacts
    assert tool_events[0].data.artifacts[0].type == "table"

    async def _run_after():
        return [event async for event in stream.after_stream()]

    after_events = asyncio.run(_run_after())
    assert any(e.event == "copilotCitationCollection" for e in after_events)


def test_artifact_detection_for_table(make_request) -> None:
    request = make_request(
        [LlmClientMessage(role=RoleEnum.human, content="Data please")]
    )
    stream = OpenBBAIEventStream(run_input=request, widget_lookup={})

    artifact = stream._artifact_from_output([{"col": 1}, {"col": 2}])
    assert artifact is not None
    assert artifact.event == "copilotMessageArtifact"
    assert artifact.data.type == "table"


def test_artifact_detection_normalizes_chart_payload(make_request) -> None:
    request = make_request([LlmClientMessage(role=RoleEnum.human, content="Chart")])
    stream = OpenBBAIEventStream(run_input=request, widget_lookup={})

    artifact = stream._artifact_from_output(
        {
            "type": "bar",
            "data": [{"label": "A", "value": 1}],
            "xKey": "label",
            "y_key": "value",
            "name": "My Chart",
        }
    )

    assert artifact is not None
    params = artifact.data.chart_params
    assert params.chartType == "bar"
    assert params.xKey == "label"
    assert params.yKey == ["value"]


@pytest.mark.parametrize(
    ("has_streamed_text", "output_text", "expected_in_after"),
    [
        (False, "Hello", True),
        (True, "Bob", False),
    ],
)
def test_final_output_handling(
    has_streamed_text, output_text, expected_in_after, make_request
):
    request = make_request([LlmClientMessage(role=RoleEnum.human, content="Hi")])
    stream = OpenBBAIEventStream(run_input=request, widget_lookup={})

    async def _run() -> None:
        if has_streamed_text:
            async for _ in stream.handle_text_start(TextPart(content=output_text)):
                pass
        run_result_event = AgentRunResultEvent(
            result=AgentRunResult(output=output_text)
        )
        async for _ in stream.handle_run_result(run_result_event):
            pass

    asyncio.run(_run())

    async def _collect_after():
        return [event async for event in stream.after_stream()]

    after_events = asyncio.run(_collect_after())
    if expected_in_after:
        assert after_events and isinstance(after_events[0], MessageChunkSSE)
        assert after_events[0].data.delta == output_text
    else:
        assert after_events == []


def test_thinking_events_emit_reasoning_step(make_request) -> None:
    request = make_request([LlmClientMessage(role=RoleEnum.human, content="Hi")])
    stream = OpenBBAIEventStream(run_input=request, widget_lookup={})

    async def _run() -> list:
        start_part = ThinkingPart(content="We must respond")
        assert [event async for event in stream.handle_thinking_start(start_part)] == []

        delta = ThinkingPartDelta(content_delta=" quickly.")
        assert [event async for event in stream.handle_thinking_delta(delta)] == []

        end_part = ThinkingPart(content="We must respond quickly.")
        end_events = [event async for event in stream.handle_thinking_end(end_part)]
        return end_events

    end_events = asyncio.run(_run())

    assert end_events[0].event == "copilotStatusUpdate"
    assert end_events[0].data.message == "Thinking"
    assert end_events[0].data.details
    assert "Thinking" in end_events[0].data.details[0]
    assert "respond quickly" in end_events[0].data.details[0]["Thinking"]


@pytest.mark.parametrize("use_get_widget_data", [False, True])
def test_deferred_results_emit_artifacts_and_citations(
    widget_collection, make_request, use_get_widget_data
) -> None:
    widget = widget_collection.primary[0]
    tool_name = build_widget_tool_name(widget)

    if use_get_widget_data:
        function_name = GET_WIDGET_DATA_TOOL_NAME
        input_args = {
            "data_sources": [
                {
                    "widget_uuid": str(widget.uuid),
                    "origin": widget.origin,
                    "id": widget.widget_id,
                    "input_args": {"symbol": "TSLA"},
                }
            ]
        }
    else:
        function_name = tool_name
        input_args = {"symbol": "AAPL"}

    result_message = LlmClientFunctionCallResultMessage(
        function=function_name,
        input_arguments=input_args,
        data=[
            DataContent(
                items=[SingleDataContent(content='[{"price": 150.0}]')],
            )
        ],
    )

    request = make_request([LlmClientMessage(role=RoleEnum.human, content="Hi")])
    stream = OpenBBAIEventStream(
        run_input=request,
        widget_lookup={tool_name: widget},
        pending_results=[result_message],
    )

    async def _collect_before():
        return [event async for event in stream.before_stream()]

    before_events = asyncio.run(_collect_before())
    status_events = [e for e in before_events if e.event == "copilotStatusUpdate"]
    assert status_events, "Expected reasoning status update"

    artifact_events = [e for e in status_events if e.data.artifacts]
    assert artifact_events
    assert artifact_events[0].data.artifacts[0].type == "table"

    async def _collect_after():
        return [event async for event in stream.after_stream()]

    after_events = asyncio.run(_collect_after())
    assert any(e.event == "copilotCitationCollection" for e in after_events)
