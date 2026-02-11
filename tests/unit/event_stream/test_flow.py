from __future__ import annotations

import json

import pytest
from openbb_ai.models import (
    Citation,
    DataContent,
    LlmClientFunctionCallResultMessage,
    LlmClientMessage,
    MessageChunkSSE,
    RoleEnum,
    SingleDataContent,
    SourceInfo,
)
from pydantic_ai import DeferredToolRequests
from pydantic_ai.messages import (
    FunctionToolResultEvent,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.run import AgentRunResult, AgentRunResultEvent

from openbb_pydantic_ai._config import GET_WIDGET_DATA_TOOL_NAME
from openbb_pydantic_ai._event_stream import OpenBBAIEventStream
from openbb_pydantic_ai._utils import format_args
from openbb_pydantic_ai._widget_registry import WidgetRegistry
from openbb_pydantic_ai._widget_toolsets import build_widget_tool_name
from tests.helpers.event_stream_assertions import (
    collect_events,
    find_status_with_artifacts,
)

pytestmark = pytest.mark.regression_contract


def test_event_stream_emits_widget_requests_and_citations(
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

    deferred = DeferredToolRequests()
    deferred.calls.append(
        ToolCallPart(
            tool_name=tool_name, tool_call_id="call-1", args={"symbol": "AAPL"}
        )
    )
    run_result_event = AgentRunResultEvent(result=AgentRunResult(output=deferred))

    events = collect_events(stream.handle_run_result(run_result_event))

    assert events[0].event == "copilotStatusUpdate"
    assert "Sample Widget" in events[0].data.message
    assert events[1].event == "copilotFunctionCall"
    assert stream._state.has_tool_call("call-1")

    tool_result_event = FunctionToolResultEvent(
        result=ToolReturnPart(
            tool_name=tool_name,
            tool_call_id="call-1",
            content={
                "data": [{"items": [{"content": json.dumps([{"col": 1}, {"col": 2}])}]}]
            },
        )
    )

    tool_events = collect_events(stream.handle_function_tool_result(tool_result_event))

    assert tool_events[0].event == "copilotStatusUpdate"
    artifact_event = find_status_with_artifacts(tool_events)
    artifacts = artifact_event.data.artifacts
    assert artifacts
    assert artifacts[0].type == "table"

    after_events = collect_events(stream.after_stream())
    citation_events = [
        e for e in after_events if e.event == "copilotCitationCollection"
    ]
    assert citation_events
    first_citation = citation_events[0].data.citations[0]
    assert first_citation.source_info.metadata.get("input_args") == {"symbol": "AAPL"}
    assert first_citation.details == [format_args({"symbol": "AAPL"})]


def test_non_widget_metadata_citations_are_emitted(make_request) -> None:
    request = make_request([LlmClientMessage(role=RoleEnum.human, content="Hi")])
    stream = OpenBBAIEventStream(run_input=request)

    stream._state.register_tool_call(
        tool_call_id="pdf-call-1",
        tool_name="pdf_query",
        args={"doc_id": "abc123"},
    )

    citation = Citation(
        source_info=SourceInfo(
            type="direct retrieval",
            name="report.pdf",
            metadata={"doc_id": "abc123"},
            citable=True,
        ),
        details=[{"section": "Introduction"}],
    )
    result_event = FunctionToolResultEvent(
        result=ToolReturnPart(
            tool_name="pdf_query",
            tool_call_id="pdf-call-1",
            content="## Introduction\n\nContent",
            metadata={"citations": [citation]},
        )
    )

    events = collect_events(stream.handle_function_tool_result(result_event))
    assert events

    after_events = collect_events(stream.after_stream())
    citation_events = [
        event for event in after_events if event.event == "copilotCitationCollection"
    ]
    assert citation_events
    assert (
        citation_events[0].data.citations[0].source_info.metadata["doc_id"] == "abc123"
    )


@pytest.mark.parametrize(
    ("has_streamed_text", "output_text", "expected_in_after"),
    [
        (False, "Hello", True),
        (True, "Bob", False),
    ],
)
def test_final_output_handling(
    has_streamed_text: bool, output_text: str, expected_in_after: bool, make_request
) -> None:
    request = make_request([LlmClientMessage(role=RoleEnum.human, content="Hi")])
    stream = OpenBBAIEventStream(run_input=request)

    if has_streamed_text:
        collect_events(stream.handle_text_start(TextPart(content=output_text)))

    run_result_event = AgentRunResultEvent(result=AgentRunResult(output=output_text))
    collect_events(stream.handle_run_result(run_result_event))

    after_events = collect_events(stream.after_stream())
    if expected_in_after:
        assert after_events and isinstance(after_events[0], MessageChunkSSE)
        assert after_events[0].data.delta == output_text
    else:
        assert after_events == []


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
        expected_args = {"symbol": "TSLA"}
    else:
        function_name = tool_name
        input_args = {"symbol": "AAPL"}
        expected_args = {"symbol": "AAPL"}

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
    registry = WidgetRegistry()
    registry._by_tool_name[tool_name] = widget
    registry._by_uuid[str(widget.uuid)] = widget
    stream = OpenBBAIEventStream(
        run_input=request,
        widget_registry=registry,
        pending_results=[result_message],
    )

    before_events = collect_events(stream.before_stream())
    status_events = [e for e in before_events if e.event == "copilotStatusUpdate"]
    assert status_events

    artifact_event = find_status_with_artifacts(status_events)
    artifacts = artifact_event.data.artifacts
    assert artifacts
    assert artifacts[0].type == "table"

    after_events = collect_events(stream.after_stream())
    citation_events = [
        e for e in after_events if e.event == "copilotCitationCollection"
    ]
    assert citation_events
    citation = citation_events[0].data.citations[0]
    assert citation.source_info.metadata.get("input_args") == expected_args
    assert citation.details == [format_args(expected_args)]


def test_deferred_result_without_widget_metadata_is_streamed(make_request) -> None:
    result_message = LlmClientFunctionCallResultMessage(
        function="orphan_widget",
        input_arguments={"symbol": "MSFT"},
        data=[
            DataContent(
                items=[SingleDataContent(content='[{"value": 1}]')],
            )
        ],
    )

    request = make_request([LlmClientMessage(role=RoleEnum.human, content="Hi")])
    stream = OpenBBAIEventStream(
        run_input=request,
        pending_results=[result_message],
    )

    events = collect_events(stream.before_stream())
    warnings = [
        e
        for e in events
        if getattr(e, "event", None) == "copilotStatusUpdate"
        and "metadata" in getattr(getattr(e, "data", object()), "message", "")
    ]
    assert warnings
    artifact_event = find_status_with_artifacts(events)
    assert artifact_event.data.artifacts
