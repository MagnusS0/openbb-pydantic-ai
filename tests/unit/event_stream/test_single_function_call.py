from __future__ import annotations

from typing import Any, cast
from unittest.mock import MagicMock

import pytest
from openbb_ai.helpers import chart
from openbb_ai.models import (
    AgentTool,
    Citation,
    ClientCommandResult,
    FunctionCallSSE,
    LlmClientFunctionCall,
    LlmClientFunctionCallResultMessage,
    LlmClientMessage,
    MessageChunkSSE,
    RoleEnum,
    SourceInfo,
)
from pydantic_ai import DeferredToolRequests
from pydantic_ai.messages import (
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    PartStartEvent,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.run import AgentRunResult, AgentRunResultEvent

from openbb_pydantic_ai import OpenBBAIAdapter
from openbb_pydantic_ai._config import (
    CONTEXT_SNAPSHOT_EXTRA_STATE_KEY,
    EXECUTE_MCP_TOOL_NAME,
    GET_WIDGET_DATA_TOOL_NAME,
    LOCAL_TOOL_CAPSULE_EXTRA_STATE_KEY,
)
from openbb_pydantic_ai._event_stream import OpenBBAIEventStream
from openbb_pydantic_ai._widget_registry import WidgetRegistry
from openbb_pydantic_ai._widget_toolsets import build_widget_tool_name
from tests.helpers.event_stream_assertions import collect_events
from tests.unit.adapter._assertions import tool_call_parts, tool_return_parts

pytestmark = pytest.mark.regression_contract


class _RequestStub:
    headers = {
        "accept": "text/event-stream",
        "content-type": "application/json",
    }

    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    async def body(self) -> bytes:
        return self._payload


def _mcp_tool(name: str) -> AgentTool:
    return AgentTool(
        server_id="srv-1",
        name=name,
        url="http://localhost:8001/mcp/",
        description=name,
    )


def _build_stream(widget_collection, make_request) -> tuple[OpenBBAIEventStream, str]:
    widget = widget_collection.primary[0]
    tool_name = build_widget_tool_name(widget)
    request = make_request(
        [LlmClientMessage(role=RoleEnum.human, content="Hi")],
        widgets=widget_collection,
    )
    registry = WidgetRegistry()
    registry._by_tool_name[tool_name] = widget
    tools = {t.name: t for t in (_mcp_tool("mcp_a"), _mcp_tool("mcp_b"))}
    stream = OpenBBAIEventStream(
        run_input=request, widget_registry=registry, mcp_tools=tools
    )
    return stream, tool_name


def _deferred(*calls: tuple[str, str]) -> AgentRunResultEvent[Any]:
    deferred = DeferredToolRequests()
    for tool_name, tool_call_id in calls:
        deferred.calls.append(
            ToolCallPart(
                tool_name=tool_name, tool_call_id=tool_call_id, args={"symbol": "AAPL"}
            )
        )
    return AgentRunResultEvent(result=AgentRunResult(output=deferred))


def _function_calls(events: list[Any]) -> list[FunctionCallSSE]:
    return [e for e in events if isinstance(e, FunctionCallSSE)]


def test_widget_requests_win_over_mcp_calls(widget_collection, make_request) -> None:
    stream, widget_tool = _build_stream(widget_collection, make_request)

    events = collect_events(
        stream.handle_run_result(
            _deferred((widget_tool, "w-1"), ("mcp_a", "m-1"), ("mcp_b", "m-2"))
        )
    )

    calls = _function_calls(events)
    assert [c.data.function for c in calls] == [GET_WIDGET_DATA_TOOL_NAME]
    assert [tc["tool_call_id"] for tc in calls[0].data.extra_state["tool_calls"]] == [
        "w-1"
    ]
    assert not stream._state.has_tool_call("m-1")


def test_only_first_mcp_call_is_emitted(widget_collection, make_request) -> None:
    stream, _ = _build_stream(widget_collection, make_request)

    events = collect_events(
        stream.handle_run_result(_deferred(("mcp_a", "m-1"), ("mcp_b", "m-2")))
    )

    calls = _function_calls(events)
    assert [c.data.function for c in calls] == [EXECUTE_MCP_TOOL_NAME]
    assert calls[0].data.input_arguments["tool_name"] == "mcp_a"
    assert [tc["tool_call_id"] for tc in calls[0].data.extra_state["tool_calls"]] == [
        "m-1"
    ]


def test_second_deferred_batch_in_same_stream_is_dropped(
    widget_collection, make_request
) -> None:
    stream, _ = _build_stream(widget_collection, make_request)

    first = collect_events(stream.handle_run_result(_deferred(("mcp_a", "m-1"))))
    second = collect_events(stream.handle_run_result(_deferred(("mcp_b", "m-2"))))

    assert len(_function_calls(first)) == 1
    assert _function_calls(second) == []


def test_function_call_is_last_event_and_buffered_events_precede_it(
    widget_collection, make_request
) -> None:
    stream, widget_tool = _build_stream(widget_collection, make_request)
    stream._state.add_thinking("pondering")
    stream._queued_viz_artifacts.append(
        chart(
            type="line",
            data=[{"p": "Q1", "v": 1}],
            x_key="p",
            y_keys=["v"],
            name="Sample",
        )
    )
    stream._state.add_citation(
        Citation(source_info=SourceInfo(type="direct retrieval", name="doc"))
    )

    events = collect_events(stream.handle_run_result(_deferred((widget_tool, "w-1"))))
    events += collect_events(stream.after_stream())

    kinds = [e.event for e in events]
    assert kinds[-1] == "copilotFunctionCall"
    assert kinds.count("copilotFunctionCall") == 1
    thinking = next(
        i for i, e in enumerate(events) if getattr(e.data, "message", "") == "pondering"
    )
    artifact = kinds.index("copilotMessageArtifact")
    citation = kinds.index("copilotCitationCollection")
    assert thinking < artifact < citation < len(kinds) - 1


def test_function_call_snapshot_carries_request_context(
    widget_collection, make_request
) -> None:
    stream, widget_tool = _build_stream(widget_collection, make_request)

    events = collect_events(stream.handle_run_result(_deferred((widget_tool, "w-1"))))

    extra_state = _function_calls(events)[0].data.extra_state
    assert extra_state[CONTEXT_SNAPSHOT_EXTRA_STATE_KEY] == {"timezone": "UTC"}


@pytest.fixture
def mixed_response_call(widget_collection, make_request):
    stream, widget_tool = _build_stream(widget_collection, make_request)

    collect_events(
        stream.handle_function_tool_call(
            FunctionToolCallEvent(
                part=ToolCallPart(
                    tool_name="list_tools", tool_call_id="local-1", args={}
                )
            )
        )
    )
    collect_events(
        stream.handle_function_tool_result(
            FunctionToolResultEvent(
                part=ToolReturnPart(
                    tool_name="list_tools", tool_call_id="local-1", content="listing"
                )
            )
        )
    )
    events = collect_events(
        stream.handle_run_result(
            _deferred((widget_tool, "w-1"), ("mcp_a", "m-1"), ("mcp_b", "m-2"))
        )
    )
    return _function_calls(events)[0].data


async def test_follow_up_history_is_valid_after_mixed_response(
    widget_collection, make_request, mixed_response_call
) -> None:
    call = mixed_response_call
    assert LOCAL_TOOL_CAPSULE_EXTRA_STATE_KEY in call.extra_state

    request = make_request(
        [
            LlmClientMessage(role=RoleEnum.human, content="Hi"),
            LlmClientMessage(
                role=RoleEnum.ai,
                content=LlmClientFunctionCall(
                    function=call.function, input_arguments=call.input_arguments
                ),
            ),
            LlmClientFunctionCallResultMessage(
                function=call.function,
                input_arguments=call.input_arguments,
                data=[ClientCommandResult(status="success", message=None)],
                extra_state=call.extra_state,
            ),
        ],
        widgets=widget_collection,
    )
    adapter = await OpenBBAIAdapter.from_request(
        cast(Any, _RequestStub(request.model_dump_json().encode())),
        agent=MagicMock(),
    )

    call_ids = [p.tool_call_id for p in tool_call_parts(adapter)]
    return_ids = [p.tool_call_id for p in tool_return_parts(adapter)]
    assert call_ids == return_ids == ["local-1", "w-1"]


def test_events_after_mid_run_function_call_are_dropped(
    widget_collection, make_request
) -> None:
    stream, tool_name = _build_stream(widget_collection, make_request)
    deferred = DeferredToolRequests()
    deferred.calls.append(
        ToolCallPart(tool_name=tool_name, tool_call_id="w-1", args={"symbol": "AAPL"})
    )

    async def native_events():
        yield FunctionToolResultEvent(
            part=ToolReturnPart(
                tool_name="call_tools", content=deferred, tool_call_id="ct-1"
            )
        )
        yield PartStartEvent(index=0, part=TextPart(content="late text"))

    events = collect_events(stream.transform_stream(native_events()))

    assert isinstance(events[-1], FunctionCallSSE)
    assert not any(isinstance(e, MessageChunkSSE) for e in events)
