from __future__ import annotations

import pytest
from openbb_ai.models import (
    AgentTool,
    LlmClientMessage,
    RoleEnum,
)
from pydantic_ai import DeferredToolRequests
from pydantic_ai.messages import (
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.run import AgentRunResult, AgentRunResultEvent

from openbb_pydantic_ai._config import LOCAL_TOOL_CAPSULE_EXTRA_STATE_KEY
from openbb_pydantic_ai._event_stream import OpenBBAIEventStream
from openbb_pydantic_ai._local_tool_capsule import unpack_tool_history
from openbb_pydantic_ai._widget_registry import WidgetRegistry
from openbb_pydantic_ai._widget_toolsets import build_widget_tool_name
from tests.helpers.event_stream_assertions import collect_events

pytestmark = pytest.mark.regression_contract


def _build_stream(widget_collection, make_request) -> tuple[OpenBBAIEventStream, str]:
    widget = widget_collection.primary[0]
    tool_name = build_widget_tool_name(widget)
    request = make_request([LlmClientMessage(role=RoleEnum.human, content="Hi")])
    registry = WidgetRegistry()
    registry._by_tool_name[tool_name] = widget
    mcp_tool = AgentTool(
        server_id="srv-1",
        name="equity_fundamental_income",
        url="http://localhost:8001/mcp/",
        description="Income statement",
    )
    stream = OpenBBAIEventStream(
        run_input=request,
        widget_registry=registry,
        mcp_tools={mcp_tool.name: mcp_tool},
    )
    return stream, tool_name


def test_local_capsule_attached_once_and_preserves_tool_calls(
    widget_collection, make_request
) -> None:
    stream, widget_tool_name = _build_stream(widget_collection, make_request)

    collect_events(
        stream.handle_function_tool_call(
            FunctionToolCallEvent(
                part=ToolCallPart(
                    tool_name="list_tools",
                    tool_call_id="local-1",
                    args={"group": "openbb_viz_tools"},
                )
            )
        )
    )
    collect_events(
        stream.handle_function_tool_result(
            FunctionToolResultEvent(
                result=ToolReturnPart(
                    tool_name="list_tools",
                    tool_call_id="local-1",
                    content="# openbb_viz_tools\ncount: 1\n- openbb_create_chart",
                )
            )
        )
    )

    deferred = DeferredToolRequests()
    deferred.calls.append(
        ToolCallPart(
            tool_name=widget_tool_name,
            tool_call_id="widget-1",
            args={"symbol": "AAPL"},
        )
    )
    deferred.calls.append(
        ToolCallPart(
            tool_name="equity_fundamental_income",
            tool_call_id="mcp-1",
            args={"symbol": "AAPL"},
        )
    )

    events = collect_events(
        stream.handle_run_result(
            AgentRunResultEvent(result=AgentRunResult(output=deferred))
        )
    )
    function_calls = [event for event in events if event.event == "copilotFunctionCall"]
    assert len(function_calls) == 2

    widget_extra_state = function_calls[0].data.extra_state
    mcp_extra_state = function_calls[1].data.extra_state

    assert widget_extra_state is not None
    assert widget_extra_state["tool_calls"][0]["tool_call_id"] == "widget-1"
    assert widget_extra_state["tool_calls"][0]["tool_name"] == widget_tool_name
    assert LOCAL_TOOL_CAPSULE_EXTRA_STATE_KEY in widget_extra_state

    assert mcp_extra_state is not None
    assert mcp_extra_state["tool_calls"][0]["tool_call_id"] == "mcp-1"
    assert LOCAL_TOOL_CAPSULE_EXTRA_STATE_KEY not in mcp_extra_state

    entries = unpack_tool_history(
        widget_extra_state[LOCAL_TOOL_CAPSULE_EXTRA_STATE_KEY]
    )
    assert len(entries) == 1
    assert entries[0].tool_name == "list_tools"
    assert entries[0].tool_call_id == "local-1"
