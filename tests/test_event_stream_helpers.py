from __future__ import annotations

from typing import Any

from openbb_ai.models import (
    DataContent,
    LlmClientFunctionCallResultMessage,
    SingleDataContent,
    Widget,
)

from openbb_pydantic_ai._config import GET_WIDGET_DATA_TOOL_NAME
from openbb_pydantic_ai._event_stream_helpers import (
    ToolCallInfo,
    extract_widget_args,
    handle_generic_tool_result,
    tool_result_events_from_content,
)
from openbb_pydantic_ai._widget_registry import WidgetRegistry


def _make_widget() -> Widget:
    return Widget(
        origin="OpenBB API",
        widget_id="sample_widget",
        name="Sample Widget",
        description="Widget used for testing.",
        params=[],
        metadata={},
    )


def _result_message(
    function: str, input_args: dict[str, Any]
) -> LlmClientFunctionCallResultMessage:
    return LlmClientFunctionCallResultMessage(
        function=function,
        input_arguments=input_args,
        data=[
            DataContent(
                items=[SingleDataContent(content='[{"value": 1}]')],
            )
        ],
    )


def test_find_widget_for_direct_result() -> None:
    widget = _make_widget()
    message = _result_message(widget.widget_id, {"symbol": "AAPL"})

    registry = WidgetRegistry()
    registry._by_tool_name[widget.widget_id] = widget
    found = registry.find_for_result(message)

    assert found is widget


def test_find_widget_for_get_widget_data_sources() -> None:
    widget = _make_widget()
    message = _result_message(
        GET_WIDGET_DATA_TOOL_NAME,
        {
            "data_sources": [
                {
                    "widget_uuid": str(widget.uuid),
                    "input_args": {"symbol": "TSLA"},
                }
            ]
        },
    )

    registry = WidgetRegistry()
    registry._by_uuid[str(widget.uuid)] = widget
    found = registry.find_for_result(message)

    assert found is widget


def test_extract_widget_args_prefers_data_sources_args() -> None:
    widget = _make_widget()
    args = {
        "data_sources": [
            {
                "widget_uuid": str(widget.uuid),
                "input_args": {"symbol": "TSLA"},
            }
        ]
    }
    message = _result_message(GET_WIDGET_DATA_TOOL_NAME, args)

    extracted = extract_widget_args(message)

    assert extracted == {"symbol": "TSLA"}


def test_tool_result_events_from_content_marks_streamed_text() -> None:
    mark_called = False

    def _mark() -> None:
        nonlocal mark_called
        mark_called = True

    events = tool_result_events_from_content(
        {
            "data": [
                {
                    "items": [
                        {
                            "content": '{"message": "hello"}',
                        }
                    ]
                }
            ]
        },
        mark_streamed_text=_mark,
    )

    assert mark_called, "Expected helper to flag streamed text when emitting chunks"
    assert events, "Expected message chunk event"


def test_handle_generic_tool_result_emits_artifacts() -> None:
    info = ToolCallInfo(tool_name="internal_tool", args={"symbol": "AAPL"})

    events = handle_generic_tool_result(
        info,
        content=[{"col": 1}, {"col": 2}],
        mark_streamed_text=lambda: None,
    )

    assert len(events) == 1
    status_event = events[0]
    assert status_event.event == "copilotStatusUpdate"
    assert status_event.data.artifacts
