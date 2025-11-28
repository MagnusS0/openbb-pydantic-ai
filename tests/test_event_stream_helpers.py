from __future__ import annotations

import json
from typing import Any, cast

from openbb_ai.models import (
    DataContent,
    LlmClientFunctionCallResultMessage,
    MessageChunkSSE,
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


def _raw_object_item(
    content: str,
    *,
    parse_as: str = "table",
    name: str | None = None,
    description: str | None = None,
) -> dict[str, Any]:
    item: dict[str, Any] = {
        "content": content,
        "data_format": {
            "data_type": "object",
            "parse_as": parse_as,
        },
    }
    if name:
        item["name"] = name
    if description:
        item["description"] = description
    return item


def test_find_widget_for_direct_result(sample_widget: Widget) -> None:
    widget = sample_widget
    message = _result_message(widget.widget_id, {"symbol": "AAPL"})

    registry = WidgetRegistry()
    registry._by_tool_name[widget.widget_id] = widget
    found = registry.find_for_result(message)

    assert found is widget


def test_find_widget_for_get_widget_data_sources(sample_widget: Widget) -> None:
    widget = sample_widget
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


def test_extract_widget_args_prefers_data_sources_args(sample_widget: Widget) -> None:
    widget = sample_widget
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


def test_extract_widget_args_falls_back_to_result_arguments() -> None:
    args = {"symbol": "NVDA"}
    message = _result_message(GET_WIDGET_DATA_TOOL_NAME, args)

    extracted = extract_widget_args(message)

    assert extracted == args


def test_tool_result_events_from_content_creates_table_for_dicts() -> None:
    mark_called = False

    def _mark() -> None:
        nonlocal mark_called
        mark_called = True

    events = tool_result_events_from_content(
        {
            "data": [
                {
                    "items": [
                        _raw_object_item(
                            '{"message": "hello"}',
                            name="Details",
                        )
                    ]
                }
            ]
        },
        mark_streamed_text=_mark,
    )

    assert events
    assert events[0].event == "copilotStatusUpdate"
    artifacts = events[0].data.artifacts
    assert artifacts and artifacts[0].type == "table"
    assert not mark_called


def test_tool_result_events_handle_double_encoded_lists() -> None:
    events = tool_result_events_from_content(
        {
            "data": [
                {
                    "items": [
                        _raw_object_item(
                            '["[{\\"field\\": \\"value\\"}]"]',
                        )
                    ]
                }
            ]
        },
        mark_streamed_text=lambda: None,
    )

    assert events
    status_event = events[0]
    assert status_event.event == "copilotStatusUpdate"
    artifacts = status_event.data.artifacts
    assert artifacts and artifacts[0].type == "table"
    rows = artifacts[0].content
    assert isinstance(rows, list)
    assert rows[0].get("field") == "value"


def test_tool_result_events_surface_function_call_errors() -> None:
    events = tool_result_events_from_content(
        {
            "data": [
                {
                    "error_type": "widget_error",
                    "content": "Widget failed to load because the symbol was invalid",
                }
            ]
        },
        mark_streamed_text=lambda: None,
    )

    assert len(events) == 1
    status_event = events[0]
    assert status_event.event == "copilotStatusUpdate"
    assert status_event.data.eventType == "ERROR"
    assert (
        status_event.data.message
        == "Widget failed to load because the symbol was invalid"
    )
    assert status_event.data.details
    assert status_event.data.details[0]["error_type"] == "widget_error"


def test_tool_result_events_expand_mapping_sections() -> None:
    payload = {
        "data": [
            {
                "items": [
                    _raw_object_item(
                        json.dumps(
                            {
                                "financial_ratios": [
                                    {"period": "Q1", "value": 1},
                                    {"period": "Q2", "value": 2},
                                ],
                                "tickers": [{"symbol": "JPM", "name": "JPMorgan"}],
                                "metadata": {"as_of": "2025-09-30"},
                            }
                        ),
                        name="Snapshot",
                        description="Ratios data",
                    )
                ]
            }
        ]
    }

    events = tool_result_events_from_content(payload, mark_streamed_text=lambda: None)

    assert events
    status_event = events[-1]
    assert status_event.event == "copilotStatusUpdate"
    artifacts = status_event.data.artifacts
    assert artifacts and len(artifacts) == 3
    names = [artifact.name for artifact in artifacts]
    assert names[0].endswith("financial_ratios")
    assert names[1].endswith("tickers")
    assert names[2].endswith("metadata")
    rows = artifacts[0].content
    assert isinstance(rows, list)
    assert rows[0]["period"] == "Q1"


def test_tool_result_events_handle_plain_text_items() -> None:
    mark_calls = 0

    def _mark() -> None:
        nonlocal mark_calls
        mark_calls += 1

    payload = {
        "data": [
            {
                "items": [
                    _raw_object_item(
                        "plain text result",
                        parse_as="text",
                        name="note",
                    )
                ]
            }
        ]
    }

    events = tool_result_events_from_content(payload, mark_streamed_text=_mark)

    assert len(events) == 1
    assert isinstance(events[0], MessageChunkSSE)
    assert events[0].data.delta == "plain text result"
    assert mark_calls == 1


def test_tool_result_events_table_parse_failure_streams_raw_text() -> None:
    events = tool_result_events_from_content(
        {
            "data": [
                {
                    "items": [
                        _raw_object_item("not-json", name="broken"),
                    ]
                }
            ]
        },
        mark_streamed_text=lambda: None,
    )

    assert len(events) == 2
    assert events[0].event == "copilotStatusUpdate"
    assert events[0].data.details
    assert events[0].data.details[0]["name"] == "broken"
    assert isinstance(events[1], MessageChunkSSE)
    assert events[1].data.delta == "not-json"


def test_tool_result_events_unsupported_format_emits_status() -> None:
    events = tool_result_events_from_content(
        {
            "data": [
                {
                    "items": [
                        {
                            "content": "ignored",
                            "data_format": {
                                "data_type": "xlsx",
                                "filename": "sheet.xlsx",
                            },
                        }
                    ]
                }
            ]
        },
        mark_streamed_text=lambda: None,
    )

    assert len(events) == 1
    assert events[0].event == "copilotStatusUpdate"
    assert "not implemented" in events[0].data.message


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


def test_tool_result_events_handle_list_of_strings_as_error() -> None:
    error_msg = (
        "Error calling tool 'query_database': (sqlite3.OperationalError) no such table"
    )
    content_json = json.dumps([error_msg])

    payload = {
        "data": [
            {
                "items": [
                    _raw_object_item(
                        content_json,
                        parse_as="json",
                    )
                ]
            }
        ]
    }

    events = tool_result_events_from_content(payload, mark_streamed_text=lambda: None)

    assert len(events) == 1
    status_event = events[0]
    assert status_event.event == "copilotStatusUpdate"
    assert status_event.data.eventType == "ERROR"
    assert error_msg in status_event.data.message
    assert status_event.data.details
    details_list = cast(list[dict[str, Any]], status_event.data.details)
    assert details_list[0]["errors"] == [error_msg]


def test_tool_result_events_handle_list_of_strings_as_text() -> None:
    """Non-error list of strings is streamed as raw text."""
    info_msg = "Some info message"
    content_json = json.dumps([info_msg])

    payload = {
        "data": [
            {
                "items": [
                    _raw_object_item(
                        content_json,
                        parse_as="json",
                    )
                ]
            }
        ]
    }

    mark_called = False

    def _mark() -> None:
        nonlocal mark_called
        mark_called = True

    events = tool_result_events_from_content(payload, mark_streamed_text=_mark)

    assert len(events) == 1
    chunk_event = events[0]
    assert isinstance(chunk_event, MessageChunkSSE)
    assert mark_called
