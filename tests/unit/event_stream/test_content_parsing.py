from __future__ import annotations

import json
from typing import Any, cast

import pytest
from openbb_ai.models import MessageChunkSSE, StatusUpdateSSE

from openbb_pydantic_ai._event_stream_helpers import (
    ToolCallInfo,
    _format_meta_tool_call_args,
    handle_generic_tool_result,
    tool_result_events_from_content,
)
from tests.unit.event_stream._builders import raw_object_item

pytestmark = pytest.mark.regression_contract


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
                        raw_object_item(
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
                        raw_object_item(
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


def test_tool_result_events_expand_mapping_sections() -> None:
    payload = {
        "data": [
            {
                "items": [
                    raw_object_item(
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
                    raw_object_item(
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
                        raw_object_item("not-json", name="broken"),
                    ]
                }
            ]
        },
        mark_streamed_text=lambda: None,
    )

    assert len(events) == 1
    status_event = cast(StatusUpdateSSE, events[0])
    status_data = status_event.data
    assert status_event.event == "copilotStatusUpdate"
    assert status_data.details
    assert status_data.details[0]["name"] == "broken"


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
    status_event = cast(StatusUpdateSSE, events[0])
    status_data = status_event.data
    assert status_event.event == "copilotStatusUpdate"
    assert "not implemented" in status_data.message


def test_handle_generic_tool_result_emits_artifacts() -> None:
    info = ToolCallInfo(tool_name="internal_tool", args={"symbol": "AAPL"})

    events = handle_generic_tool_result(
        info,
        content=[{"col": 1}, {"col": 2}],
        mark_streamed_text=lambda: None,
    )

    assert len(events) == 1
    status_event = cast(StatusUpdateSSE, events[0])
    status_data = status_event.data
    assert status_event.event == "copilotStatusUpdate"
    assert status_data.artifacts


def test_handle_generic_tool_result_formats_discovery_list_tools() -> None:
    info = ToolCallInfo(tool_name="list_tools", args={})

    events = handle_generic_tool_result(
        info,
        content={
            "openbb_create_chart": "Create chart artifacts",
            "openbb_create_table": "Create table artifacts",
        },
        mark_streamed_text=lambda: None,
    )

    assert len(events) == 1
    status_event = cast(StatusUpdateSSE, events[0])
    assert status_event.event == "copilotStatusUpdate"
    details = cast(list[dict[str, Any]], status_event.data.details)
    assert details
    detail_row = details[0]
    assert detail_row["Tool count"] == "2"
    assert "openbb_create_chart" in detail_row["Tools"]
    assert "Result" not in detail_row


def test_handle_generic_tool_result_formats_discovery_search_empty() -> None:
    info = ToolCallInfo(tool_name="search_tools", args={"query": "database"})

    events = handle_generic_tool_result(
        info,
        content={},
        mark_streamed_text=lambda: None,
    )

    assert len(events) == 1
    status_event = cast(StatusUpdateSSE, events[0])
    details = cast(list[dict[str, Any]], status_event.data.details)
    assert details
    detail_row = details[0]
    assert detail_row["Match count"] == "0"
    assert detail_row["Matches"] == "(none)"
    # Raw args should not leak into discovery details
    assert "query" not in detail_row


def test_handle_generic_tool_result_formats_discovery_tool_schema() -> None:
    info = ToolCallInfo(tool_name="get_tool_schema", args={"tool_name": "query_db"})
    schema_payload = json.dumps(
        {
            "name": "query_db",
            "group": "openbb_mcp_tools",
            "description": "Run a database query",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "limit": {"type": "integer"},
                },
                "required": ["query"],
            },
        }
    )

    events = handle_generic_tool_result(
        info,
        content=schema_payload,
        mark_streamed_text=lambda: None,
    )

    assert len(events) == 1
    status_event = cast(StatusUpdateSSE, events[0])
    details = cast(list[dict[str, Any]], status_event.data.details)
    assert details
    detail_row = details[0]
    assert detail_row["Name"] == "query_db"
    assert detail_row["Group"] == "openbb_mcp_tools"
    assert detail_row["Parameter count"] == "2"
    assert detail_row["Required count"] == "1"
    assert "query" in detail_row["Parameters"]
    assert "Result" not in detail_row


def test_handle_generic_tool_result_formats_discovery_multi_tool_schema() -> None:
    info = ToolCallInfo(
        tool_name="get_tool_schema",
        args={"tool_names": ["query_db", "search_news"]},
    )
    schema_payload = json.dumps(
        {
            "tools": [
                {
                    "name": "query_db",
                    "group": "openbb_mcp_tools",
                    "description": "Run a database query",
                    "parameters": {"type": "object", "properties": {}},
                },
                {
                    "name": "search_news",
                    "group": "openbb_mcp_tools",
                    "description": "Search recent news",
                    "parameters": {"type": "object", "properties": {}},
                },
            ],
            "count": 2,
        }
    )

    events = handle_generic_tool_result(
        info,
        content=schema_payload,
        mark_streamed_text=lambda: None,
    )

    assert len(events) == 1
    status_event = cast(StatusUpdateSSE, events[0])
    details = cast(list[dict[str, Any]], status_event.data.details)
    assert details
    detail_row = details[0]
    assert detail_row["Schema count"] == "2"
    assert "query_db" in detail_row["Tools"]
    assert "search_news" in detail_row["Tools"]
    assert "Result" not in detail_row


def test_tool_result_events_handle_list_of_strings_as_error() -> None:
    error_msg = (
        "Error calling tool 'query_database': (sqlite3.OperationalError) no such table"
    )
    content_json = json.dumps([error_msg])

    payload = {
        "data": [
            {
                "items": [
                    raw_object_item(
                        content_json,
                        parse_as="json",
                    )
                ]
            }
        ]
    }

    events = tool_result_events_from_content(payload, mark_streamed_text=lambda: None)

    assert len(events) == 1
    status_event = cast(StatusUpdateSSE, events[0])
    status_data = status_event.data
    assert status_event.event == "copilotStatusUpdate"
    assert status_data.eventType == "ERROR"
    assert error_msg in status_data.message
    assert status_data.details
    details_list = cast(list[dict[str, Any]], status_data.details)
    assert details_list[0]["errors"] == [error_msg]


def test_tool_result_events_handle_list_of_strings_as_text() -> None:
    """Non-error list of strings is streamed as raw text."""
    info_msg = "Some info message"
    content_json = json.dumps([info_msg])

    payload = {
        "data": [
            {
                "items": [
                    raw_object_item(
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


# --- _format_meta_tool_call_args tests ---


def test_format_meta_tool_call_args_call_tools_multi() -> None:
    args = {
        "calls": [
            {
                "tool_name": "openbb_widget_financial_statements",
                "arguments": {"symbol": "CDE", "period": "annual"},
            },
            {
                "tool_name": "openbb_widget_price_chart",
                "arguments": {"symbol": "CDE"},
            },
        ]
    }
    result = _format_meta_tool_call_args("call_tools", args)
    assert result is not None
    assert result["Tool count"] == "2"
    assert "openbb_widget_financial_statements" in result["Tools"]
    assert "openbb_widget_price_chart" in result["Tools"]
    assert 'symbol="CDE"' in result["Tools"]


def test_format_meta_tool_call_args_call_tools_single_formats_entry() -> None:
    """Single-call call_tools is unwrapped by _extract_effective_tool_call."""
    args = {
        "calls": [
            {"tool_name": "some_tool", "arguments": {"x": 1}},
        ]
    }
    # Single call still returns formatting (the caller decides via extract)
    result = _format_meta_tool_call_args("call_tools", args)
    assert result is not None
    assert result["Tool count"] == "1"


def test_format_meta_tool_call_args_get_tool_schema() -> None:
    args = {"tool_names": ["openbb_widget_financial_statements", "price_chart"]}
    result = _format_meta_tool_call_args("get_tool_schema", args)
    assert result is not None
    assert "openbb_widget_financial_statements" in result["Tools"]
    assert "price_chart" in result["Tools"]


# --- call_tools result formatting ---


def test_handle_generic_tool_result_formats_call_tools_results() -> None:
    info = ToolCallInfo(tool_name="call_tools", args={})

    content = [
        {"tool_name": "openbb_widget_financial_statements", "result": {"rows": 10}},
        {"tool_name": "openbb_widget_price_chart", "result": "ok"},
    ]

    events = handle_generic_tool_result(
        info,
        content=content,
        mark_streamed_text=lambda: None,
    )

    assert len(events) == 1
    status_event = cast(StatusUpdateSSE, events[0])
    details = cast(list[dict[str, Any]], status_event.data.details)
    assert details
    detail_row = details[0]
    assert detail_row["Result count"] == "2"
    assert "openbb_widget_financial_statements" in detail_row["Results"]
    assert "openbb_widget_price_chart" in detail_row["Results"]


def test_handle_generic_tool_result_discovery_no_raw_args_prefix() -> None:
    """Discovery meta-tool results should not include raw args in details."""
    info = ToolCallInfo(
        tool_name="get_tool_schema",
        args={"tool_names": ["openbb_widget_financial_statements"]},
    )
    schema_payload = json.dumps(
        {
            "tools": [
                {
                    "name": "openbb_widget_financial_statements",
                    "description": "Financial statements",
                    "parameters": {"type": "object", "properties": {}},
                }
            ],
            "count": 1,
        }
    )

    events = handle_generic_tool_result(
        info,
        content=schema_payload,
        mark_streamed_text=lambda: None,
    )

    assert len(events) == 1
    status_event = cast(StatusUpdateSSE, events[0])
    details = cast(list[dict[str, Any]], status_event.data.details)
    assert details
    detail_row = details[0]
    # Should have discovery details, not raw args
    assert "Schema count" in detail_row
    assert "tool_names" not in detail_row
