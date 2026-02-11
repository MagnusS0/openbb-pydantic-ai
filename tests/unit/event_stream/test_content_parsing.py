from __future__ import annotations

import json
from typing import cast

import pytest
from openbb_ai.models import MessageChunkSSE, StatusUpdateSSE

from openbb_pydantic_ai._event_stream_helpers import (
    ToolCallInfo,
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
