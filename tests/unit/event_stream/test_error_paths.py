from __future__ import annotations

import json
from typing import Any, cast

import pytest
from openbb_ai.models import MessageChunkSSE, StatusUpdateSSE

from openbb_pydantic_ai._event_stream_helpers import tool_result_events_from_content
from tests.unit.event_stream._builders import raw_object_item

pytestmark = pytest.mark.regression_contract


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
    status_event = cast(StatusUpdateSSE, events[0])
    status_data = status_event.data
    assert status_event.event == "copilotStatusUpdate"
    assert status_data.eventType == "ERROR"
    assert status_data.message == "Widget failed to load because the symbol was invalid"
    assert status_data.details
    assert status_data.details[0]["error_type"] == "widget_error"


@pytest.mark.parametrize(
    ("items", "expect_error", "expected_message"),
    [
        (
            [
                (
                    "Error calling tool 'query_database': "
                    "(sqlite3.OperationalError) no such table"
                )
            ],
            True,
            (
                "Error calling tool 'query_database': "
                "(sqlite3.OperationalError) no such table"
            ),
        ),
        (["Some info message"], False, '["Some info message"]'),
    ],
    ids=["list_of_strings_error", "list_of_strings_text"],
)
def test_tool_result_events_handle_list_of_strings(
    items: list[str],
    expect_error: bool,
    expected_message: str,
) -> None:
    payload = {
        "data": [
            {
                "items": [
                    raw_object_item(
                        json.dumps(items),
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
    event = events[0]

    if expect_error:
        status_event = cast(StatusUpdateSSE, event)
        status_data = status_event.data
        assert status_event.event == "copilotStatusUpdate"
        assert status_data.eventType == "ERROR"
        assert expected_message in status_data.message
        assert status_data.details
        details_list = cast(list[dict[str, Any]], status_data.details)
        assert details_list[0]["errors"] == [expected_message]
        assert not mark_called
        return

    assert isinstance(event, MessageChunkSSE)
    assert event.data.delta == expected_message
    assert mark_called
