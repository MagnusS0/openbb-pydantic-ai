from __future__ import annotations

from typing import Any

import pytest
from openbb_ai.models import Widget

from openbb_pydantic_ai._config import GET_WIDGET_DATA_TOOL_NAME
from openbb_pydantic_ai._event_stream_helpers import extract_widget_args
from openbb_pydantic_ai._widget_registry import WidgetRegistry
from tests.unit.event_stream._builders import result_message

pytestmark = pytest.mark.regression_contract


def test_find_widget_for_direct_result(sample_widget: Widget) -> None:
    widget = sample_widget
    message = result_message(widget.widget_id, {"symbol": "AAPL"})

    registry = WidgetRegistry()
    registry._by_tool_name[widget.widget_id] = widget
    found = registry.find_for_result(message)

    assert found is widget


def test_find_widget_for_get_widget_data_sources(sample_widget: Widget) -> None:
    widget = sample_widget
    message = result_message(
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


@pytest.mark.parametrize(
    "input_args",
    [
        {"data_sources": ["not_a_mapping"]},
        {"data_sources": [{"widget_uuid": 123}]},
        {"data_sources": [{"widget_uuid": None}]},
    ],
)
def test_find_widget_for_get_widget_data_ignores_invalid_sources(
    sample_widget: Widget,
    input_args: dict[str, Any],
) -> None:
    widget = sample_widget
    message = result_message(GET_WIDGET_DATA_TOOL_NAME, input_args)

    registry = WidgetRegistry()
    registry._by_uuid[str(widget.uuid)] = widget

    assert registry.find_for_result(message) is None


@pytest.mark.parametrize(
    ("function", "input_args", "expected"),
    [
        (
            GET_WIDGET_DATA_TOOL_NAME,
            {
                "data_sources": [
                    {
                        "widget_uuid": "abc",
                        "input_args": {"symbol": "TSLA"},
                    }
                ]
            },
            {"symbol": "TSLA"},
        ),
        (
            "openbb_widget_quote",
            {"symbol": "NVDA"},
            {"symbol": "NVDA"},
        ),
        (GET_WIDGET_DATA_TOOL_NAME, {"data_sources": ["not_a_mapping"]}, {}),
        (GET_WIDGET_DATA_TOOL_NAME, {"data_sources": [{"input_args": "bad"}]}, {}),
        (GET_WIDGET_DATA_TOOL_NAME, {"data_sources": [{}]}, {}),
    ],
    ids=[
        "prefers_data_sources",
        "falls_back_to_result_arguments",
        "rejects_non_mapping_data_source",
        "rejects_non_mapping_input_args",
        "rejects_missing_input_args",
    ],
)
def test_extract_widget_args(
    function: str,
    input_args: dict[str, Any],
    expected: dict[str, Any],
) -> None:
    message = result_message(function, input_args)

    extracted = extract_widget_args(message)

    assert extracted == expected
