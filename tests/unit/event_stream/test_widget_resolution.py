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
    ("input_args", "expected"),
    [
        (
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
        ({"symbol": "NVDA"}, {"symbol": "NVDA"}),
    ],
    ids=["prefers_data_sources", "falls_back_to_result_arguments"],
)
def test_extract_widget_args(
    input_args: dict[str, Any], expected: dict[str, str]
) -> None:
    message = result_message(GET_WIDGET_DATA_TOOL_NAME, input_args)

    extracted = extract_widget_args(message)

    assert extracted == expected
