from __future__ import annotations

import json
from typing import Any, cast

import pytest
from openbb_ai.helpers import chart
from openbb_ai.models import (
    LlmClientMessage,
    MessageArtifactSSE,
    MessageChunkSSE,
    RoleEnum,
)
from pydantic_ai.messages import FunctionToolResultEvent, TextPart, ToolReturnPart

from openbb_pydantic_ai._config import (
    CHART_PLACEHOLDER_TOKEN,
    CHART_TOOL_NAME,
    HTML_TOOL_NAME,
)
from openbb_pydantic_ai._event_stream import OpenBBAIEventStream
from openbb_pydantic_ai._viz_toolsets import _html_artifact
from openbb_pydantic_ai._widget_registry import WidgetRegistry
from openbb_pydantic_ai._widget_toolsets import build_widget_tool_name
from tests.helpers.event_stream_assertions import (
    collect_events,
    find_status_with_artifacts,
)

pytestmark = pytest.mark.regression_contract


def _chart_artifact() -> MessageArtifactSSE:
    return chart(
        type="line",
        data=[{"period": "Q1", "value": 1}],
        x_key="period",
        y_keys=["value"],
        name="Sample",
    )


def test_widget_dict_results_render_tool_output(
    widget_collection, make_request
) -> None:
    widget = widget_collection.primary[0]
    tool_name = build_widget_tool_name(widget)
    request = make_request([LlmClientMessage(role=RoleEnum.human, content="Hi")])
    registry = WidgetRegistry()
    registry._by_tool_name[tool_name] = widget
    stream = OpenBBAIEventStream(run_input=request, widget_registry=registry)

    stream._state.register_tool_call(
        tool_call_id="call-dict",
        tool_name=tool_name,
        args={"symbol": "AAPL"},
        widget=widget,
    )

    dict_payload = {
        "symbol": "AAPL",
        "price": 272.41,
        "beta": 1.1,
    }

    result_event = FunctionToolResultEvent(
        result=ToolReturnPart(
            tool_name=tool_name,
            tool_call_id="call-dict",
            content={
                "data": [
                    {
                        "items": [
                            {
                                "name": "Ticker Info",
                                "description": "Quote snapshot",
                                "content": json.dumps(dict_payload),
                            }
                        ]
                    }
                ]
            },
        )
    )

    events = collect_events(stream.handle_function_tool_result(result_event))
    artifact_event = find_status_with_artifacts(events)
    artifacts = artifact_event.data.artifacts
    assert artifacts
    artifact = artifacts[0]
    assert artifact.type == "table"
    rows: dict[str, str] = {}
    for row in artifact.content:
        if not isinstance(row, dict):
            continue
        field = row.get("Field")
        value = row.get("Value")
        if isinstance(field, str) and isinstance(value, str):
            rows[field] = value
    assert rows.get("symbol") == "AAPL"
    assert rows.get("beta") == "1.1"
    price = rows.get("price")
    assert isinstance(price, str)
    assert price.startswith("272")


def test_artifact_detection_for_table(make_request) -> None:
    request = make_request(
        [LlmClientMessage(role=RoleEnum.human, content="Data please")]
    )
    stream = OpenBBAIEventStream(run_input=request)

    artifact = stream._artifact_from_output([{"col": 1}, {"col": 2}])
    assert artifact is not None
    assert artifact.event == "copilotMessageArtifact"
    message_artifact = cast(MessageArtifactSSE, artifact)
    assert message_artifact.data.type == "table"


def test_artifact_detection_normalizes_chart_payload(make_request) -> None:
    request = make_request([LlmClientMessage(role=RoleEnum.human, content="Chart")])
    stream = OpenBBAIEventStream(run_input=request)

    artifact = stream._artifact_from_output(
        {
            "type": "bar",
            "data": [{"label": "A", "value": 1}],
            "xKey": "label",
            "y_key": "value",
            "name": "My Chart",
        }
    )

    assert artifact is not None
    assert artifact.event == "copilotMessageArtifact"
    message_artifact = cast(MessageArtifactSSE, artifact)
    params = message_artifact.data.chart_params
    assert params is not None
    assert params.chartType == "bar"
    params_any = cast(Any, params)
    assert params_any.xKey == "label"
    assert params_any.yKey == ["value"]


def test_chart_tool_result_replaces_placeholder_inline(make_request) -> None:
    request = make_request([LlmClientMessage(role=RoleEnum.human, content="Chart")])
    stream = OpenBBAIEventStream(run_input=request)

    text_part = TextPart(content=f"Intro {CHART_PLACEHOLDER_TOKEN} Outro")
    text_events = collect_events(stream.handle_text_start(text_part))
    assert len(text_events) == 1
    assert isinstance(text_events[0], MessageChunkSSE)
    assert text_events[0].data.delta == "Intro "

    tool_event = FunctionToolResultEvent(
        result=ToolReturnPart(
            tool_name=CHART_TOOL_NAME,
            tool_call_id="chart-1",
            content=None,
            metadata={"chart": _chart_artifact()},
        )
    )

    chart_events = collect_events(stream.handle_function_tool_result(tool_event))
    assert len(chart_events) == 2
    assert isinstance(chart_events[0], MessageArtifactSSE)
    assert isinstance(chart_events[1], MessageChunkSSE)
    assert chart_events[1].data.delta == " Outro"


def test_chart_tool_result_without_placeholder_streams_artifact(make_request) -> None:
    request = make_request([LlmClientMessage(role=RoleEnum.human, content="Chart")])
    stream = OpenBBAIEventStream(run_input=request)

    text_events = collect_events(stream.handle_text_start(TextPart(content="Intro")))
    assert len(text_events) == 1
    assert isinstance(text_events[0], MessageChunkSSE)

    tool_event = FunctionToolResultEvent(
        result=ToolReturnPart(
            tool_name=CHART_TOOL_NAME,
            tool_call_id="chart-2",
            content=None,
            metadata={"chart": _chart_artifact()},
        )
    )

    chart_events = collect_events(stream.handle_function_tool_result(tool_event))
    assert chart_events == []

    after_events = collect_events(stream.after_stream())
    assert len(after_events) == 1
    assert isinstance(after_events[0], MessageArtifactSSE)


def test_html_tool_result_queues_artifact(make_request) -> None:
    request = make_request([LlmClientMessage(role=RoleEnum.human, content="HTML")])
    stream = OpenBBAIEventStream(run_input=request)

    html_artifact = _html_artifact(
        content="<div>Hello World</div>",
        name="Test HTML",
        description="Test description",
    )

    tool_event = FunctionToolResultEvent(
        result=ToolReturnPart(
            tool_name=HTML_TOOL_NAME,
            tool_call_id="html-1",
            content=None,
            metadata={"html": html_artifact},
        )
    )

    html_events = collect_events(stream.handle_function_tool_result(tool_event))
    # No placeholder, so nothing emitted immediately (queued for after_stream)
    assert html_events == []

    # Artifact emitted in after_stream (same as charts)
    after_events = collect_events(stream.after_stream())
    assert len(after_events) == 1
    assert isinstance(after_events[0], MessageArtifactSSE)
    assert after_events[0].data.type == "html"
    assert after_events[0].data.content == "<div>Hello World</div>"


@pytest.mark.parametrize(
    ("payload", "expected_content", "expected_name", "expected_description"),
    [
        (
            {
                "type": "html",
                "content": "<section>Section content</section>",
                "name": "Section",
                "description": "A section artifact",
            },
            "<section>Section content</section>",
            "Section",
            "A section artifact",
        ),
        (
            {
                "type": "html",
                "html": "<p>Paragraph</p>",
            },
            "<p>Paragraph</p>",
            "HTML Content",
            "HTML artifact",
        ),
    ],
    ids=["html_content_key", "html_html_key"],
)
def test_artifact_detection_for_html_output(
    make_request,
    payload: dict[str, str],
    expected_content: str,
    expected_name: str | None,
    expected_description: str | None,
) -> None:
    request = make_request(
        [LlmClientMessage(role=RoleEnum.human, content="Generate HTML")]
    )
    stream = OpenBBAIEventStream(run_input=request)

    artifact = stream._artifact_from_output(payload)

    assert artifact is not None
    assert artifact.event == "copilotMessageArtifact"
    assert isinstance(artifact, MessageArtifactSSE)
    assert artifact.data.type == "html"
    assert artifact.data.content == expected_content
    assert artifact.data.name == expected_name
    assert artifact.data.description == expected_description
