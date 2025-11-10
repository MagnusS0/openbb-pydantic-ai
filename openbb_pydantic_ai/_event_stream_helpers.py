"""Helper utilities for OpenBB event stream transformations."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal, Mapping, cast
from uuid import uuid4

from openbb_ai.helpers import chart, message_chunk, reasoning_step, table
from openbb_ai.models import (
    SSE,
    ClientArtifact,
    LlmClientFunctionCallResultMessage,
    MessageArtifactSSE,
    Widget,
)

from ._config import GET_WIDGET_DATA_TOOL_NAME
from ._event_builder import EventBuilder
from ._serializers import ContentSerializer
from ._types import SerializedContent, TextStreamCallback
from ._utils import (
    format_arg_value,
    format_args,
    get_str,
    get_str_list,
)


@dataclass(slots=True)
class ToolCallInfo:
    """Metadata captured when a tool call event is received.

    Attributes
    ----------
    tool_name : str
        Name of the tool being called
    args : dict[str, Any]
        Arguments passed to the tool
    widget : Widget | None
        Associated widget if this is a widget tool call, None otherwise
    """

    tool_name: str
    args: dict[str, Any]
    widget: Widget | None = None


def find_widget_for_result(
    result_message: LlmClientFunctionCallResultMessage,
    widget_lookup: Mapping[str, Widget],
) -> Widget | None:
    """Locate the widget that produced a deferred result message.

    Attempts to find the widget first by direct tool name match, then by
    checking if the result is from a get_widget_data call with a widget_uuid.

    Parameters
    ----------
    result_message : LlmClientFunctionCallResultMessage
        The result message to find a widget for
    widget_lookup : Mapping[str, Widget]
        Mapping from tool names to widgets

    Returns
    -------
    Widget | None
        The widget that produced the result, or None if not found
    """
    widget = widget_lookup.get(result_message.function)
    if widget is not None:
        return widget

    if result_message.function == GET_WIDGET_DATA_TOOL_NAME:
        data_sources = result_message.input_arguments.get("data_sources", [])
        if data_sources:
            data_source = data_sources[0]
            widget_uuid = data_source.get("widget_uuid")
            for candidate in widget_lookup.values():
                if str(candidate.uuid) == widget_uuid:
                    return candidate

    return None


def extract_widget_args(
    result_message: LlmClientFunctionCallResultMessage,
) -> dict[str, Any]:
    """Extract the arguments originally supplied to a widget invocation.

    For get_widget_data calls, extracts the input_args from the first data source.
    For direct widget calls, returns the input_arguments directly.

    Parameters
    ----------
    result_message : LlmClientFunctionCallResultMessage
        The result message to extract arguments from

    Returns
    -------
    dict[str, Any]
        The widget invocation arguments
    """
    if result_message.function == GET_WIDGET_DATA_TOOL_NAME:
        data_sources = result_message.input_arguments.get("data_sources", [])
        if data_sources:
            return data_sources[0].get("input_args", {})
    return result_message.input_arguments


def serialized_content_from_result(
    result_message: LlmClientFunctionCallResultMessage,
) -> SerializedContent:
    """Serialize a result message into structured content.

    This is a thin wrapper around ContentSerializer.serialize_result() with
    clearer intent for event stream processing.

    Parameters
    ----------
    result_message : LlmClientFunctionCallResultMessage
        The result message to serialize

    Returns
    -------
    SerializedContent
        Typed dictionary with input_arguments, data, and optional extra_state
    """
    return ContentSerializer.serialize_result(result_message)


def handle_generic_tool_result(
    info: ToolCallInfo,
    content: Any,
    *,
    mark_streamed_text: TextStreamCallback,
) -> list[SSE]:
    """Emit SSE events for a non-widget tool result.

    Attempts to parse the content and create appropriate SSE events. Falls back
    to reasoning steps with formatted details if content cannot be structured.

    Parameters
    ----------
    info : ToolCallInfo
        Metadata about the tool call
    content : Any
        The tool result content to process
    mark_streamed_text : TextStreamCallback
        Callback to mark that text has been streamed

    Returns
    -------
    list[SSE]
        List of SSE events representing the tool result
    """
    events = tool_result_events_from_content(
        content, mark_streamed_text=mark_streamed_text
    )
    if events:
        events.insert(0, reasoning_step(f"Tool '{info.tool_name}' returned"))
        return events

    artifact = artifact_from_output(content)
    if artifact is not None:
        if isinstance(artifact, MessageArtifactSSE):
            return [
                EventBuilder.reasoning_with_artifacts(
                    f"Tool '{info.tool_name}' returned",
                    [artifact.data],
                )
            ]
        return [
            reasoning_step(f"Tool '{info.tool_name}' returned"),
            artifact,
        ]

    details: dict[str, Any] | None = None
    if info.args:
        formatted = format_args(info.args)
        if formatted:
            details = formatted.copy()

    result_text = ContentSerializer.to_string(content)
    if result_text:
        details = details or {}
        details["Result"] = format_arg_value(content)

    return [
        reasoning_step(
            f"Tool '{info.tool_name}' returned",
            details=details,
        )
    ]


def tool_result_events_from_content(
    content: Any,
    *,
    mark_streamed_text: TextStreamCallback,
) -> list[SSE]:
    """Transform tool result payloads into SSE events.

    Processes structured content with a 'data' field containing items and
    converts them into appropriate SSE events (artifacts, message chunks, etc.).

    Parameters
    ----------
    content : Any
        The tool result content to transform
    mark_streamed_text : TextStreamCallback
        Callback to mark that text has been streamed

    Returns
    -------
    list[SSE]
        List of SSE events, may be empty if content is not structured
    """
    if not isinstance(content, dict):
        return []

    data_entries = content.get("data") or []
    if not isinstance(data_entries, list):
        return []

    events: list[SSE] = []
    artifacts: list[ClientArtifact] = []

    for entry in data_entries:
        if not isinstance(entry, dict):
            continue

        command_event = _process_command_result(entry)
        if command_event:
            events.append(command_event)

        entry_artifacts, entry_events = _process_data_items(entry, mark_streamed_text)
        artifacts.extend(entry_artifacts)
        events.extend(entry_events)

    if artifacts:
        events.append(
            EventBuilder.reasoning_with_artifacts("Data retrieved", artifacts)
        )

    return events





def artifact_from_output(output: Any) -> SSE | None:
    """Create an artifact SSE from generic tool output payloads.

    Detects and creates appropriate artifacts (charts or tables) from structured
    output. Supports various chart types (line, bar, scatter, pie, donut) and
    table formats.

    Parameters
    ----------
    output : Any
        The tool output to convert to an artifact. Can be:
        - dict with 'type' and 'data' for charts
        - dict with 'table' key for tables
        - list of dicts for automatic table creation

    Returns
    -------
    SSE | None
        A chart or table artifact event, or None if output format is not recognized

    Notes
    -----
    Chart types require specific keys:
    - line/bar/scatter: x_key and y_keys required
    - pie/donut: angle_key and callout_label_key required
    """
    if isinstance(output, dict):
        chart_type = output.get("type")
        data = output.get("data")

        if isinstance(chart_type, str) and chart_type in {
            "line",
            "bar",
            "scatter",
            "pie",
            "donut",
        }:
            rows = (
                [row for row in data or [] if isinstance(row, dict)]
                if isinstance(data, list)
                else []
            )
            if not rows:
                return None

            chart_type_literal = cast(
                Literal["line", "bar", "scatter", "pie", "donut"], chart_type
            )

            x_key = get_str(output, "x_key", "xKey")
            y_keys = get_str_list(output, "y_keys", "yKeys", "y_key", "yKey")
            angle_key = get_str(output, "angle_key", "angleKey")
            callout_label_key = get_str(output, "callout_label_key", "calloutLabelKey")

            if chart_type_literal in {"line", "bar", "scatter"}:
                if not x_key or not y_keys:
                    return None
            elif chart_type_literal in {"pie", "donut"}:
                if not angle_key or not callout_label_key:
                    return None

            return chart(
                type=chart_type_literal,
                data=rows,
                x_key=x_key,
                y_keys=y_keys,
                angle_key=angle_key,
                callout_label_key=callout_label_key,
                name=output.get("name"),
                description=output.get("description"),
            )

        table_data = None
        if isinstance(output.get("table"), list):
            table_data = output["table"]
        elif isinstance(data, list) and all(isinstance(item, dict) for item in data):
            table_data = data

        if table_data:
            return table(
                data=table_data,
                name=output.get("name"),
                description=output.get("description"),
            )

    if (
        isinstance(output, list)
        and output
        and all(isinstance(item, dict) for item in output)
    ):
        return table(data=output, name=None, description=None)

    return None


def _process_command_result(entry: dict[str, Any]) -> SSE | None:
    """Process command result status messages.

    Parameters
    ----------
    entry : dict[str, Any]
        Data entry potentially containing status and message fields

    Returns
    -------
    SSE | None
        Reasoning step event if status/message found, None otherwise
    """
    status = entry.get("status")
    message = entry.get("message")
    if status and message:
        return reasoning_step(f"[{status}] {message}")
    return None


def _process_data_items(
    entry: dict[str, Any], mark_streamed_text: TextStreamCallback
) -> tuple[list[ClientArtifact], list[SSE]]:
    """Process data items from a data entry into artifacts or SSE events.

    Parses JSON content from items and converts them into table artifacts
    for list-of-dicts data, or message chunks for other content types.

    Parameters
    ----------
    entry : dict[str, Any]
        Data entry containing an 'items' field with content
    mark_streamed_text : TextStreamCallback
        Callback to mark that text has been streamed

    Returns
    -------
    tuple[list[ClientArtifact], list[SSE]]
        Tuple of (artifacts created, events emitted)
    """
    items = entry.get("items")
    if not isinstance(items, list):
        return [], []

    artifacts: list[ClientArtifact] = []
    events: list[SSE] = []

    for item in items:
        if not isinstance(item, dict):
            continue

        raw_content = item.get("content")
        if not isinstance(raw_content, str):
            continue

        parsed = ContentSerializer.parse_json(raw_content)

        if (
            isinstance(parsed, list)
            and parsed
            and all(isinstance(row, dict) for row in parsed)
        ):
            artifacts.append(
                ClientArtifact(
                    type="table",
                    name=item.get("name") or f"Table_{uuid4().hex[:4]}",
                    description=item.get("description") or "Widget data",
                    content=parsed,
                )
            )
        elif isinstance(parsed, dict):
            mark_streamed_text()
            events.append(message_chunk(json.dumps(parsed)))
        else:
            mark_streamed_text()
            events.append(message_chunk(raw_content))

    return artifacts, events
