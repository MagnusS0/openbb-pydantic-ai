"""Helper utilities for OpenBB event stream transformations."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal, Mapping, cast
from uuid import uuid4

from openbb_ai.helpers import chart, message_chunk, reasoning_step, table
from openbb_ai.models import (
    SSE,
    AgentTool,
    ClientArtifact,
    LlmClientFunctionCallResultMessage,
    MessageArtifactSSE,
    StatusUpdateSSE,
    StatusUpdateSSEData,
    Widget,
)

from openbb_pydantic_ai._viz_toolsets import _html_artifact

from ._config import EVENT_TYPE_ERROR, GET_WIDGET_DATA_TOOL_NAME
from ._event_stream_formatters import (
    _format_discovery_meta_result,
)
from ._serializers import parse_json, to_string
from ._types import TextStreamCallback
from ._utils import (
    format_arg_value,
    format_args,
    get_str,
    get_str_list,
)


@dataclass(slots=True, frozen=True)
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
    agent_tool : AgentTool | None
        Agent tool metadata when the call targets an MCP tool
    """

    tool_name: str
    args: dict[str, Any]
    widget: Widget | None = None
    agent_tool: AgentTool | None = None


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
    discovery_details = _format_discovery_meta_result(info.tool_name, content)
    if discovery_details:
        return [
            reasoning_step(
                f"Tool '{info.tool_name}' returned",
                details=discovery_details,
            )
        ]

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
                StatusUpdateSSE(
                    data=StatusUpdateSSEData(
                        eventType="INFO",
                        message=f"Tool '{info.tool_name}' returned",
                        group="reasoning",
                        artifacts=[artifact.data],
                    )
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

    result_text = to_string(content)
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
    widget_entries: list[tuple[Widget | None, dict[str, Any]]] | None = None,
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
    widget_entries : list[tuple[Widget | None, dict[str, Any]]] | None
        Optional list of widget entries corresponding to the content items.
        Used to resolve widget names when missing from the content.

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

    for i, entry in enumerate(data_entries):
        if not isinstance(entry, dict):
            continue

        command_event = _process_command_result(entry)
        if command_event:
            events.append(command_event)

        error_event = _process_error_result(entry)
        if error_event:
            events.append(error_event)
            continue

        # Determine context for this entry
        current_entries = None
        default_widget = None

        if widget_entries:
            if len(data_entries) == 1:
                # Single entry, items inside map to widget_entries
                current_entries = widget_entries
            elif len(data_entries) == len(widget_entries):
                # Multiple entries, assume 1-to-1 mapping
                default_widget = widget_entries[i][0]

        entry_artifacts, entry_events = _process_data_items(
            entry,
            mark_streamed_text,
            widget_entries=current_entries,
            default_widget=default_widget,
        )
        artifacts.extend(entry_artifacts)
        events.extend(entry_events)

    if artifacts:
        events.append(
            StatusUpdateSSE(
                data=StatusUpdateSSEData(
                    eventType="INFO",
                    message="Data retrieved",
                    group="reasoning",
                    artifacts=artifacts,
                )
            )
        )

    return events


def artifact_from_output(output: Any) -> SSE | None:
    """Create an artifact SSE from generic tool output payloads.

    Detects and creates appropriate artifacts (charts, tables, or HTML) from
    structured output. Supports various chart types (line, bar, scatter, pie,
    donut), table formats, and HTML content.

    Parameters
    ----------
    output : Any
        The tool output to convert to an artifact. Can be:
        - dict with 'type': 'html' and 'content' for HTML artifacts
        - dict with 'type' and 'data' for charts
        - dict with 'table' key for tables
        - list of dicts for automatic table creation

    Returns
    -------
    SSE | None
        A chart, table, or HTML artifact event, or None if output format is
        not recognized

    Notes
    -----
    Chart types require specific keys:
    - line/bar/scatter: x_key and y_keys required
    - pie/donut: angle_key and callout_label_key required
    """
    if isinstance(output, dict):
        output_type = output.get("type")

        # Check for HTML artifact
        if output_type == "html":
            content = output.get("content") or output.get("html")
            if isinstance(content, str):
                return _html_artifact(
                    content=content,
                    name=output.get("name"),
                    description=output.get("description"),
                )

        chart_type = output_type
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


def _process_error_result(entry: dict[str, Any]) -> SSE | None:
    """Process error result messages.

    Parameters
    ----------
    entry : dict[str, Any]
        Data entry potentially containing error_type and content fields

    Returns
    -------
    SSE | None
        Reasoning step event if error found, None otherwise
    """
    error_type = entry.get("error_type")
    content = entry.get("content")
    if error_type and content:
        return StatusUpdateSSE(
            data=StatusUpdateSSEData(
                eventType="ERROR",
                message=str(content),
                details=[{"error_type": error_type}],
            )
        )
    return None


def _looks_like_error_text(text: str) -> bool:
    """Heuristic to decide if a string represents an error message."""

    lowered = text.strip().lower()
    return (
        lowered.startswith("error")
        or lowered.startswith("exception")
        or "traceback" in lowered
    )


def _extract_error_messages(value: Any) -> list[str] | None:
    """Extract error-looking strings from common payload shapes."""

    if isinstance(value, dict):
        candidate = value.get("error") or value.get("message")
        if isinstance(candidate, str) and _looks_like_error_text(candidate):
            return [candidate]

    if isinstance(value, str) and _looks_like_error_text(value):
        return [value]

    if (
        isinstance(value, list)
        and value
        and all(isinstance(item, str) for item in value)
    ):
        errors = [item for item in value if _looks_like_error_text(item)]
        if errors:
            return errors

    return None


def _dict_items_to_rows(data: Mapping[str, Any]) -> list[dict[str, str]]:
    """Convert a mapping into key/value rows for table rendering."""

    rows: list[dict[str, str]] = []
    for key, value in data.items():
        rows.append(
            {
                "Field": str(key),
                "Value": format_arg_value(value),
            }
        )
    return rows


def _try_expand_mapping(
    data: dict[str, Any],
    base_name: str | None,
    base_description: str | None,
) -> list[ClientArtifact] | None:
    """Attempt to expand a dictionary into multiple table artifacts."""

    should_expand = False
    for value in data.values():
        if _extract_table_rows(value) is not None:
            should_expand = True
            break
        if isinstance(value, dict):
            should_expand = True
            break

    if not should_expand:
        return None

    artifacts: list[ClientArtifact] = []
    for key, value in data.items():
        name = f"{base_name}_{key}" if base_name else key
        description = f"{base_description} - {key}" if base_description else key

        rows = _extract_table_rows(value)
        if rows:
            artifacts.append(
                ClientArtifact(
                    type="table",
                    name=name,
                    description=description,
                    content=rows,
                )
            )
        elif isinstance(value, dict):
            kv_rows = _dict_items_to_rows(value)
            if kv_rows:
                artifacts.append(
                    ClientArtifact(
                        type="table",
                        name=name,
                        description=description,
                        content=kv_rows,
                    )
                )
        else:
            artifacts.append(
                ClientArtifact(
                    type="table",
                    name=name,
                    description=description,
                    content=[{"Field": "Value", "Value": format_arg_value(value)}],
                )
            )

    return artifacts


def _resolve_widget_name(
    item: dict[str, Any],
    widget_entries: list[tuple[Widget | None, dict[str, Any]]] | None,
    default_widget: Widget | None,
    idx: int,
) -> str | None:
    """Resolve the name for a data item."""
    name = item.get("name")

    if name:
        return name

    resolved_widget = None
    if default_widget:
        resolved_widget = default_widget
    elif widget_entries and idx < len(widget_entries):
        resolved_widget = widget_entries[idx][0]

    if resolved_widget:
        return resolved_widget.name

    return None


def _parse_item_content(
    raw_content: str,
    display_name: str,
) -> tuple[Any, StatusUpdateSSE | None]:
    """Parse raw content string into structured data or return an error event."""
    try:
        parsed = _decode_nested_json(json.loads(raw_content))
        return parsed, None
    except (json.JSONDecodeError, ValueError):
        error_messages = _extract_error_messages(raw_content)
        if error_messages:
            return None, StatusUpdateSSE(
                data=StatusUpdateSSEData(
                    eventType=EVENT_TYPE_ERROR,
                    message=error_messages[0],
                    details=[{"errors": error_messages}],
                )
            )

        return None, StatusUpdateSSE(
            data=StatusUpdateSSEData(
                eventType="WARNING",
                message=f"Failed to parse content for '{display_name}'",
                details=[{"name": display_name}],
            )
        )


def _item_to_artifact(
    parsed: Any,
    name: str | None,
    description: str | None,
) -> tuple[list[ClientArtifact], SSE | None]:
    """Convert parsed data into artifacts or error event."""
    error_messages = _extract_error_messages(parsed)
    if error_messages:
        return [], StatusUpdateSSE(
            data=StatusUpdateSSEData(
                eventType=EVENT_TYPE_ERROR,
                message=error_messages[0],
                details=[{"errors": error_messages}],
            )
        )

    # Check for HTML artifact structure in parsed data
    if isinstance(parsed, dict) and parsed.get("type") == "html":
        html_content = parsed.get("content") or parsed.get("html")
        if isinstance(html_content, str):
            return [
                ClientArtifact(
                    type="html",
                    name=parsed.get("name") or name or "HTML Content",
                    description=parsed.get("description")
                    or description
                    or "HTML artifact",
                    content=html_content,
                )
            ], None

    table_rows = _extract_table_rows(parsed)
    if table_rows is not None:
        return [
            ClientArtifact(
                type="table",
                name=name or f"Table_{uuid4().hex[:4]}",
                description=description or "Widget data",
                content=table_rows,
            )
        ], None

    if isinstance(parsed, dict):
        expanded = _try_expand_mapping(parsed, name, description)
        if expanded:
            return expanded, None

        rows = _dict_items_to_rows(parsed)
        if rows:
            return [
                ClientArtifact(
                    type="table",
                    name=name or f"Details_{uuid4().hex[:4]}",
                    description=description or "Widget data",
                    content=rows,
                )
            ], None

    return [], None


def _process_data_items(
    entry: dict[str, Any],
    mark_streamed_text: TextStreamCallback,
    widget_entries: list[tuple[Widget | None, dict[str, Any]]] | None = None,
    default_widget: Widget | None = None,
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
    widget_entries : list[tuple[Widget | None, dict[str, Any]]] | None
        Optional list of widget entries to resolve missing names
    default_widget : Widget | None
        Optional default widget to use if name is missing and widget_entries
        cannot be used for resolution.

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

    html_widget_ids: set[str] = set()
    if widget_entries:
        for widget, _ in widget_entries:
            if widget and isinstance(widget.widget_id, str):
                if widget.widget_id.startswith("html-"):
                    html_widget_ids.add(widget.widget_id)
    if default_widget and isinstance(default_widget.widget_id, str):
        if default_widget.widget_id.startswith("html-"):
            html_widget_ids.add(default_widget.widget_id)

    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            continue

        raw_content = item.get("content")
        if not isinstance(raw_content, str):
            continue

        name = _resolve_widget_name(item, widget_entries, default_widget, idx)
        display_name = name or "unknown"

        data_format = item.get("data_format", {})
        parse_as = data_format.get("parse_as")
        data_type = data_format.get("data_type")

        if parse_as == "text":
            mark_streamed_text()
            events.append(message_chunk(raw_content))
            continue

        widget_for_item = default_widget
        if not widget_for_item and widget_entries and idx < len(widget_entries):
            widget_for_item = widget_entries[idx][0]

        is_html_widget = bool(
            widget_for_item
            and isinstance(widget_for_item.widget_id, str)
            and widget_for_item.widget_id in html_widget_ids
        )
        if data_type == "html" or is_html_widget:
            html_content = _html_content_from_raw(raw_content) or raw_content
            artifacts.append(
                ClientArtifact(
                    type="html",
                    name=name or "HTML Content",
                    description=item.get("description") or "HTML artifact",
                    content=html_content,
                )
            )
            continue

        if data_type == "pdf":
            events.append(
                reasoning_step(
                    "PDF content received but could not be extracted. "
                    'Install PDF support: pip install "openbb-pydantic-ai[pdf]"',
                    event_type="WARNING",
                )
            )
            continue

        if data_type and data_type not in ("object", "json"):
            events.append(
                reasoning_step(
                    f"Format '{data_type}' not implemented", event_type="WARNING"
                )
            )
            continue

        parsed, error_event = _parse_item_content(raw_content, display_name)
        if error_event:
            events.append(error_event)
            continue

        new_artifacts, error_event = _item_to_artifact(
            parsed,
            name,
            item.get("description"),
        )

        if error_event:
            events.append(error_event)
            continue

        if new_artifacts:
            artifacts.extend(new_artifacts)
        else:
            mark_streamed_text()
            events.append(message_chunk(raw_content))

    return artifacts, events


def _extract_table_rows(value: Any) -> list[dict[str, Any]] | None:
    """Try to coerce nested JSON structures into table rows."""

    if isinstance(value, list) and value:
        if all(isinstance(row, dict) for row in value):
            return [dict(row) for row in value]

        # Handle double-encoded results: ["[{...}]"]
        nested: list[Any] = []
        all_strings = True
        for entry in value:
            if not isinstance(entry, str):
                all_strings = False
                break
            decoded = _decode_nested_json(parse_json(entry))
            nested.append(decoded)

        if all_strings:
            if nested and len(nested) == 1 and isinstance(nested[0], list):
                candidate = nested[0]
                if candidate and all(isinstance(row, dict) for row in candidate):
                    return [dict(row) for row in candidate]
            if nested and all(isinstance(row, dict) for row in nested):
                return [dict(row) for row in nested]

    if isinstance(value, str):
        parsed = _decode_nested_json(parse_json(value))
        if parsed == value:
            return None
        return _extract_table_rows(parsed)

    return None


def _decode_nested_json(value: Any, *, max_depth: int = 3) -> Any:
    """Recursively parse JSON strings to handle double-encoded payloads."""

    depth = 0
    while isinstance(value, str) and depth < max_depth:
        parsed = parse_json(value)
        if parsed == value:
            break
        value = parsed
        depth += 1
    return value


def _html_content_from_raw(value: str) -> str | None:
    """Extract HTML from raw string payloads, handling double-encoded JSON."""
    parsed = _decode_nested_json(parse_json(value))
    if isinstance(parsed, str):
        return parsed.replace("\\n", "\n")
    if isinstance(parsed, dict):
        candidate = parsed.get("content") or parsed.get("html")
        if isinstance(candidate, str):
            return candidate.replace("\\n", "\n")
    # Fallback: normalize literal \n sequences.
    if "\\n" in value:
        return value.replace("\\n", "\n")
    return None
