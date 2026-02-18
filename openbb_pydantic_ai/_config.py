"""Centralized configuration for OpenBB Pydantic AI adapter."""

from __future__ import annotations

from typing import Any, Mapping

# Tool name constants
GET_WIDGET_DATA_TOOL_NAME = "get_widget_data"
EXECUTE_MCP_TOOL_NAME = "execute_agent_tool"
CHART_TOOL_NAME = "openbb_create_chart"
TABLE_TOOL_NAME = "openbb_create_table"
HTML_TOOL_NAME = "openbb_create_html"
PDF_QUERY_TOOL_NAME = "pdf_query"
CHART_PLACEHOLDER_TOKEN = "{{place_chart_here}}"  # noqa: S105
CHART_PLACEHOLDER_TOKENS = (CHART_PLACEHOLDER_TOKEN,)

# Local tool history capsule constants
LOCAL_TOOL_CAPSULE_EXTRA_STATE_KEY = "openbb_local_tool_capsule"
LOCAL_TOOL_CAPSULE_REHYDRATED_KEY = "openbb_local_tool_rehydrated"
LOCAL_TOOL_CAPSULE_RESULT_KEY = "openbb_local_tool_result"


# Event type constants
EVENT_TYPE_THINKING = "Thinking"
EVENT_TYPE_ERROR = "ERROR"
EVENT_TYPE_WARNING = "WARNING"

# Widget parameter type to JSON schema mapping
PARAM_TYPE_SCHEMA_MAP: Mapping[str, dict[str, Any]] = {
    "string": {"type": "string"},
    "text": {"type": "string"},
    "number": {"type": "number"},
    "integer": {"type": "integer"},
    "boolean": {"type": "boolean"},
    "date": {"type": "string", "format": "date"},
    "ticker": {"type": "string"},
    "endpoint": {"type": "string"},
}

# Content formatting limits
MAX_ARG_DISPLAY_CHARS = 160
MAX_ARG_PREVIEW_ITEMS = 2
CONTENT_PREVIEW_MAX_CHARS = 120

# JSON/table parsing knobs
MAX_TABLE_PARSE_DEPTH = 5
MAX_NESTED_JSON_DECODE_DEPTH = 3
