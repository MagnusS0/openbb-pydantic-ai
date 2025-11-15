"""Centralized configuration for OpenBB Pydantic AI adapter."""

from __future__ import annotations

from typing import Any, Mapping

# Tool name constants
GET_WIDGET_DATA_TOOL_NAME = "get_widget_data"
EXECUTE_MCP_TOOL_NAME = "execute_agent_tool"
CHART_TOOL_NAME = "openbb_create_chart"
CHART_PLACEHOLDER_TOKEN = "{{openbb_chart}}"  # noqa: S105

# Field exclusion lists for citation and status update details
CITATION_EXCLUDED_FIELDS = frozenset(
    [
        "lastupdated",
        "source",
        "id",
        "uuid",
        "storedfileuuid",
        "datakey",
        "originalfilename",
        "extension",
        "category",
        "subcategory",
        "transcript_url",
    ]
)

STATUS_UPDATE_EXCLUDED_FIELDS = frozenset(
    [
        "lastupdated",
        "source",
        "id",
        "uuid",
        "storedfileuuid",
        "url",
        "datakey",
        "originalfilename",
        "extension",
        "category",
        "subcategory",
        "transcript_url",
    ]
)

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
