"""Pydantic AI UI adapter for OpenBB Workspace."""

from ._adapter import OpenBBAIAdapter
from ._dependencies import OpenBBDeps, build_deps_from_request
from ._event_stream import OpenBBAIEventStream
from ._toolsets import (
    WidgetToolset,
    build_widget_tool,
    build_widget_tool_name,
    build_widget_toolsets,
)
from ._utils import GET_WIDGET_DATA_TOOL_NAME

__all__ = [
    "OpenBBAIAdapter",
    "OpenBBAIEventStream",
    "OpenBBDeps",
    "build_deps_from_request",
    "WidgetToolset",
    "build_widget_tool",
    "build_widget_tool_name",
    "build_widget_toolsets",
    "GET_WIDGET_DATA_TOOL_NAME",
]
