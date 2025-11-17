"""Pydantic AI UI adapter for OpenBB Workspace."""

from importlib.metadata import PackageNotFoundError, version

from openbb_pydantic_ai._adapter import OpenBBAIAdapter
from openbb_pydantic_ai._config import GET_WIDGET_DATA_TOOL_NAME
from openbb_pydantic_ai._dependencies import OpenBBDeps, build_deps_from_request
from openbb_pydantic_ai._event_builder import EventBuilder
from openbb_pydantic_ai._event_stream import OpenBBAIEventStream
from openbb_pydantic_ai._exceptions import (
    InvalidToolCallError,
    OpenBBPydanticAIError,
    SerializationError,
    WidgetNotFoundError,
)
from openbb_pydantic_ai._message_transformer import MessageTransformer
from openbb_pydantic_ai._serializers import ContentSerializer
from openbb_pydantic_ai._toolsets import (
    WidgetToolset,
    build_widget_tool,
    build_widget_tool_name,
    build_widget_toolsets,
)
from openbb_pydantic_ai._types import SerializedContent, TextStreamCallback
from openbb_pydantic_ai._widget_registry import WidgetRegistry

try:
    __version__ = version("openbb-pydantic-ai")
except PackageNotFoundError:
    __version__ = "0.0.0"

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
    "EventBuilder",
    "ContentSerializer",
    "MessageTransformer",
    "WidgetRegistry",
    "OpenBBPydanticAIError",
    "WidgetNotFoundError",
    "InvalidToolCallError",
    "SerializationError",
    "SerializedContent",
    "TextStreamCallback",
]
