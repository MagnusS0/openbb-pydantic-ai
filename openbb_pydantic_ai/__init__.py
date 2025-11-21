"""Pydantic AI UI adapter for OpenBB Workspace."""

from importlib.metadata import PackageNotFoundError, version

from openbb_pydantic_ai._adapter import OpenBBAIAdapter
from openbb_pydantic_ai._dependencies import OpenBBDeps
from openbb_pydantic_ai._event_stream import OpenBBAIEventStream
from openbb_pydantic_ai._exceptions import OpenBBPydanticAIError
from openbb_pydantic_ai._toolsets import WidgetToolset

try:
    __version__ = version("openbb-pydantic-ai")
except PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = [
    "OpenBBAIAdapter",
    "OpenBBAIEventStream",
    "OpenBBDeps",
    "WidgetToolset",
    "OpenBBPydanticAIError",
]
