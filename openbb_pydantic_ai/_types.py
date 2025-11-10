"""Type definitions and protocols for OpenBB Pydantic AI adapter."""

from __future__ import annotations

from typing import Any, NotRequired, Protocol, TypedDict


class SerializedContent(TypedDict):
    """Structure for serialized tool result content."""

    input_arguments: dict[str, Any]
    data: list[Any]
    extra_state: NotRequired[dict[str, Any]]


class ToolCallMetadata(TypedDict):
    """Metadata for tracking tool calls in flight."""

    tool_call_id: str
    widget_uuid: str
    widget_id: str


class TextStreamCallback(Protocol):
    """Protocol for callbacks that mark text as having been streamed."""

    def __call__(self) -> None:
        """Mark that text has been streamed."""
        ...


# Type alias for structured detail entries
DetailEntry = dict[str, Any] | str
