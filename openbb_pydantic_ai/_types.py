"""Type definitions and protocols for OpenBB Pydantic AI adapter."""

from __future__ import annotations

import sys
from typing import Any, Protocol, TypedDict

if sys.version_info >= (3, 11):
    from typing import NotRequired
else:
    from typing_extensions import NotRequired


class SerializedContent(TypedDict):
    """Structure for serialized tool result content."""

    input_arguments: dict[str, Any]
    data: list[Any]
    extra_state: NotRequired[dict[str, Any]]


class TextStreamCallback(Protocol):
    """Protocol for callbacks that mark text as having been streamed."""

    def __call__(self) -> None:
        """Mark that text has been streamed."""
        ...


# Type alias for structured detail entries
DetailEntry = dict[str, Any] | str
