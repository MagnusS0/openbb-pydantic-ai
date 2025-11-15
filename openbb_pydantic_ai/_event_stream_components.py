"""State management components for OpenBB event stream."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from openbb_ai.models import AgentTool, Citation, Widget

from ._event_stream_helpers import ToolCallInfo


@dataclass
class ThinkingBuffer:
    """Manages thinking content accumulation during streaming."""

    _buffer: list[str] = field(default_factory=list, init=False)

    def append(self, content: str) -> None:
        """Add content to the thinking buffer.

        Parameters
        ----------
        content : str
            Content to append
        """
        self._buffer.append(content)

    def get_content(self) -> str:
        """Get accumulated thinking content.

        Returns
        -------
        str
            Concatenated thinking content
        """
        return "".join(self._buffer)

    def clear(self) -> None:
        """Clear the thinking buffer."""
        self._buffer.clear()

    def is_empty(self) -> bool:
        """Check if buffer is empty.

        Returns
        -------
        bool
            True if buffer has no content
        """
        return len(self._buffer) == 0


@dataclass
class CitationCollector:
    """Tracks and manages citations during streaming."""

    _citations: list[Citation] = field(default_factory=list, init=False)

    def add(self, citation: Citation) -> None:
        """Add a citation to the collection.

        Parameters
        ----------
        citation : Citation
            Citation to add
        """
        self._citations.append(citation)

    def get_all(self) -> list[Citation]:
        """Get all collected citations.

        Returns
        -------
        list[Citation]
            List of all citations
        """
        return self._citations.copy()

    def clear(self) -> None:
        """Clear all citations."""
        self._citations.clear()

    def has_citations(self) -> bool:
        """Check if any citations have been collected.

        Returns
        -------
        bool
            True if there are citations
        """
        return len(self._citations) > 0


@dataclass
class ToolCallTracker:
    """Maps tool call IDs to their metadata and results."""

    _pending: dict[str, ToolCallInfo] = field(default_factory=dict, init=False)

    def register_call(
        self,
        tool_call_id: str,
        tool_name: str,
        args: dict[str, Any],
        widget: Widget | None = None,
        *,
        agent_tool: AgentTool | None = None,
    ) -> None:
        """Register a pending tool call.

        Parameters
        ----------
        tool_call_id : str
            Unique identifier for the tool call
        tool_name : str
            Name of the tool being called
        args : dict[str, Any]
            Arguments passed to the tool
        widget : Widget | None
            Associated widget if this is a widget tool call
        """
        self._pending[tool_call_id] = ToolCallInfo(
            tool_name=tool_name,
            args=args,
            widget=widget,
            agent_tool=agent_tool,
        )

    def get_call_info(self, tool_call_id: str) -> ToolCallInfo | None:
        """Retrieve and remove call info for a tool call ID.

        Parameters
        ----------
        tool_call_id : str
            Tool call ID to look up

        Returns
        -------
        ToolCallInfo | None
            Tool call metadata if found, None otherwise
        """
        return self._pending.pop(tool_call_id, None)

    def has_pending(self, tool_call_id: str) -> bool:
        """Check if a tool call ID is registered.

        Parameters
        ----------
        tool_call_id : str
            Tool call ID to check

        Returns
        -------
        bool
            True if the ID is registered
        """
        return tool_call_id in self._pending
