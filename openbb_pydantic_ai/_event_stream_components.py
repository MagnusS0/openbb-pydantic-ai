"""State management components for OpenBB event stream."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from openbb_ai.models import AgentTool, Citation, Widget

from openbb_pydantic_ai._event_stream_helpers import ToolCallInfo


@dataclass
class StreamState:
    """Manages state for the event stream."""

    thinking: list[str] = field(default_factory=list)
    citations: list[Citation] = field(default_factory=list)
    pending_tool_calls: dict[str, ToolCallInfo] = field(default_factory=dict)

    def add_thinking(self, content: str) -> None:
        """Add content to the thinking buffer."""
        self.thinking.append(content)

    def get_thinking(self) -> str:
        """Get accumulated thinking content."""
        return "".join(self.thinking)

    def clear_thinking(self) -> None:
        """Clear the thinking buffer."""
        self.thinking.clear()

    def has_thinking(self) -> bool:
        """Check if thinking buffer has content."""
        return bool(self.thinking)

    def add_citation(self, citation: Citation) -> None:
        """Add a citation to the collection."""
        self.citations.append(citation)

    def get_citations(self) -> list[Citation]:
        """Get all collected citations."""
        return self.citations.copy()

    def clear_citations(self) -> None:
        """Clear all citations."""
        self.citations.clear()

    def has_citations(self) -> bool:
        """Check if any citations have been collected."""
        return bool(self.citations)

    def register_tool_call(
        self,
        tool_call_id: str,
        tool_name: str,
        args: dict[str, Any],
        widget: Widget | None = None,
        *,
        agent_tool: AgentTool | None = None,
    ) -> None:
        """Register a pending tool call."""
        self.pending_tool_calls[tool_call_id] = ToolCallInfo(
            tool_name=tool_name,
            args=args,
            widget=widget,
            agent_tool=agent_tool,
        )

    def get_tool_call(self, tool_call_id: str) -> ToolCallInfo | None:
        """Retrieve and remove call info for a tool call ID."""
        return self.pending_tool_calls.pop(tool_call_id, None)

    def has_tool_call(self, tool_call_id: str) -> bool:
        """Check if a tool call ID is registered."""
        return tool_call_id in self.pending_tool_calls
