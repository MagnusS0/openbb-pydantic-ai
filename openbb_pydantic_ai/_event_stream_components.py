"""State management components for OpenBB event stream."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from openbb_ai.models import AgentTool, Citation, Widget

from openbb_pydantic_ai._event_stream_helpers import ToolCallInfo
from openbb_pydantic_ai._local_tool_capsule import LocalToolEntry


@dataclass
class StreamState:
    """Manages state for the event stream."""

    thinking: list[str] = field(default_factory=list)
    citations: list[Citation] = field(default_factory=list)
    pending_tool_calls: dict[str, ToolCallInfo] = field(default_factory=dict)
    local_pending_calls: dict[str, LocalToolEntry] = field(default_factory=dict)
    local_completed_entries: list[LocalToolEntry] = field(default_factory=list)

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

    def drain_citations(self) -> list[Citation]:
        """Return and clear all collected citations."""
        drained, self.citations = self.citations, []
        return drained

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

    def register_local_tool_call(
        self,
        *,
        tool_call_id: str,
        tool_name: str,
        args: dict[str, Any],
    ) -> None:
        """Register a local (non-deferred) tool call for capsule capture."""
        self.local_pending_calls[tool_call_id] = LocalToolEntry(
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            args=args,
        )

    def complete_local_tool_call(self, tool_call_id: str, result: Any) -> None:
        """Mark a local tool call complete and store its result."""
        entry = self.local_pending_calls.pop(tool_call_id, None)
        if entry is None:
            return
        entry.result = result
        self.local_completed_entries.append(entry)

    def drain_unflushed_local_entries(self) -> list[LocalToolEntry]:
        """Return all completed entries not yet emitted into a capsule."""
        if not self.local_completed_entries:
            return []
        entries, self.local_completed_entries = self.local_completed_entries, []
        return entries
