"""Event stream transformer for OpenBB Workspace SSE protocol."""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator, Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any

from openbb_ai.helpers import (
    citations,
    cite,
    get_widget_data,
    reasoning_step,
)
from openbb_ai.models import (
    SSE,
    AgentTool,
    Citation,
    FunctionCallSSE,
    FunctionCallSSEData,
    LlmClientFunctionCallResultMessage,
    MessageArtifactSSE,
    MessageChunkSSE,
    QueryRequest,
    StatusUpdateSSE,
    Widget,
    WidgetRequest,
)
from pydantic_ai import DeferredToolRequests
from pydantic_ai.messages import (
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    RetryPromptPart,
    TextPart,
    TextPartDelta,
    ThinkingPart,
    ThinkingPartDelta,
    ToolReturnPart,
)
from pydantic_ai.run import AgentRunResultEvent
from pydantic_ai.ui import UIEventStream

from openbb_pydantic_ai._config import (
    CHART_TOOL_NAME,
    EVENT_TYPE_ERROR,
    EVENT_TYPE_THINKING,
    EVENT_TYPE_WARNING,
    EXECUTE_MCP_TOOL_NAME,
    GET_WIDGET_DATA_TOOL_NAME,
    HTML_TOOL_NAME,
    PDF_QUERY_TOOL_NAME,
    TABLE_TOOL_NAME,
)
from openbb_pydantic_ai._dependencies import OpenBBDeps
from openbb_pydantic_ai._event_stream_components import StreamState
from openbb_pydantic_ai._event_stream_helpers import (
    ToolCallInfo,
    artifact_from_output,
    extract_widget_args,
    handle_generic_tool_result,
    serialized_content_from_result,
    tool_result_events_from_content,
)
from openbb_pydantic_ai._pdf_preprocess import preprocess_pdf_in_results
from openbb_pydantic_ai._stream_parser import StreamParser
from openbb_pydantic_ai._utils import format_args, normalize_args
from openbb_pydantic_ai._widget_registry import WidgetRegistry

logger = logging.getLogger(__name__)


def _encode_sse(event: SSE) -> str:
    payload = event.model_dump()
    return f"event: {payload['event']}\ndata: {payload['data']}\n\n"


@dataclass
class OpenBBAIEventStream(UIEventStream[QueryRequest, SSE, OpenBBDeps, Any]):
    """Transform native Pydantic AI events into OpenBB SSE events."""

    widget_registry: WidgetRegistry = field(default_factory=WidgetRegistry)
    """Registry for widget lookup and discovery."""
    pending_results: list[LlmClientFunctionCallResultMessage] = field(
        default_factory=list
    )
    mcp_tools: Mapping[str, AgentTool] | None = None

    # State management components
    _state: StreamState = field(init=False, default_factory=StreamState)
    _queued_viz_artifacts: list[MessageArtifactSSE] = field(
        init=False, default_factory=list
    )
    _stream_parser: StreamParser = field(init=False, default_factory=StreamParser)

    # Simple state flags
    _has_streamed_text: bool = field(init=False, default=False)
    _deferred_results_emitted: bool = field(init=False, default=False)
    _final_output: str | None = field(init=False, default=None)

    def encode_event(self, event: SSE) -> str:
        return _encode_sse(event)

    def _record_text_streamed(self) -> None:
        self._has_streamed_text = True

    async def before_stream(self) -> AsyncIterator[SSE]:
        """Emit tool results for any deferred results provided upfront."""
        if self._deferred_results_emitted:
            return

        self._deferred_results_emitted = True

        # Extract text from any PDF content before passing to the LLM
        self.pending_results = await preprocess_pdf_in_results(self.pending_results)

        # Process any pending deferred tool results from previous requests
        for result_message in self.pending_results:
            async for event in self._process_deferred_result(result_message):
                yield event

    async def _process_deferred_result(
        self, result_message: LlmClientFunctionCallResultMessage
    ) -> AsyncIterator[SSE]:
        """Process a single deferred result message and yield SSE events."""
        if result_message.function == EXECUTE_MCP_TOOL_NAME:
            async for event in self._process_mcp_result(result_message):
                yield event
            return

        widget_entries = self._widget_entries_from_result(result_message)
        content = serialized_content_from_result(result_message)

        for idx, (widget, widget_args) in enumerate(widget_entries, start=1):
            if widget is not None:
                citation_details = format_args(widget_args)
                citation = cite(
                    widget,
                    widget_args,
                    extra_details=citation_details if citation_details else None,
                )
                enriched = self._enrich_citation(widget, citation)
                self._state.add_citation(enriched)
                continue

            details = format_args(widget_args)
            suffix = f" #{idx}" if len(widget_entries) > 1 else ""
            message = (
                f"Received result{suffix} for '{result_message.function}' "
                "without widget metadata"
            )
            yield reasoning_step(
                message,
                details=details if details else None,
                event_type=EVENT_TYPE_WARNING,
            )

        # Extracted PDF text should appear in a reasoning dropdown, not in chat
        text_label = self._extracted_text_label(result_message, widget_entries)
        if text_label:
            yield reasoning_step(f"PDF — {text_label} returned")
            return

        primary_widget: Widget | None = None
        primary_args: dict[str, Any] = {}
        if widget_entries:
            primary_widget = widget_entries[0][0]
            primary_args = widget_entries[0][1]

        call_args = primary_args if len(widget_entries) == 1 else {}
        call_info = ToolCallInfo(
            tool_name=result_message.function,
            args=call_args,
            widget=primary_widget if len(widget_entries) == 1 else None,
        )

        for event in self._widget_result_events(
            call_info, content, widget_entries=widget_entries
        ):
            yield event

    @staticmethod
    def _extracted_text_label(
        result_message: LlmClientFunctionCallResultMessage,
        widget_entries: list[tuple[Widget | None, dict[str, Any]]],
    ) -> str | None:
        """Return a display label if all data items are PDF TOC content.

        Checks that every item has ``parse_as="text"`` and contains the
        ``pdf_query`` tool reference, which is injected by PDF preprocessing.
        """
        if not result_message.data:
            return None
        from openbb_ai.models import DataContent

        for entry in result_message.data:
            if not isinstance(entry, DataContent) or not entry.items:
                return None
            for item in entry.items:
                fmt = item.data_format
                if fmt is None:
                    return None
                if getattr(fmt, "parse_as", None) != "text":
                    return None
                if PDF_QUERY_TOOL_NAME not in item.content:
                    return None

        for widget, _args in widget_entries:
            if widget is not None and widget.name:
                return widget.name
        return result_message.function

    async def _process_mcp_result(
        self, result_message: LlmClientFunctionCallResultMessage
    ) -> AsyncIterator[SSE]:
        """Process a deferred MCP tool result."""
        tool_name = result_message.input_arguments.get("tool_name")
        parameters = result_message.input_arguments.get("parameters", {})
        args = normalize_args(parameters)
        agent_tool = (
            self._find_agent_tool(tool_name) if tool_name and self.mcp_tools else None
        )
        call_info = ToolCallInfo(
            tool_name=tool_name or EXECUTE_MCP_TOOL_NAME,
            args=args,
            agent_tool=agent_tool,
        )
        content = serialized_content_from_result(result_message)
        for sse in handle_generic_tool_result(
            call_info,
            content,
            mark_streamed_text=self._record_text_streamed,
        ):
            yield sse

    def _widget_entries_from_result(
        self, result: LlmClientFunctionCallResultMessage
    ) -> list[tuple[Widget | None, dict[str, Any]]]:
        if result.function == GET_WIDGET_DATA_TOOL_NAME:
            data_sources = result.input_arguments.get("data_sources", [])
            entries: list[tuple[Widget | None, dict[str, Any]]] = []

            if isinstance(data_sources, list):
                for source in data_sources:
                    if not isinstance(source, dict):
                        continue

                    widget: Widget | None = None
                    widget_uuid = source.get("widget_uuid")
                    if isinstance(widget_uuid, str):
                        widget = self.widget_registry.find_by_uuid(widget_uuid)

                    args = source.get("input_args")
                    args_dict = args if isinstance(args, dict) else {}
                    entries.append((widget, args_dict))

            if entries:
                return entries

        widget = self.widget_registry.find_for_result(result)
        widget_args = extract_widget_args(result)
        return [(widget, widget_args)]

    def _find_agent_tool(self, tool_name: str) -> AgentTool | None:
        if not self.mcp_tools:
            return None
        return self.mcp_tools.get(tool_name)

    def _build_mcp_function_call(
        self,
        *,
        tool_call_id: str,
        agent_tool: AgentTool,
        args: dict[str, Any],
    ) -> FunctionCallSSE:
        server_identifier = agent_tool.server_id or agent_tool.url
        input_arguments: dict[str, Any] = {
            "server_id": server_identifier,
            "tool_name": agent_tool.name,
            "parameters": args,
        }
        if agent_tool.endpoint:
            input_arguments["endpoint"] = agent_tool.endpoint
        if agent_tool.url:
            input_arguments.setdefault("url", agent_tool.url)

        extra_state = {
            "tool_calls": [
                {
                    "tool_call_id": tool_call_id,
                    "tool_name": agent_tool.name,
                    "server_id": server_identifier,
                }
            ]
        }

        return FunctionCallSSE(
            data=FunctionCallSSEData(
                function=EXECUTE_MCP_TOOL_NAME,
                input_arguments=input_arguments,
                extra_state=extra_state,
            )
        )

    async def on_error(self, error: Exception) -> AsyncIterator[SSE]:
        yield reasoning_step(str(error), event_type=EVENT_TYPE_ERROR)

    async def handle_text_start(
        self, part: TextPart, follows_text: bool = False
    ) -> AsyncIterator[SSE]:
        if part.content:
            for event in self._text_events_with_artifacts(part.content):
                yield event

    async def handle_text_delta(self, delta: TextPartDelta) -> AsyncIterator[SSE]:
        if delta.content_delta:
            for event in self._text_events_with_artifacts(delta.content_delta):
                yield event

    async def handle_thinking_start(
        self,
        part: ThinkingPart,
        follows_thinking: bool = False,
    ) -> AsyncIterator[SSE]:
        self._state.clear_thinking()
        if part.content:
            self._state.add_thinking(part.content)
        # yield needed to make this an async generator
        if False:
            yield

    async def handle_thinking_delta(
        self,
        delta: ThinkingPartDelta,
    ) -> AsyncIterator[SSE]:
        if delta.content_delta:
            self._state.add_thinking(delta.content_delta)
        # yield needed to make this an async generator
        if False:
            yield

    async def handle_thinking_end(
        self,
        part: ThinkingPart,
        followed_by_thinking: bool = False,
    ) -> AsyncIterator[SSE]:
        content = part.content or self._state.get_thinking()
        if not content and self._state.has_thinking():
            content = self._state.get_thinking()

        if content:
            details = {EVENT_TYPE_THINKING: content}
            yield reasoning_step(EVENT_TYPE_THINKING, details=details)

        self._state.clear_thinking()

    async def handle_run_result(
        self, event: AgentRunResultEvent[Any]
    ) -> AsyncIterator[SSE]:
        """Handle agent run result events, including deferred tool requests."""
        result = event.result
        output = getattr(result, "output", None)

        if isinstance(output, DeferredToolRequests):
            async for sse_event in self._handle_deferred_tool_requests(output):
                yield sse_event
            return

        artifact = self._artifact_from_output(output)
        if artifact is not None:
            yield artifact
            return

        if isinstance(output, str) and output and not self._has_streamed_text:
            self._final_output = output

    async def _handle_deferred_tool_requests(
        self, output: DeferredToolRequests
    ) -> AsyncIterator[SSE]:
        """Process deferred tool requests and yield widget request events."""
        widget_requests: list[WidgetRequest] = []
        tool_call_ids: list[dict[str, Any]] = []
        mcp_requests: list[tuple[str, AgentTool, dict[str, Any]]] = []

        for call in output.calls:
            widget = self.widget_registry.find_by_tool_name(call.tool_name)
            if widget is None:
                agent_tool = self._find_agent_tool(call.tool_name)
                if agent_tool is None:
                    continue

                args = normalize_args(call.args)
                self._state.register_tool_call(
                    tool_call_id=call.tool_call_id,
                    tool_name=call.tool_name,
                    args=args,
                    agent_tool=agent_tool,
                )

                mcp_requests.append((call.tool_call_id, agent_tool, args))
                continue

            args = normalize_args(call.args)
            widget_requests.append(WidgetRequest(widget=widget, input_arguments=args))
            self._state.register_tool_call(
                tool_call_id=call.tool_call_id,
                tool_name=call.tool_name,
                args=args,
                widget=widget,
            )
            tool_call_ids.append(
                {
                    "tool_call_id": call.tool_call_id,
                    "widget_uuid": str(widget.uuid),
                    "widget_id": widget.widget_id,
                }
            )

            # Create details dict with widget info and arguments for display
            details = {
                "Origin": widget.origin,
                "Widget Id": widget.widget_id,
                **format_args(args),
            }
            yield reasoning_step(
                f"Requesting widget '{widget.name}'",
                details=details,
            )

        if widget_requests:
            sse = get_widget_data(widget_requests)
            sse.data.extra_state = {"tool_calls": tool_call_ids}
            yield sse

        for tool_call_id, agent_tool, args in mcp_requests:
            yield self._build_mcp_function_call(
                tool_call_id=tool_call_id,
                agent_tool=agent_tool,
                args=args,
            )

    def _build_mcp_call_info(
        self, tool_name: str | None, args: dict[str, Any]
    ) -> ToolCallInfo:
        agent_tool = (
            self._find_agent_tool(tool_name) if tool_name and self.mcp_tools else None
        )
        return ToolCallInfo(
            tool_name=tool_name or EXECUTE_MCP_TOOL_NAME,
            args=args,
            agent_tool=agent_tool,
        )

    async def handle_function_tool_call(
        self, event: FunctionToolCallEvent
    ) -> AsyncIterator[SSE]:
        """Surface non-widget tool calls as reasoning steps."""

        part = event.part
        tool_name = part.tool_name

        is_widget_call = self.widget_registry.find_by_tool_name(tool_name)
        if is_widget_call or tool_name == GET_WIDGET_DATA_TOOL_NAME:
            return

        tool_call_id = part.tool_call_id
        if not tool_call_id or self._state.has_tool_call(tool_call_id):
            return

        args = normalize_args(part.args)
        self._state.register_tool_call(
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            args=args,
        )

        formatted_args = format_args(args)
        details = formatted_args if formatted_args else None
        yield reasoning_step(f"Calling tool '{tool_name}'", details=details)

    async def handle_function_tool_result(
        self, event: FunctionToolResultEvent
    ) -> AsyncIterator[SSE]:
        result_part = event.result

        if isinstance(result_part, RetryPromptPart):
            if result_part.content:
                content = result_part.content
                message = (
                    content
                    if isinstance(content, str)
                    else json.dumps(content, default=str)
                )
                yield reasoning_step(message, event_type=EVENT_TYPE_ERROR)
            return

        if not isinstance(result_part, ToolReturnPart):
            return

        tool_call_id = result_part.tool_call_id
        if not tool_call_id:
            return

        # Visualization tools (chart, table, html) - all use the same pattern
        viz_tools = {
            CHART_TOOL_NAME: "chart",
            TABLE_TOOL_NAME: "table",
            HTML_TOOL_NAME: "html",
        }

        if result_part.tool_name in viz_tools:
            key = viz_tools[result_part.tool_name]
            metadata = getattr(result_part, "metadata", {}) or {}
            viz_artifact = metadata.get(key)

            if isinstance(viz_artifact, MessageArtifactSSE):
                self._queued_viz_artifacts.append(viz_artifact)
                for sse_event in self._emit_placeholder_artifact():
                    yield sse_event

            content = result_part.content
            if isinstance(content, MessageArtifactSSE):
                self._queued_viz_artifacts.append(content)
                for sse_event in self._emit_placeholder_artifact():
                    yield sse_event
                return

            if isinstance(viz_artifact, MessageArtifactSSE):
                return

        self._collect_metadata_citations(getattr(result_part, "metadata", None))

        content = result_part.content
        if isinstance(content, MessageArtifactSSE):
            yield content
            return

        if isinstance(content, (MessageChunkSSE, StatusUpdateSSE)):
            yield content
            return

        call_info = self._state.get_tool_call(tool_call_id)
        if call_info is None:
            return

        if call_info.widget is not None:
            citation_details = format_args(call_info.args)
            # Collect citation for later emission (at the end)
            citation = cite(
                call_info.widget,
                call_info.args,
                extra_details=citation_details if citation_details else None,
            )
            enriched = self._enrich_citation(call_info.widget, citation)
            self._state.add_citation(enriched)

            widget_entries: list[tuple[Widget | None, dict[str, Any]]] = [
                (call_info.widget, call_info.args)
            ]
            for sse in self._widget_result_events(
                call_info, result_part.content, widget_entries=widget_entries
            ):
                yield sse
            return

        for sse in handle_generic_tool_result(
            call_info,
            result_part.content,
            mark_streamed_text=self._record_text_streamed,
        ):
            yield sse

    async def after_stream(self) -> AsyncIterator[SSE]:
        if self._state.has_thinking():
            content = self._state.get_thinking()
            if content:
                yield reasoning_step(content)
            self._state.clear_thinking()

        # Flush any remaining text in the parser buffer
        if self._has_streamed_text or self._final_output is None:
            for event in self._stream_parser.flush(self._record_text_streamed):
                yield event

        while self._queued_viz_artifacts:
            artifact = self._pop_next_viz_artifact()
            if artifact is not None:
                yield artifact

        # Emit all citations at the end
        if self._state.has_citations():
            yield citations(self._state.get_citations())
            self._state.clear_citations()

        if self._final_output and not self._has_streamed_text:
            events = self._text_events_with_artifacts(self._final_output)
            for event in events:
                yield event

    def _artifact_from_output(self, output: Any) -> SSE | None:
        """Create an artifact (chart or table) from agent output if possible."""
        return artifact_from_output(output)

    def _widget_result_events(
        self,
        call_info: ToolCallInfo,
        content: Any,
        widget_entries: list[tuple[Widget | None, dict[str, Any]]] | None = None,
    ) -> list[SSE]:
        """Emit SSE events for widget results with graceful fallbacks."""

        events = tool_result_events_from_content(
            content,
            mark_streamed_text=self._record_text_streamed,
            widget_entries=widget_entries,
        )
        if events:
            return events

        return handle_generic_tool_result(
            call_info,
            content,
            mark_streamed_text=self._record_text_streamed,
        )

    def _text_events_with_artifacts(self, text: str) -> list[SSE]:
        return self._stream_parser.parse(
            text,
            self._artifact_generator(),
            on_text_streamed=self._record_text_streamed,
        )

    def _artifact_generator(self) -> Iterator[MessageArtifactSSE]:
        while self._queued_viz_artifacts:
            yield self._queued_viz_artifacts.pop(0)

    def _pop_next_viz_artifact(self) -> MessageArtifactSSE | None:
        if self._queued_viz_artifacts:
            return self._queued_viz_artifacts.pop(0)
        return None

    def _emit_placeholder_artifact(self) -> list[SSE]:
        if not self._stream_parser.has_pending_placeholder():
            return []
        return self._text_events_with_artifacts("")

    def _collect_metadata_citations(self, metadata: Any) -> None:
        """Collect citation payloads attached to tool metadata."""
        for citation in self._citations_from_metadata(metadata):
            self._state.add_citation(citation)

    @staticmethod
    def _citations_from_metadata(metadata: Any) -> list[Citation]:
        """Parse citation objects from tool metadata payloads."""
        if not isinstance(metadata, Mapping):
            return []

        raw = metadata.get("citations")
        if raw is None:
            return []
        raw_items = raw if isinstance(raw, list) else [raw]

        citations_out: list[Citation] = []
        for item in raw_items:
            if isinstance(item, Citation):
                citations_out.append(item)
                continue
            if not isinstance(item, Mapping):
                continue
            try:
                citations_out.append(Citation.model_validate(item))
            except Exception:  # noqa: S112
                logger.debug("Failed to parse citation from metadata: %s", item)
                continue
        return citations_out

    @staticmethod
    def _enrich_citation(widget: Widget, citation: Citation) -> Citation:
        """Ensure widget metadata (uuid/name/description) is present in citations."""

        source = citation.source_info
        updates: dict[str, Any] = {}
        if source.uuid is None:
            updates["uuid"] = widget.uuid
        if source.name is None and widget.name:
            updates["name"] = widget.name
        if source.description is None and widget.description:
            updates["description"] = widget.description

        if not updates:
            return citation

        enriched_source = source.model_copy(update=updates)
        return citation.model_copy(update={"source_info": enriched_source})
