"""Event stream transformer for OpenBB Workspace SSE protocol."""

from __future__ import annotations

import logging
from collections import deque
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
    DataContent,
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
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.run import AgentRunResultEvent
from pydantic_ai.ui import UIEventStream
from pydantic_core import to_json as _pydantic_to_json

from openbb_pydantic_ai._config import (
    CHART_TOOL_NAME,
    EVENT_TYPE_ERROR,
    EVENT_TYPE_THINKING,
    EVENT_TYPE_WARNING,
    EXECUTE_MCP_TOOL_NAME,
    GET_WIDGET_DATA_TOOL_NAME,
    HTML_TOOL_NAME,
    LOCAL_TOOL_CAPSULE_EXTRA_STATE_KEY,
    PDF_QUERY_TOOL_NAME,
    TABLE_TOOL_NAME,
)
from openbb_pydantic_ai._dependencies import OpenBBDeps
from openbb_pydantic_ai._event_stream_components import StreamState
from openbb_pydantic_ai._event_stream_formatters import _format_meta_tool_call_args
from openbb_pydantic_ai._event_stream_helpers import (
    ToolCallInfo,
    artifact_from_output,
    extract_widget_args,
    handle_generic_tool_result,
    tool_result_events_from_content,
)
from openbb_pydantic_ai._local_tool_capsule import pack_tool_history
from openbb_pydantic_ai._pdf_preprocess import preprocess_pdf_in_results
from openbb_pydantic_ai._serializers import serialize_result
from openbb_pydantic_ai._stream_parser import StreamParser
from openbb_pydantic_ai._utils import format_args, normalize_args
from openbb_pydantic_ai._widget_registry import WidgetRegistry

logger = logging.getLogger(__name__)

_MAX_WIDGET_ARG_UNWRAP_DEPTH = 3


def _encode_sse(event: SSE) -> str:
    data = event.data.model_dump_json(exclude_none=True)
    return f"event: {event.event}\ndata: {data}\n\n"


@dataclass
class OpenBBAIEventStream(UIEventStream[QueryRequest, SSE, OpenBBDeps, Any]):
    """Transform native Pydantic AI events into OpenBB SSE events."""

    widget_registry: WidgetRegistry = field(default_factory=WidgetRegistry)
    """Registry for widget lookup and discovery."""
    pending_results: list[LlmClientFunctionCallResultMessage] = field(
        default_factory=list
    )
    mcp_tools: Mapping[str, AgentTool] | None = None
    enable_local_tool_history_capsule: bool = True

    # State management components
    _state: StreamState = field(init=False, default_factory=StreamState)
    _queued_viz_artifacts: deque[MessageArtifactSSE] = field(
        init=False, default_factory=deque
    )
    _stream_parser: StreamParser = field(init=False, default_factory=StreamParser)

    # Simple state flags
    _deferred_results_emitted: bool = field(init=False, default=False)
    _final_output: str | None = field(init=False, default=None)

    def encode_event(self, event: SSE) -> str:
        return _encode_sse(event)

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
        content = serialize_result(result_message)

        for idx, (widget, widget_args) in enumerate(widget_entries, start=1):
            if widget is not None:
                citation_details = format_args(widget_args)
                citation = cite(
                    widget,
                    widget_args,
                    extra_details=citation_details or None,
                )
                enriched = self._enrich_citation(widget, citation)
                self._state.add_citation(enriched)
                continue

            details = format_args(widget_args)
            suffix = f" #{idx}" if len(widget_entries) > 1 else ""
            yield reasoning_step(
                f"Received result{suffix} for '{result_message.function}' "
                "without widget metadata",
                details=details or None,
                event_type=EVENT_TYPE_WARNING,
            )

        # Extracted PDF text should appear in a reasoning dropdown, not in chat
        text_label = self._extracted_text_label(result_message, widget_entries)
        if text_label:
            yield reasoning_step(f"PDF — {text_label} returned")
            return

        is_single = len(widget_entries) == 1
        call_info = ToolCallInfo(
            tool_name=result_message.function,
            args=widget_entries[0][1] if is_single else {},
            widget=widget_entries[0][0] if is_single else None,
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
        content = serialize_result(result_message)
        for sse in handle_generic_tool_result(
            call_info,
            content,
            mark_streamed_text=self._state.record_text_streamed,
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
        capsule_payload: str | None = None,
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
            ],
            "persist_tool_result": True,
        }
        if capsule_payload is not None:
            extra_state[LOCAL_TOOL_CAPSULE_EXTRA_STATE_KEY] = capsule_payload

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
        if False:  # pragma: no cover
            yield reasoning_step("")

    async def handle_thinking_delta(
        self,
        delta: ThinkingPartDelta,
    ) -> AsyncIterator[SSE]:
        if delta.content_delta:
            self._state.add_thinking(delta.content_delta)
        if False:  # pragma: no cover
            yield reasoning_step("")

    async def handle_thinking_end(
        self,
        part: ThinkingPart,
        followed_by_thinking: bool = False,
    ) -> AsyncIterator[SSE]:
        content = part.content or self._state.get_thinking()

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

        artifact = artifact_from_output(output)
        if artifact is not None:
            yield artifact
            return

        if isinstance(output, str) and output and not self._state.has_streamed_text:
            self._final_output = output

    @staticmethod
    def _expand_deferred_calls(
        output: DeferredToolRequests,
    ) -> list[ToolCallPart]:
        """Expand ``call_tools`` wrapper entries into individual calls.

        When progressive discovery is active, the graph defers the
        ``call_tools`` meta-tool itself.  The actual nested tool calls
        are stored in ``DeferredToolRequests.metadata`` under the
        ``deferred_calls`` key.  This helper expands those entries into
        individual ``ToolCallPart`` instances so downstream handling
        works identically to the non-progressive path.
        """
        expanded: list[ToolCallPart] = []
        for call in output.calls:
            if call.tool_name != "call_tools":
                expanded.append(call)
                continue

            # Check metadata first (populated by CallDeferred).
            meta = (output.metadata or {}).get(call.tool_call_id)
            nested: list[dict[str, Any]] | None = None
            if isinstance(meta, dict):
                nested = meta.get("deferred_calls")

            # Fallback: extract from the call args directly.
            if not nested:
                raw_args = normalize_args(call.args)
                raw_calls = raw_args.get("calls")
                if isinstance(raw_calls, dict):
                    nested = [raw_calls]
                elif isinstance(raw_calls, list):
                    nested = raw_calls

            if not nested:
                expanded.append(call)
                continue

            for idx, entry in enumerate(nested):
                if not isinstance(entry, dict):
                    continue
                tool_name = entry.get("tool_name")
                if not isinstance(tool_name, str) or not tool_name:
                    continue
                # Derive a stable tool_call_id from the parent call plus index
                derived_id = f"{call.tool_call_id}-{idx}"
                expanded.append(
                    ToolCallPart(
                        tool_name=tool_name,
                        args=entry.get("arguments") or {},
                        tool_call_id=derived_id,
                    )
                )

        return expanded

    async def _handle_deferred_tool_requests(
        self, output: DeferredToolRequests
    ) -> AsyncIterator[SSE]:
        """Process deferred tool requests and yield widget request events."""
        widget_requests: list[WidgetRequest] = []
        tool_call_ids: list[dict[str, Any]] = []
        mcp_requests: list[tuple[str, AgentTool, dict[str, Any]]] = []
        capsule_payload = self._build_local_tool_capsule_payload()
        capsule_attached = False

        expanded_calls = self._expand_deferred_calls(output)

        for call in expanded_calls:
            raw_args = normalize_args(call.args)
            effective_tool_name, effective_args = self._extract_effective_tool_call(
                call.tool_name, raw_args
            )

            widget = self.widget_registry.find_by_tool_name(effective_tool_name)
            if widget is None:
                agent_tool = self._find_agent_tool(effective_tool_name)
                if agent_tool is None:
                    continue

                self._state.register_tool_call(
                    tool_call_id=call.tool_call_id,
                    tool_name=effective_tool_name,
                    args=effective_args,
                    agent_tool=agent_tool,
                )

                mcp_requests.append((call.tool_call_id, agent_tool, effective_args))
                continue

            widget_requests.append(
                WidgetRequest(widget=widget, input_arguments=effective_args)
            )
            self._state.register_tool_call(
                tool_call_id=call.tool_call_id,
                tool_name=effective_tool_name,
                args=effective_args,
                widget=widget,
            )
            tool_call_ids.append(
                {
                    "tool_call_id": call.tool_call_id,
                    "widget_uuid": str(widget.uuid),
                    "widget_id": widget.widget_id,
                    "tool_name": effective_tool_name,
                }
            )

            # Create details dict with widget info and arguments for display
            details = {
                "Origin": widget.origin,
                "Widget Id": widget.widget_id,
                **format_args(effective_args),
            }
            yield reasoning_step(
                f"Requesting widget '{widget.name}'",
                details=details,
            )

        if widget_requests:
            sse = get_widget_data(widget_requests)
            extra_state: dict[str, Any] = {
                "tool_calls": tool_call_ids,
                "persist_tool_result": True,
            }
            if capsule_payload is not None:
                extra_state[LOCAL_TOOL_CAPSULE_EXTRA_STATE_KEY] = capsule_payload
                capsule_attached = True
            sse.data.extra_state = extra_state
            yield sse

        for tool_call_id, agent_tool, args in mcp_requests:
            yield self._build_mcp_function_call(
                tool_call_id=tool_call_id,
                agent_tool=agent_tool,
                args=args,
                capsule_payload=capsule_payload if not capsule_attached else None,
            )
            capsule_attached = True

    @staticmethod
    def _extract_effective_tool_call(
        tool_name: str, args: dict[str, Any]
    ) -> tuple[str, dict[str, Any]]:
        """Map meta `call_tools(...)` invocations to the nested target tool."""
        if tool_name != "call_tools":
            return tool_name, OpenBBAIEventStream._normalize_tool_args(tool_name, args)

        calls = args.get("calls")
        if not isinstance(calls, list) or len(calls) != 1:
            return tool_name, args

        entry = calls[0]
        if not isinstance(entry, dict):
            return tool_name, args

        nested_name = entry.get("tool_name")
        if not isinstance(nested_name, str) or not nested_name:
            return tool_name, args

        nested_args = entry.get("arguments", {})
        if not isinstance(nested_args, dict):
            nested_args = {}

        return nested_name, OpenBBAIEventStream._normalize_tool_args(
            nested_name, normalize_args(nested_args)
        )

    @staticmethod
    def _normalize_tool_args(tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
        """Normalize widget and MCP transport envelopes for nested tool calls."""
        normalized = normalize_args(args)

        if tool_name.startswith("openbb_widget_"):
            current = normalized
            for _ in range(_MAX_WIDGET_ARG_UNWRAP_DEPTH):
                data_sources = current.get("data_sources")
                if not isinstance(data_sources, list) or len(data_sources) != 1:
                    break
                first = data_sources[0]
                if not isinstance(first, dict):
                    break
                inner = first.get("input_args")
                if not isinstance(inner, dict):
                    break
                current = inner
            normalized = normalize_args(current)

        nested = normalized.get("parameters")
        if not isinstance(nested, dict):
            return normalized

        nested_tool_name = normalized.get("tool_name")
        if not isinstance(nested_tool_name, str) or nested_tool_name != tool_name:
            return normalized

        if not any(key in normalized for key in ("server_id", "url", "endpoint")):
            return normalized

        return normalize_args(nested)

    async def handle_function_tool_call(
        self, event: FunctionToolCallEvent
    ) -> AsyncIterator[SSE]:
        """Surface non-widget tool calls as reasoning steps."""

        part = event.part
        tool_name = part.tool_name

        raw_args = normalize_args(part.args)
        effective_tool_name, effective_args = self._extract_effective_tool_call(
            tool_name, raw_args
        )

        is_widget_call = self.widget_registry.find_by_tool_name(effective_tool_name)
        if is_widget_call or effective_tool_name == GET_WIDGET_DATA_TOOL_NAME:
            return

        tool_call_id = part.tool_call_id
        if not tool_call_id or self._state.has_tool_call(tool_call_id):
            return

        self._state.register_tool_call(
            tool_call_id=tool_call_id,
            tool_name=effective_tool_name,
            args=effective_args,
        )
        self._state.register_local_tool_call(
            tool_call_id=tool_call_id,
            tool_name=effective_tool_name,
            args=effective_args,
        )

        details: dict[str, Any] | None = (
            _format_meta_tool_call_args(effective_tool_name, effective_args)
            or format_args(effective_args)
            or None
        )
        yield reasoning_step(f"Calling tool '{effective_tool_name}'", details=details)

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
                    else _pydantic_to_json(content, serialize_unknown=True).decode()
                )
                yield reasoning_step(message, event_type=EVENT_TYPE_ERROR)
            return

        if not isinstance(result_part, ToolReturnPart):
            return

        tool_call_id = result_part.tool_call_id
        if not tool_call_id:
            return

        call_info = self._state.get_tool_call(tool_call_id)
        effective_tool_name = (
            call_info.tool_name if call_info else result_part.tool_name
        )

        # Visualization tools (chart, table, html) - all use the same pattern
        viz_tools = {
            CHART_TOOL_NAME: "chart",
            TABLE_TOOL_NAME: "table",
            HTML_TOOL_NAME: "html",
        }

        if effective_tool_name in viz_tools:
            key = viz_tools[effective_tool_name]
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

        if isinstance(content, DeferredToolRequests):
            async for sse in self._handle_deferred_tool_requests(content):
                yield sse
            return

        if call_info is None:
            return

        self._state.complete_local_tool_call(tool_call_id, result_part.content)

        if call_info.widget is not None:
            citation_details = format_args(call_info.args)
            # Collect citation for later emission (at the end)
            citation = cite(
                call_info.widget,
                call_info.args,
                extra_details=citation_details or None,
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
            mark_streamed_text=self._state.record_text_streamed,
        ):
            yield sse

    def _build_local_tool_capsule_payload(self) -> str | None:
        """Build a capsule payload from completed local tool entries."""
        if not self.enable_local_tool_history_capsule:
            return None

        entries = self._state.drain_unflushed_local_entries()
        if not entries:
            return None

        return pack_tool_history(entries)

    async def after_stream(self) -> AsyncIterator[SSE]:
        if self._state.has_thinking():
            content = self._state.get_thinking()
            if content:
                yield reasoning_step(content)
            self._state.clear_thinking()

        # Flush any remaining text in the parser buffer
        if self._state.has_streamed_text or self._final_output is None:
            for event in self._stream_parser.flush(self._state.record_text_streamed):
                yield event

        while self._queued_viz_artifacts:
            yield self._queued_viz_artifacts.popleft()

        # Emit all citations at the end
        drained_citations = self._state.drain_citations()
        if drained_citations:
            yield citations(drained_citations)

        if self._final_output and not self._state.has_streamed_text:
            for event in self._text_events_with_artifacts(self._final_output):
                yield event

    def _widget_result_events(
        self,
        call_info: ToolCallInfo,
        content: Any,
        widget_entries: list[tuple[Widget | None, dict[str, Any]]] | None = None,
    ) -> list[SSE]:
        """Emit SSE events for widget results with graceful fallbacks."""

        events = tool_result_events_from_content(
            content,
            mark_streamed_text=self._state.record_text_streamed,
            widget_entries=widget_entries,
        )
        if events:
            return events

        return handle_generic_tool_result(
            call_info,
            content,
            mark_streamed_text=self._state.record_text_streamed,
            content_events=events,
        )

    def _text_events_with_artifacts(self, text: str) -> list[SSE]:
        return self._stream_parser.parse(
            text,
            self._artifact_generator(),
            on_text_streamed=self._state.record_text_streamed,
        )

    def _artifact_generator(self) -> Iterator[MessageArtifactSSE]:
        while self._queued_viz_artifacts:
            yield self._queued_viz_artifacts.popleft()

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
