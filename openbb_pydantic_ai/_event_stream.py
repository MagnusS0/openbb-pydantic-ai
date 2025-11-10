"""Event stream transformer for OpenBB Workspace SSE protocol."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from openbb_ai.helpers import (
    citations,
    cite,
    get_widget_data,
    message_chunk,
    reasoning_step,
)
from openbb_ai.models import (
    SSE,
    LlmClientFunctionCallResultMessage,
    MessageArtifactSSE,
    MessageChunkSSE,
    QueryRequest,
    StatusUpdateSSE,
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

from ._config import GET_WIDGET_DATA_TOOL_NAME
from ._dependencies import OpenBBDeps
from ._event_stream_components import (
    CitationCollector,
    ThinkingBuffer,
    ToolCallTracker,
)
from ._event_stream_helpers import (
    ToolCallInfo,
    artifact_from_output,
    extract_widget_args,
    handle_generic_tool_result,
    serialized_content_from_result,
    tool_result_events_from_content,
)
from ._utils import format_args, normalize_args
from ._widget_registry import WidgetRegistry


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

    # State management components
    _tool_calls: ToolCallTracker = field(init=False, default_factory=ToolCallTracker)
    _citations: CitationCollector = field(init=False, default_factory=CitationCollector)
    _thinking: ThinkingBuffer = field(init=False, default_factory=ThinkingBuffer)

    # Simple state flags
    _has_streamed_text: bool = field(init=False, default=False)
    _final_output_pending: str | None = field(init=False, default=None)
    _deferred_results_emitted: bool = field(init=False, default=False)

    def encode_event(self, event: SSE) -> str:
        return _encode_sse(event)

    def _record_text_streamed(self) -> None:
        """Record that text content has been streamed to the client."""
        self._has_streamed_text = True

    async def before_stream(self) -> AsyncIterator[SSE]:
        """Emit tool results for any deferred results provided upfront."""
        if self._deferred_results_emitted:
            return

        self._deferred_results_emitted = True

        # Process any pending deferred tool results from previous requests
        for result_message in self.pending_results:
            async for event in self._process_deferred_result(result_message):
                yield event

    async def _process_deferred_result(
        self, result_message: LlmClientFunctionCallResultMessage
    ) -> AsyncIterator[SSE]:
        """Process a single deferred result message and yield SSE events."""
        widget = self.widget_registry.find_for_result(result_message)

        widget_args = extract_widget_args(result_message)
        content = serialized_content_from_result(result_message)
        call_info = ToolCallInfo(
            tool_name=result_message.function,
            args=widget_args,
            widget=widget,
        )

        if widget is not None:
            citation = cite(widget, widget_args)
            self._citations.add(citation)
        else:
            details = format_args(widget_args)
            yield reasoning_step(
                f"Received result for '{result_message.function}' "
                "without widget metadata",
                details=details if details else None,
                event_type="WARNING",
            )

        for event in self._widget_result_events(call_info, content):
            yield event

    async def on_error(self, error: Exception) -> AsyncIterator[SSE]:
        yield reasoning_step(str(error), event_type="ERROR")

    async def handle_text_start(
        self, part: TextPart, follows_text: bool = False
    ) -> AsyncIterator[SSE]:
        if part.content:
            self._record_text_streamed()
            yield message_chunk(part.content)

    async def handle_text_delta(self, delta: TextPartDelta) -> AsyncIterator[SSE]:
        if delta.content_delta:
            self._record_text_streamed()
            yield message_chunk(delta.content_delta)

    async def handle_thinking_start(
        self,
        part: ThinkingPart,
        follows_thinking: bool = False,
    ) -> AsyncIterator[SSE]:
        self._thinking.clear()
        if part.content:
            self._thinking.append(part.content)
        return
        yield  # pragma: no cover

    async def handle_thinking_delta(
        self,
        delta: ThinkingPartDelta,
    ) -> AsyncIterator[SSE]:
        if delta.content_delta:
            self._thinking.append(delta.content_delta)
        return
        yield  # pragma: no cover

    async def handle_thinking_end(
        self,
        part: ThinkingPart,
        followed_by_thinking: bool = False,
    ) -> AsyncIterator[SSE]:
        content = part.content or self._thinking.get_content()
        if not content and not self._thinking.is_empty():
            content = self._thinking.get_content()

        if content:
            details = {"Thinking": content}
            yield reasoning_step("Thinking", details=details)

        self._thinking.clear()

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
            self._final_output_pending = output

    async def _handle_deferred_tool_requests(
        self, output: DeferredToolRequests
    ) -> AsyncIterator[SSE]:
        """Process deferred tool requests and yield widget request events."""
        widget_requests: list[WidgetRequest] = []
        tool_call_ids: list[dict[str, Any]] = []

        for call in output.calls:
            widget = self.widget_registry.find_by_tool_name(call.tool_name)
            if widget is None:
                continue

            args = normalize_args(call.args)
            widget_requests.append(WidgetRequest(widget=widget, input_arguments=args))
            self._tool_calls.register_call(
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
        if not tool_call_id or self._tool_calls.has_pending(tool_call_id):
            return

        args = normalize_args(part.args)
        self._tool_calls.register_call(
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
                yield reasoning_step(message, event_type="ERROR")
            return

        if not isinstance(result_part, ToolReturnPart):
            return

        tool_call_id = result_part.tool_call_id
        if not tool_call_id:
            return

        if isinstance(
            result_part.content, (MessageArtifactSSE, MessageChunkSSE, StatusUpdateSSE)
        ):
            yield result_part.content
            return

        call_info = self._tool_calls.get_call_info(tool_call_id)
        if call_info is None:
            return

        if call_info.widget is not None:
            # Collect citation for later emission (at the end)
            citation = cite(call_info.widget, call_info.args)
            self._citations.add(citation)

            for sse in self._widget_result_events(call_info, result_part.content):
                yield sse
            return

        for sse in handle_generic_tool_result(
            call_info,
            result_part.content,
            mark_streamed_text=self._record_text_streamed,
        ):
            yield sse

    async def after_stream(self) -> AsyncIterator[SSE]:
        if not self._thinking.is_empty():
            content = self._thinking.get_content()
            if content:
                yield reasoning_step(content)
            self._thinking.clear()

        if self._final_output_pending and not self._has_streamed_text:
            yield message_chunk(self._final_output_pending)

        self._final_output_pending = None

        # Emit all citations at the end
        if self._citations.has_citations():
            yield citations(self._citations.get_all())
            self._citations.clear()

        return
        yield  # pragma: no cover

    def _artifact_from_output(self, output: Any) -> SSE | None:
        """Create an artifact (chart or table) from agent output if possible."""
        return artifact_from_output(output)

    def _widget_result_events(
        self,
        call_info: ToolCallInfo,
        content: Any,
    ) -> list[SSE]:
        """Emit SSE events for widget results with graceful fallbacks."""

        events = tool_result_events_from_content(
            content, mark_streamed_text=self._record_text_streamed
        )
        if events:
            return events

        return handle_generic_tool_result(
            call_info,
            content,
            mark_streamed_text=self._record_text_streamed,
        )
