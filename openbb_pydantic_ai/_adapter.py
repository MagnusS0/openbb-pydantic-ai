from __future__ import annotations

from collections.abc import Sequence
from dataclasses import KW_ONLY, dataclass, field
from functools import cached_property
from typing import Any, cast

from openbb_ai.models import (
    SSE,
    LlmClientFunctionCallResultMessage,
    LlmMessage,
    QueryRequest,
)
from pydantic_ai import DeferredToolResults
from pydantic_ai.messages import (
    ModelMessage,
    SystemPromptPart,
)
from pydantic_ai.toolsets import AbstractToolset, CombinedToolset, FunctionToolset
from pydantic_ai.ui import UIAdapter

from ._dependencies import OpenBBDeps, build_deps_from_request
from ._event_stream import OpenBBAIEventStream
from ._message_transformer import MessageTransformer
from ._serializers import ContentSerializer
from ._toolsets import build_widget_toolsets
from ._utils import hash_tool_call
from ._widget_registry import WidgetRegistry


@dataclass(slots=True)
class OpenBBAIAdapter(UIAdapter[QueryRequest, LlmMessage, SSE, OpenBBDeps, Any]):
    """UI adapter that bridges OpenBB Workspace requests with Pydantic AI."""

    _: KW_ONLY
    accept: str | None = None

    # Initialized in __post_init__
    _transformer: MessageTransformer = field(init=False)
    _registry: WidgetRegistry = field(init=False)
    _base_messages: list[LlmMessage] = field(init=False, default_factory=list)
    _pending_results: list[LlmClientFunctionCallResultMessage] = field(
        init=False, default_factory=list
    )

    def __post_init__(self) -> None:
        base, pending = self._split_messages(self.run_input.messages)
        self._base_messages = base
        self._pending_results = pending

        # Build tool call ID overrides for consistent IDs
        tool_call_id_overrides: dict[str, str] = {}
        for message in self._base_messages:
            if isinstance(message, LlmClientFunctionCallResultMessage):
                key = hash_tool_call(message.function, message.input_arguments)
                tool_call_id = self._tool_call_id_from_result(message)
                tool_call_id_overrides[key] = tool_call_id

        for message in self._pending_results:
            key = hash_tool_call(message.function, message.input_arguments)
            tool_call_id_overrides.setdefault(
                key,
                self._tool_call_id_from_result(message),
            )

        # Initialize transformer and registry
        self._transformer = MessageTransformer(tool_call_id_overrides)
        self._registry = WidgetRegistry(
            collection=self.run_input.widgets,
            toolsets=self._widget_toolsets,
        )

    @classmethod
    def build_run_input(cls, body: bytes) -> QueryRequest:
        return QueryRequest.model_validate_json(body)

    @classmethod
    def load_messages(cls, messages: Sequence[LlmMessage]) -> list[ModelMessage]:
        """Convert OpenBB messages to Pydantic AI messages.

        Note: This creates a transformer without overrides for standalone use.
        """
        transformer = MessageTransformer()
        return transformer.transform_batch(messages)

    @staticmethod
    def _split_messages(
        messages: Sequence[LlmMessage],
    ) -> tuple[list[LlmMessage], list[LlmClientFunctionCallResultMessage]]:
        """Split messages into base history and pending deferred results.

        Only results after the last AI message are considered pending. Results
        followed by AI messages were already processed in previous turns.

        Parameters
        ----------
        messages : Sequence[LlmMessage]
            Full message sequence

        Returns
        -------
        tuple[list[LlmMessage], list[LlmClientFunctionCallResultMessage]]
            (base messages, pending results that need processing)
        """
        base = list(messages)
        pending: list[LlmClientFunctionCallResultMessage] = []

        # Treat only the trailing tool results (those after the final assistant
        # message) as pending. Leave them in the base history so the next model
        # call still sees the complete tool call/result exchange.
        idx = len(base) - 1
        while idx >= 0:
            message = base[idx]
            if not isinstance(message, LlmClientFunctionCallResultMessage):
                break
            pending.insert(0, cast(LlmClientFunctionCallResultMessage, message))
            idx -= 1

        return base, pending

    def _tool_call_id_from_result(
        self, message: LlmClientFunctionCallResultMessage
    ) -> str:
        """Extract or generate a tool call ID from a result message."""
        extra_id = (
            message.extra_state.get("tool_call_id") if message.extra_state else None
        )
        if isinstance(extra_id, str):
            return extra_id
        return hash_tool_call(message.function, message.input_arguments)

    @cached_property
    def deps(self) -> OpenBBDeps:
        return build_deps_from_request(self.run_input)

    @cached_property
    def deferred_tool_results(self) -> DeferredToolResults | None:
        """Build deferred tool results from pending result messages."""
        if not self._pending_results:
            return None

        # When those trailing results already sit in the base history, skip
        # emitting DeferredToolResults; resending them would show up as a
        # conflicting duplicate tool response upstream.
        if self._pending_results_are_in_history():
            return None

        results = DeferredToolResults()
        for message in self._pending_results:
            actual_id = self._tool_call_id_from_result(message)
            serialized = ContentSerializer.serialize_result(message)
            results.calls[actual_id] = serialized
        return results

    def _pending_results_are_in_history(self) -> bool:
        if not self._pending_results:
            return False
        pending_len = len(self._pending_results)
        if pending_len > len(self._base_messages):
            return False
        tail = self._base_messages[-pending_len:]
        return all(
            orig is pending
            for orig, pending in zip(tail, self._pending_results, strict=True)
        )

    @cached_property
    def _widget_toolsets(self) -> tuple[FunctionToolset[OpenBBDeps], ...]:
        return build_widget_toolsets(self.run_input.widgets)

    def build_event_stream(self) -> OpenBBAIEventStream:
        return OpenBBAIEventStream(
            run_input=self.run_input,
            widget_registry=self._registry,
            pending_results=self._pending_results,
        )

    @cached_property
    def messages(self) -> list[ModelMessage]:
        """Build message history with context prompts."""
        from pydantic_ai.ui import MessagesBuilder

        builder = MessagesBuilder()
        self._add_context_prompts(builder)

        # Use transformer to convert messages with ID overrides
        transformed = self._transformer.transform_batch(self._base_messages)
        for msg in transformed:
            for part in msg.parts:
                builder.add(part)

        return builder.messages

    def _add_context_prompts(self, builder) -> None:
        """Add system prompts with workspace context, URLs, and dashboard info."""
        lines: list[str] = []

        if self.deps.context:
            lines.append("<workspace_context>")
            for ctx in self.deps.context:
                row_count = len(ctx.data.items) if ctx.data and ctx.data.items else 0
                summary = f"- {ctx.name} ({row_count} rows): {ctx.description}"
                lines.append(summary)
            lines.append("</workspace_context>")

        if self.deps.urls:
            lines.append("<relevant_urls>")
            joined = ", ".join(self.deps.urls)
            lines.append(f"Relevant URLs: {joined}")
            lines.append("</relevant_urls>")

        workspace_state = self.deps.workspace_state
        if workspace_state and workspace_state.current_dashboard_info:
            dashboard = workspace_state.current_dashboard_info
            lines.append(
                f"Active dashboard: {dashboard.name} (tab {dashboard.current_tab_id})"
            )

        if lines:
            builder.add(SystemPromptPart(content="\n".join(lines)))

    @cached_property
    def toolset(self) -> AbstractToolset[OpenBBDeps] | None:
        """Build combined toolset from widget toolsets."""
        if not self._widget_toolsets:
            return None
        if len(self._widget_toolsets) == 1:
            return self._widget_toolsets[0]
        combined = CombinedToolset(self._widget_toolsets)
        return cast(AbstractToolset[OpenBBDeps], combined)

    @cached_property
    def state(self) -> dict[str, Any] | None:
        """Extract workspace state as a dictionary."""
        if self.run_input.workspace_state is None:
            return None
        return self.run_input.workspace_state.model_dump(exclude_none=True)

    def run_stream_native(
        self,
        *,
        output_type=None,
        message_history=None,
        deferred_tool_results=None,
        model=None,
        instructions=None,
        deps=None,
        model_settings=None,
        usage_limits=None,
        usage=None,
        infer_name=True,
        toolsets=None,
        builtin_tools=None,
    ):
        """
        Run the agent with OpenBB-specific defaults for
        deps, messages, and deferred results.
        """
        deps = deps or self.deps
        deferred_tool_results = deferred_tool_results or self.deferred_tool_results
        message_history = message_history or self.messages

        return OpenBBAIAdapter.run_stream_native(
            self,
            output_type=output_type,
            message_history=message_history,
            deferred_tool_results=deferred_tool_results,
            model=model,
            instructions=instructions,
            deps=deps,
            model_settings=model_settings,
            usage_limits=usage_limits,
            usage=usage,
            infer_name=infer_name,
            toolsets=toolsets,
            builtin_tools=builtin_tools,
        )

    def run_stream(
        self,
        *,
        output_type=None,
        message_history=None,
        deferred_tool_results=None,
        model=None,
        instructions=None,
        deps=None,
        model_settings=None,
        usage_limits=None,
        usage=None,
        infer_name=True,
        toolsets=None,
        builtin_tools=None,
        on_complete=None,
    ):
        """Run the agent and stream protocol-specific events with OpenBB defaults."""
        deps = deps or self.deps  # type: ignore[assignment]
        deferred_tool_results = deferred_tool_results or self.deferred_tool_results
        message_history = message_history or self.messages

        return OpenBBAIAdapter.run_stream(
            self,
            output_type=output_type,
            message_history=message_history,
            deferred_tool_results=deferred_tool_results,
            model=model,
            instructions=instructions,
            deps=deps,
            model_settings=model_settings,
            usage_limits=usage_limits,
            usage=usage,
            infer_name=infer_name,
            toolsets=toolsets,
            builtin_tools=builtin_tools,
            on_complete=on_complete,
        )
