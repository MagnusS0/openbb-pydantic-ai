from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import KW_ONLY, dataclass, field
from functools import cached_property
from typing import TYPE_CHECKING, Any, cast

from openbb_ai.models import (
    SSE,
    LlmClientFunctionCallResultMessage,
    LlmMessage,
    QueryRequest,
    Undefined,
)
from pydantic_ai import DeferredToolResults
from pydantic_ai.agent import AbstractAgent
from pydantic_ai.agent.abstract import Instructions
from pydantic_ai.builtin_tools import AbstractBuiltinTool
from pydantic_ai.messages import (
    ModelMessage,
    SystemPromptPart,
)
from pydantic_ai.models import KnownModelName, Model
from pydantic_ai.output import OutputSpec
from pydantic_ai.settings import ModelSettings
from pydantic_ai.toolsets import AbstractToolset, CombinedToolset, FunctionToolset
from pydantic_ai.ui import OnCompleteFunc, UIAdapter
from pydantic_ai.usage import RunUsage, UsageLimits
from typing_extensions import override

from ._dependencies import OpenBBDeps, build_deps_from_request
from ._event_stream import OpenBBAIEventStream
from ._message_transformer import MessageTransformer
from ._serializers import ContentSerializer
from ._toolsets import build_widget_toolsets
from ._utils import hash_tool_call
from ._widget_registry import WidgetRegistry

if TYPE_CHECKING:
    from starlette.requests import Request
    from starlette.responses import Response


@dataclass
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

        widget_defaults = self._widget_default_lines()
        if widget_defaults:
            lines.append("<widget_defaults>")
            lines.append(
                "Preloaded widget values (reuse unless the user requests different data):"  # noqa: E501
            )
            lines.extend(widget_defaults)
            lines.append("</widget_defaults>")

        workspace_state = self.deps.workspace_state
        if workspace_state and workspace_state.current_dashboard_info:
            dashboard = workspace_state.current_dashboard_info
            lines.append(
                f"Active dashboard: {dashboard.name} (tab {dashboard.current_tab_id})"
            )

        if lines:
            builder.add(SystemPromptPart(content="\n".join(lines)))

    def _widget_default_lines(self) -> list[str]:
        """Build prompt lines summarizing current or default widget parameter values."""
        if not self.deps.widgets:
            return []

        widgets_iter = self.deps.iter_widgets()

        lines: list[str] = []
        for widget in widgets_iter:
            params = getattr(widget, "params", None)
            if not params:
                continue

            param_entries: list[str] = []
            for param in params:
                source: str | None = None
                value: Any | None = None

                if param.current_value is not None:
                    source = "current"
                    value = param.current_value
                elif param.default_value is not Undefined.UNDEFINED:
                    source = "default"
                    value = param.default_value

                if value is None:
                    continue

                formatted_value = self._format_widget_value(value)
                entry = f"{param.name}={formatted_value}"
                if source == "current":
                    entry += " (current)"
                elif source == "default":
                    entry += " (default)"
                param_entries.append(entry)

            if not param_entries:
                continue

            widget_name = widget.name or widget.widget_id
            lines.append(f"- {widget_name}: {', '.join(param_entries)}")

        return lines

    @staticmethod
    def _format_widget_value(value: Any) -> str:
        """Return a human-readable representation of a widget parameter value."""
        if isinstance(value, str):
            return value
        if isinstance(value, Mapping):
            return ", ".join(f"{key}={val}" for key, val in value.items())
        if isinstance(value, Sequence) and not isinstance(
            value, (str, bytes, bytearray)
        ):
            return ", ".join(str(item) for item in value)
        return str(value)

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

    @classmethod
    async def dispatch_request(
        cls,
        request: "Request",
        *,
        agent: AbstractAgent[OpenBBDeps, Any],
        message_history: Sequence[ModelMessage] | None = None,
        deferred_tool_results: DeferredToolResults | None = None,
        model: Model | KnownModelName | str | None = None,
        instructions: Instructions[OpenBBDeps] = None,
        deps: OpenBBDeps | None = None,
        output_type: OutputSpec[Any] | None = None,
        model_settings: ModelSettings | None = None,
        usage_limits: UsageLimits | None = None,
        usage: RunUsage | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[OpenBBDeps]] | None = None,
        builtin_tools: Sequence[AbstractBuiltinTool] | None = None,
        on_complete: OnCompleteFunc[SSE] | None = None,
    ) -> "Response":
        """Override UIAdapter.dispatch_request to allow optional deps."""
        deps_to_forward = cast(OpenBBDeps, deps)

        return await super().dispatch_request(
            request,
            agent=agent,
            message_history=message_history,
            deferred_tool_results=deferred_tool_results,
            model=model,
            instructions=instructions,
            deps=deps_to_forward,
            output_type=output_type,
            model_settings=model_settings,
            usage_limits=usage_limits,
            usage=usage,
            infer_name=infer_name,
            toolsets=toolsets,
            builtin_tools=builtin_tools,
            on_complete=on_complete,
        )

    @override
    def run_stream_native(
        self,
        *,
        output_type: OutputSpec[Any] | None = None,
        message_history: Sequence[ModelMessage] | None = None,
        deferred_tool_results: DeferredToolResults | None = None,
        model: Model | KnownModelName | str | None = None,
        instructions: Instructions[OpenBBDeps] = None,
        deps: OpenBBDeps | None = None,
        model_settings: ModelSettings | None = None,
        usage_limits: UsageLimits | None = None,
        usage: RunUsage | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[OpenBBDeps]] | None = None,
        builtin_tools: Sequence[AbstractBuiltinTool] | None = None,
    ):
        """
        Run the agent with OpenBB-specific defaults for
        deps, messages, and deferred results.
        """
        resolved_deps: OpenBBDeps = deps or self.deps
        deferred_tool_results = deferred_tool_results or self.deferred_tool_results
        message_history = message_history or self.messages

        return super().run_stream_native(
            output_type=output_type,
            message_history=message_history,
            deferred_tool_results=deferred_tool_results,
            model=model,
            instructions=instructions,
            deps=resolved_deps,
            model_settings=model_settings,
            usage_limits=usage_limits,
            usage=usage,
            infer_name=infer_name,
            toolsets=toolsets,
            builtin_tools=builtin_tools,
        )

    @override
    def run_stream(
        self,
        *,
        output_type: OutputSpec[Any] | None = None,
        message_history: Sequence[ModelMessage] | None = None,
        deferred_tool_results: DeferredToolResults | None = None,
        model: Model | KnownModelName | str | None = None,
        instructions: Instructions[OpenBBDeps] = None,
        deps: OpenBBDeps | None = None,
        model_settings: ModelSettings | None = None,
        usage_limits: UsageLimits | None = None,
        usage: RunUsage | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[OpenBBDeps]] | None = None,
        builtin_tools: Sequence[AbstractBuiltinTool] | None = None,
        on_complete: OnCompleteFunc[SSE] | None = None,
    ):
        """Run the agent and stream protocol-specific events with OpenBB defaults."""
        resolved_deps: OpenBBDeps = deps or self.deps
        deferred_tool_results = deferred_tool_results or self.deferred_tool_results
        message_history = message_history or self.messages

        return super().run_stream(
            output_type=output_type,
            message_history=message_history,
            deferred_tool_results=deferred_tool_results,
            model=model,
            instructions=instructions,
            deps=resolved_deps,
            model_settings=model_settings,
            usage_limits=usage_limits,
            usage=usage,
            infer_name=infer_name,
            toolsets=toolsets,
            builtin_tools=builtin_tools,
            on_complete=on_complete,
        )
