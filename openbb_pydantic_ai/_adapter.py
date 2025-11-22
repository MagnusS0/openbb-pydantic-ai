from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import KW_ONLY, dataclass, field
from datetime import datetime
from functools import cached_property
from typing import TYPE_CHECKING, Any, cast
from zoneinfo import ZoneInfo

from openbb_ai.models import (
    SSE,
    AgentTool,
    LlmClientFunctionCallResultMessage,
    LlmMessage,
    QueryRequest,
    Undefined,
    Widget,
)
from pydantic_ai import DeferredToolResults
from pydantic_ai.agent import AbstractAgent
from pydantic_ai.agent.abstract import Instructions
from pydantic_ai.builtin_tools import AbstractBuiltinTool
from pydantic_ai.messages import (
    ModelMessage,
)
from pydantic_ai.models import KnownModelName, Model
from pydantic_ai.output import OutputSpec
from pydantic_ai.settings import ModelSettings
from pydantic_ai.toolsets import AbstractToolset, CombinedToolset, FunctionToolset
from pydantic_ai.ui import OnCompleteFunc, UIAdapter
from pydantic_ai.usage import RunUsage, UsageLimits
from typing_extensions import override

from openbb_pydantic_ai._dependencies import OpenBBDeps, build_deps_from_request
from openbb_pydantic_ai._event_stream import OpenBBAIEventStream
from openbb_pydantic_ai._mcp_toolsets import build_mcp_toolsets
from openbb_pydantic_ai._message_transformer import MessageTransformer
from openbb_pydantic_ai._serializers import ContentSerializer
from openbb_pydantic_ai._toolsets import build_widget_toolsets
from openbb_pydantic_ai._utils import hash_tool_call
from openbb_pydantic_ai._widget_registry import WidgetRegistry

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
    _base_messages: list[LlmMessage] = field(init=False)
    _pending_results: list[LlmClientFunctionCallResultMessage] = field(init=False)

    def __post_init__(self) -> None:
        (
            self._base_messages,
            self._pending_results,
        ) = self._split_messages(self.run_input.messages)

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
        """Parse the raw request body into a ``QueryRequest`` instance."""

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
        for message in reversed(base):
            if not isinstance(message, LlmClientFunctionCallResultMessage):
                break
            pending.insert(0, message)

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
            isinstance(orig, LlmClientFunctionCallResultMessage)
            and orig.function == pending.function
            and orig.input_arguments == pending.input_arguments
            for orig, pending in zip(tail, self._pending_results, strict=True)
        )

    @cached_property
    def _widget_toolsets(self) -> tuple[FunctionToolset[OpenBBDeps], ...]:
        return build_widget_toolsets(self.run_input.widgets)

    @cached_property
    def _mcp_toolsets(self) -> tuple[AbstractToolset[Any], ...]:
        return build_mcp_toolsets(
            self.run_input.tools,
            self.run_input.workspace_options,
        )

    @cached_property
    def _toolsets(self) -> tuple[AbstractToolset[OpenBBDeps], ...]:
        toolsets: list[AbstractToolset[OpenBBDeps]] = list(self._widget_toolsets)
        if self._mcp_toolsets:
            toolsets.extend(
                cast("Sequence[AbstractToolset[OpenBBDeps]]", self._mcp_toolsets)
            )
        return tuple(toolsets)

    @cached_property
    def _mcp_tool_lookup(self) -> dict[str, AgentTool]:
        tools: dict[str, AgentTool] = {}
        for toolset in self._mcp_toolsets:
            mapping = getattr(toolset, "tools_by_name", None)
            if mapping:
                tools.update(mapping)
        return tools

    def build_event_stream(self) -> OpenBBAIEventStream:
        """Create the event stream wrapper for this adapter run."""

        return OpenBBAIEventStream(
            run_input=self.run_input,
            widget_registry=self._registry,
            pending_results=self._pending_results,
            mcp_tools=self._mcp_tool_lookup or None,
        )

    @cached_property
    def messages(self) -> list[ModelMessage]:
        """Build message history with context prompts."""
        from pydantic_ai.ui import MessagesBuilder

        builder = MessagesBuilder()

        # Use transformer to convert messages with ID overrides
        transformed = self._transformer.transform_batch(self._base_messages)
        for msg in transformed:
            for part in msg.parts:
                builder.add(part)

        return builder.messages

    @cached_property
    def instructions(self) -> str:
        """Build runtime instructions with workspace context and dashboard info."""
        lines: list[str] = []

        lines.append("Following is context about the current active OpenBB Workspace:")

        if self.deps.timezone:
            lines.append(f"The user is in timezone: {self.deps.timezone}")
            current_time = datetime.now(ZoneInfo(self.deps.timezone)).isoformat()
            lines.append(f"Current date and time: {current_time}")

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
        dashboard = workspace_state.current_dashboard_info if workspace_state else None

        if dashboard:
            lines.extend(self._dashboard_context_lines(dashboard))
        else:
            widget_defaults = self._widget_default_lines()
            if widget_defaults:
                lines.append("<widget_defaults>")
                lines.append(
                    "Preloaded widget values (reuse unless the user requests different data):"  # noqa: E501
                )
                lines.extend(widget_defaults)
                lines.append("</widget_defaults>")

        return "\n".join(lines)

    def _dashboard_context_lines(self, dashboard: Any) -> list[str]:
        """Build prompt lines for dashboard info and widget values."""
        lines = ["<dashboard_info>"]
        lines.append(f"Active dashboard: {dashboard.name}")
        lines.append(f"Current tab: {dashboard.current_tab_id}")
        lines.append("")

        processed_uuids = set()

        if dashboard.tabs:
            lines.append("Widgets by Tab:")
            for tab in dashboard.tabs:
                lines.append(f"## {tab.tab_id}")

                if not tab.widgets:
                    lines.append("(No widgets)")
                    continue

                for widget_ref in tab.widgets:
                    widget = self.deps.get_widget_by_uuid(widget_ref.widget_uuid)
                    if widget:
                        processed_uuids.add(str(widget.uuid))
                        params_str = self._format_widget_params(widget)
                        name = widget_ref.name or widget.name or widget.widget_id
                        if params_str:
                            lines.append(f"- {name}: {params_str}")
                        else:
                            lines.append(f"- {name}")
                    else:
                        lines.append(f"- {widget_ref.name}")
                lines.append("")

        # Handle widgets not in dashboard
        all_widgets = list(self.deps.iter_widgets())
        orphan_widgets = [w for w in all_widgets if str(w.uuid) not in processed_uuids]

        if orphan_widgets:
            lines.append("Other Available Widgets:")
            for widget in orphan_widgets:
                params_str = self._format_widget_params(widget)
                name = widget.name or widget.widget_id
                if params_str:
                    lines.append(f"- {name}: {params_str}")
                else:
                    lines.append(f"- {name}")

        lines.append("</dashboard_info>")
        return lines

    def _widget_default_lines(self) -> list[str]:
        """Build prompt lines summarizing current or default widget parameter values."""
        if not self.deps.widgets:
            return []

        widgets_iter = self.deps.iter_widgets()

        lines: list[str] = []
        for widget in widgets_iter:
            params_str = self._format_widget_params(widget)
            if params_str:
                widget_name = widget.name or widget.widget_id
                lines.append(f"- {widget_name}: {params_str}")

        return lines

    def _format_widget_params(self, widget: Widget) -> str | None:
        """Format widget parameters into a string."""
        params = getattr(widget, "params", None)
        if not params:
            return None

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
                entry += ""
            elif source == "default":
                entry += " (no current value showing default)"
            param_entries.append(entry)

        if not param_entries:
            return None

        return ", ".join(param_entries)

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
        if not self._toolsets:
            return None
        if len(self._toolsets) == 1:
            return self._toolsets[0]
        combined = CombinedToolset(self._toolsets)
        return cast(AbstractToolset[OpenBBDeps], combined)

    @cached_property
    def state(self) -> dict[str, Any] | None:
        """Extract workspace state as a dictionary."""
        return (
            self.run_input.workspace_state.model_dump(exclude_none=True)
            if self.run_input.workspace_state
            else None
        )

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

        # Merge dynamic dashboard context with caller-provided instructions
        combined_instructions = self.instructions
        if instructions:
            if isinstance(instructions, str):
                # Avoid duplication if caller already merged context
                if combined_instructions not in instructions:
                    combined_instructions = f"{combined_instructions}\n\n{instructions}"
                else:
                    combined_instructions = instructions
            elif isinstance(instructions, list):
                combined_instructions = [combined_instructions] + instructions

        return super().run_stream_native(
            output_type=output_type,
            message_history=message_history,
            deferred_tool_results=deferred_tool_results,
            model=model,
            instructions=combined_instructions,
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

        # Merge dynamic dashboard context with caller-provided instructions
        combined_instructions = self.instructions
        if instructions:
            if isinstance(instructions, str):
                # Avoid duplication if caller already merged context
                if combined_instructions not in instructions:
                    combined_instructions = f"{combined_instructions}\n\n{instructions}"
                else:
                    combined_instructions = instructions
            elif isinstance(instructions, list):
                combined_instructions = [combined_instructions] + instructions

        return super().run_stream(
            output_type=output_type,
            message_history=message_history,
            deferred_tool_results=deferred_tool_results,
            model=model,
            instructions=combined_instructions,
            deps=resolved_deps,
            model_settings=model_settings,
            usage_limits=usage_limits,
            usage=usage,
            infer_name=infer_name,
            toolsets=toolsets,
            builtin_tools=builtin_tools,
            on_complete=on_complete,
        )
