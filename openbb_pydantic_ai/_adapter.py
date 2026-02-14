from __future__ import annotations

import logging
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from functools import cached_property
from http import HTTPStatus
from typing import TYPE_CHECKING, Any, Callable, Protocol, cast, runtime_checkable
from zoneinfo import ZoneInfo

from openbb_ai.models import (
    SSE,
    AgentTool,
    ClientCommandResult,
    DashboardInfo,
    LlmClientFunctionCall,
    LlmClientFunctionCallResultMessage,
    LlmClientMessage,
    LlmMessage,
    QueryRequest,
    RoleEnum,
    Undefined,
    Widget,
    WidgetCollection,
)
from pydantic import ValidationError
from pydantic_ai.agent import AbstractAgent, AgentMetadata
from pydantic_ai.agent.abstract import Instructions
from pydantic_ai.builtin_tools import AbstractBuiltinTool
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models import KnownModelName, Model
from pydantic_ai.output import OutputSpec
from pydantic_ai.settings import ModelSettings
from pydantic_ai.toolsets import AbstractToolset, CombinedToolset, FunctionToolset
from pydantic_ai.ui import OnCompleteFunc, UIAdapter
from pydantic_ai.usage import RunUsage, UsageLimits
from typing_extensions import override

from openbb_pydantic_ai._config import (
    LOCAL_TOOL_CAPSULE_EXTRA_STATE_KEY,
    LOCAL_TOOL_CAPSULE_REHYDRATED_KEY,
    LOCAL_TOOL_CAPSULE_RESULT_KEY,
)
from openbb_pydantic_ai._dependencies import OpenBBDeps, build_deps_from_request
from openbb_pydantic_ai._event_stream import OpenBBAIEventStream
from openbb_pydantic_ai._local_tool_capsule import (
    LocalToolEntry,
    unpack_tool_history,
)
from openbb_pydantic_ai._mcp_toolsets import build_mcp_toolsets
from openbb_pydantic_ai._message_transformer import MessageTransformer
from openbb_pydantic_ai._pdf_preprocess import preprocess_pdf_in_messages
from openbb_pydantic_ai._viz_toolsets import build_viz_toolsets
from openbb_pydantic_ai._widget_registry import WidgetRegistry
from openbb_pydantic_ai._widget_toolsets import build_widget_toolsets
from openbb_pydantic_ai.tool_discovery import ToolDiscoveryToolset
from openbb_pydantic_ai.tool_discovery.progressive import (
    ProgressiveConfig,
    get_progressive_config,
)

if TYPE_CHECKING:
    from pydantic_ai import DeferredToolResults
    from starlette.requests import Request
    from starlette.responses import Response

logger = logging.getLogger(__name__)


@runtime_checkable
class _HasToolsByName(Protocol):
    tools_by_name: Mapping[str, AgentTool]


@runtime_checkable
class _HasFuncAttr(Protocol):
    func: Callable[..., Any]


@runtime_checkable
class _HasFunctionAttr(Protocol):
    function: Callable[..., Any]


@runtime_checkable
class _HasPrivateFuncAttr(Protocol):
    _func: Callable[..., Any]


@dataclass(kw_only=True)
class OpenBBAIAdapter(UIAdapter[QueryRequest, LlmMessage, SSE, OpenBBDeps, Any]):
    """UI adapter that bridges OpenBB Workspace requests with Pydantic AI."""

    _PROGRESSIVE_CACHE_KEYS = (
        "_progressive_named_toolsets",
        "_progressive_group_descriptions",
        "_progressive_toolset",
        "instructions",
        "toolset",
    )

    accept: str | None = None
    enable_progressive_tool_discovery: bool = True
    enable_local_tool_history_capsule: bool = True

    # Initialized in __post_init__
    _transformer: MessageTransformer = field(init=False)
    _registry: WidgetRegistry = field(init=False)
    _base_messages: list[LlmMessage] = field(init=False)
    _pending_results: list[LlmClientFunctionCallResultMessage] = field(init=False)
    _runtime_progressive_toolsets: tuple[
        tuple[str, AbstractToolset[OpenBBDeps]], ...
    ] = field(init=False, default=())
    _runtime_progressive_descriptions: dict[str, str] = field(
        init=False,
        default_factory=dict,
    )

    def __post_init__(self) -> None:
        (
            self._base_messages,
            self._pending_results,
        ) = self._split_messages(self.run_input.messages)

        # Initialize transformer and registry
        self._transformer = MessageTransformer(
            rewrite_deferred_tool_names=self.enable_progressive_tool_discovery,
        )
        self._registry = WidgetRegistry(
            collection=self.run_input.widgets,
            toolsets=self._widget_toolsets,
        )

    @classmethod
    def build_run_input(cls, body: bytes) -> QueryRequest:
        """Parse the raw request body into a ``QueryRequest`` instance."""
        return QueryRequest.model_validate_json(body)

    @classmethod
    async def from_request(
        cls,
        request: Request,
        *,
        agent: AbstractAgent[OpenBBDeps, Any],
        enable_progressive_tool_discovery: bool = True,
        enable_local_tool_history_capsule: bool = True,
    ) -> OpenBBAIAdapter:
        """Create adapter and preprocess PDF payloads before message transforms."""
        run_input = cls.build_run_input(await request.body())
        run_input = await cls._preprocess_run_input(run_input)
        adapter = cls(
            agent=agent,
            run_input=run_input,
            accept=request.headers.get("accept"),
            enable_progressive_tool_discovery=enable_progressive_tool_discovery,
            enable_local_tool_history_capsule=enable_local_tool_history_capsule,
        )
        await adapter._rehydrate_local_capsules()
        return adapter

    @classmethod
    async def _preprocess_run_input(cls, run_input: QueryRequest) -> QueryRequest:
        """Preprocess PDF-bearing result messages in full history."""
        processed_messages = await preprocess_pdf_in_messages(list(run_input.messages))
        if len(processed_messages) == len(run_input.messages) and all(
            old is new
            for old, new in zip(run_input.messages, processed_messages, strict=True)
        ):
            return run_input

        return run_input.model_copy(update={"messages": processed_messages})

    @classmethod
    def load_messages(cls, messages: Sequence[LlmMessage]) -> list[ModelMessage]:
        """Convert OpenBB messages to Pydantic AI messages."""
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

    async def _rehydrate_local_capsules(self) -> None:
        """Inject rehydrated local tool messages from stateless extra_state capsules."""
        if not self.enable_local_tool_history_capsule:
            return

        original_messages = list(self.run_input.messages)
        if not original_messages:
            return

        rehydrated = await self._messages_with_rehydrated_capsules(original_messages)
        if len(rehydrated) == len(original_messages):
            return

        self.run_input = self.run_input.model_copy(update={"messages": rehydrated})
        self._base_messages, self._pending_results = self._split_messages(rehydrated)
        self.__dict__.pop("messages", None)

    async def _messages_with_rehydrated_capsules(
        self,
        messages: Sequence[LlmMessage],
    ) -> list[LlmMessage]:
        rehydrated: list[LlmMessage] = []
        seen_capsules: set[str] = set()
        paired_result_indices: set[int] = set()

        for index, message in enumerate(messages):
            if self._is_function_call_message(message):
                next_message = (
                    messages[index + 1] if index + 1 < len(messages) else None
                )
                if isinstance(next_message, LlmClientFunctionCallResultMessage):
                    if await self._try_rehydrate_capsule(
                        next_message, rehydrated, seen_capsules
                    ):
                        paired_result_indices.add(index + 1)

                rehydrated.append(message)
                continue

            if (
                isinstance(message, LlmClientFunctionCallResultMessage)
                and index not in paired_result_indices
            ):
                await self._try_rehydrate_capsule(message, rehydrated, seen_capsules)

            rehydrated.append(message)

        return rehydrated

    async def _try_rehydrate_capsule(
        self,
        message: LlmClientFunctionCallResultMessage,
        target: list[LlmMessage],
        seen: set[str],
    ) -> bool:
        """Decode capsule and append rehydrated messages. Returns True if rehydrated."""
        entries, capsule_key = self._decode_local_capsule(message)
        if entries is None or capsule_key is None or capsule_key in seen:
            return False
        target.extend(await self._rehydrated_messages_from_entries(entries))
        seen.add(capsule_key)
        return True

    @staticmethod
    def _is_function_call_message(message: LlmMessage) -> bool:
        return isinstance(message, LlmClientMessage) and isinstance(
            message.content, LlmClientFunctionCall
        )

    def _decode_local_capsule(
        self, message: LlmClientFunctionCallResultMessage
    ) -> tuple[list[LocalToolEntry] | None, str | None]:
        extra_state = message.extra_state or {}
        raw_capsule = extra_state.get(LOCAL_TOOL_CAPSULE_EXTRA_STATE_KEY)
        if raw_capsule is None:
            return None, None

        if not isinstance(raw_capsule, str):
            logger.warning(
                "Ignoring invalid local-tool capsule in message history: "
                "expected string payload"
            )
            return None, None

        try:
            return unpack_tool_history(raw_capsule), raw_capsule
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Ignoring invalid local-tool capsule in message history: %s",
                exc,
            )
            return None, None

    async def _rehydrated_messages_from_entries(
        self,
        entries: Sequence[LocalToolEntry],
    ) -> list[LlmMessage]:
        messages: list[LlmMessage] = []
        for entry in entries:
            args = dict(entry.args)
            messages.extend(
                [
                    LlmClientMessage(
                        role=RoleEnum.ai,
                        content=LlmClientFunctionCall(
                            function=entry.tool_name,
                            input_arguments=args,
                        ),
                    ),
                    LlmClientFunctionCallResultMessage(
                        function=entry.tool_name,
                        input_arguments=args,
                        data=[ClientCommandResult(status="success", message=None)],
                        extra_state={
                            LOCAL_TOOL_CAPSULE_REHYDRATED_KEY: True,
                            LOCAL_TOOL_CAPSULE_RESULT_KEY: entry.result,
                            "tool_calls": [
                                {
                                    "tool_call_id": entry.tool_call_id,
                                    "tool_name": entry.tool_name,
                                }
                            ],
                        },
                    ),
                ]
            )
        return messages

    @cached_property
    def deps(self) -> OpenBBDeps:
        return build_deps_from_request(self.run_input)

    @cached_property
    def _widget_toolsets(self) -> tuple[AbstractToolset[OpenBBDeps], ...]:
        return build_widget_toolsets(self.run_input.widgets)

    @cached_property
    def _viz_toolset(self) -> AbstractToolset[OpenBBDeps]:
        return build_viz_toolsets()

    @cached_property
    def _mcp_toolsets(self) -> tuple[AbstractToolset[Any], ...]:
        return build_mcp_toolsets(self.run_input.tools)

    @cached_property
    def _pdf_toolset(self) -> AbstractToolset[OpenBBDeps] | None:
        try:
            from openbb_pydantic_ai.pdf._toolsets import build_pdf_toolset
        except ImportError:
            return None

        return build_pdf_toolset()

    @cached_property
    def _toolsets(self) -> tuple[AbstractToolset[OpenBBDeps], ...]:
        toolsets: list[AbstractToolset[OpenBBDeps]] = list(self._widget_toolsets)
        toolsets.append(self._viz_toolset)
        if self._pdf_toolset is not None:
            toolsets.append(self._pdf_toolset)
        if self._mcp_toolsets:
            toolsets.extend(
                cast("Sequence[AbstractToolset[OpenBBDeps]]", self._mcp_toolsets)
            )
        return tuple(toolsets)

    @cached_property
    def _mcp_tool_lookup(self) -> dict[str, AgentTool]:
        tools: dict[str, AgentTool] = {}
        for toolset in self._mcp_toolsets:
            if isinstance(toolset, _HasToolsByName):
                tools.update(toolset.tools_by_name)
        return tools

    @cached_property
    def _progressive_named_toolsets(
        self,
    ) -> tuple[tuple[str, AbstractToolset[OpenBBDeps]], ...]:
        named: list[tuple[str, AbstractToolset[OpenBBDeps]]] = []

        collection = self.run_input.widgets or WidgetCollection()
        widget_groups = (
            ("openbb_widgets_primary", collection.primary),
            ("openbb_widgets_secondary", collection.secondary),
            ("openbb_widgets_extra", collection.extra),
        )
        widget_toolsets_iter = iter(self._widget_toolsets)
        for group_id, widgets in widget_groups:
            if widgets:
                toolset = next(widget_toolsets_iter, None)
                if toolset is not None:
                    named.append((group_id, toolset))

        named.append(("openbb_viz_tools", self._viz_toolset))

        mcp_toolsets = cast("Sequence[AbstractToolset[OpenBBDeps]]", self._mcp_toolsets)
        if len(mcp_toolsets) == 1:
            named.append(("openbb_mcp_tools", mcp_toolsets[0]))
        else:
            for index, toolset in enumerate(mcp_toolsets):
                named.append((f"openbb_mcp_tools_{index}", toolset))

        if self._runtime_progressive_toolsets:
            named.extend(self._runtime_progressive_toolsets)

        return tuple(named)

    @cached_property
    def _progressive_group_descriptions(self) -> dict[str, str]:
        descriptions: dict[str, str] = {
            "openbb_widgets_primary": "Primary dashboard widget tools",
            "openbb_widgets_secondary": "Secondary dashboard widget tools",
            "openbb_widgets_extra": "Additional dashboard widget tools",
            "openbb_viz_tools": "OpenBB table/chart/html artifact tools",
            "openbb_mcp_tools": "Workspace-selected MCP tools",
        }
        for group_id, _toolset in self._progressive_named_toolsets:
            if group_id.startswith("openbb_mcp_tools_"):
                descriptions[group_id] = "Workspace-selected MCP tools"
            elif group_id not in descriptions:
                descriptions[group_id] = "Additional OpenBB toolset"

        if self._runtime_progressive_descriptions:
            descriptions.update(self._runtime_progressive_descriptions)
        return descriptions

    @cached_property
    def _progressive_toolset(self) -> ToolDiscoveryToolset:
        return ToolDiscoveryToolset(
            toolsets=self._progressive_named_toolsets,
            group_descriptions=self._progressive_group_descriptions,
            id="openbb_progressive_tools",
        )

    @staticmethod
    def _function_tool_callable(tool: Any) -> Any | None:
        if isinstance(tool, _HasFuncAttr) and callable(tool.func):
            return tool.func
        if isinstance(tool, _HasFunctionAttr) and callable(tool.function):
            return tool.function
        if isinstance(tool, _HasPrivateFuncAttr) and callable(tool._func):
            return tool._func
        return None

    def _invalidate_progressive_cache(self) -> None:
        for cache_name in self._PROGRESSIVE_CACHE_KEYS:
            self.__dict__.pop(cache_name, None)

    def _merge_instructions(
        self, instructions: Instructions[OpenBBDeps]
    ) -> Instructions[OpenBBDeps]:
        combined: Instructions[OpenBBDeps] = self.instructions
        if instructions:
            if isinstance(instructions, str):
                # Avoid duplication if caller already merged context
                if combined not in instructions:
                    combined = f"{combined}\n\n{instructions}"
                else:
                    combined = instructions
            elif isinstance(instructions, list):
                combined = [combined] + instructions
            else:
                combined = instructions
        return combined

    def _infer_progressive_config_from_functions(
        self, toolset: AbstractToolset[OpenBBDeps]
    ) -> ProgressiveConfig | None:
        if not isinstance(toolset, FunctionToolset):
            return None

        tools_map = getattr(toolset, "tools", None)
        if not isinstance(tools_map, dict):
            return None

        matched_configs: list[ProgressiveConfig] = []
        for tool in tools_map.values():
            func = self._function_tool_callable(tool)
            if func is None:
                continue
            cfg = get_progressive_config(func)
            if cfg is not None:
                matched_configs.append(cfg)

        if not matched_configs:
            return None

        groups = {cfg.group for cfg in matched_configs if cfg.group}
        if len(groups) > 1:
            warnings.warn(
                "Multiple @progressive groups were used in one FunctionToolset. "
                "Using toolset id/default group for this toolset.",
                UserWarning,
                stacklevel=2,
            )
            group: str | None = None
        else:
            group = next(iter(groups)) if groups else None

        descriptions = [cfg.description for cfg in matched_configs if cfg.description]
        description = descriptions[0] if descriptions else None
        return ProgressiveConfig(group=group, description=description)

    def _progressive_config_for_toolset(
        self, toolset: AbstractToolset[OpenBBDeps]
    ) -> ProgressiveConfig | None:
        direct = get_progressive_config(toolset)
        if direct is not None:
            return direct
        return self._infer_progressive_config_from_functions(toolset)

    def _apply_runtime_progressive_toolsets(
        self,
        toolsets: Sequence[AbstractToolset[OpenBBDeps]] | None,
    ) -> Sequence[AbstractToolset[OpenBBDeps]] | None:
        if not self.enable_progressive_tool_discovery or not toolsets:
            return toolsets

        # Clear previous runtime state to prevent accumulation across repeated calls
        self._runtime_progressive_toolsets = ()
        self._runtime_progressive_descriptions = {}
        self._invalidate_progressive_cache()

        # Compute used_group_ids from base toolsets only (runtime now cleared)
        used_group_ids = {group_id for group_id, _ in self._progressive_named_toolsets}
        runtime_named: list[tuple[str, AbstractToolset[OpenBBDeps]]] = []
        runtime_descriptions: dict[str, str] = {}
        passthrough: list[AbstractToolset[OpenBBDeps]] = []

        for index, toolset in enumerate(toolsets):
            cfg = self._progressive_config_for_toolset(toolset)
            if cfg is None:
                passthrough.append(toolset)
                continue

            base_group = cfg.group or str(
                getattr(toolset, "id", None) or f"user_toolset_{index}"
            )
            group_id = base_group
            suffix = 1
            while group_id in used_group_ids:
                group_id = f"{base_group}_{suffix}"
                suffix += 1
            used_group_ids.add(group_id)

            runtime_named.append((group_id, toolset))
            if cfg.description:
                runtime_descriptions[group_id] = cfg.description

        if not runtime_named:
            return toolsets

        self._runtime_progressive_toolsets = tuple(runtime_named)
        self._runtime_progressive_descriptions = runtime_descriptions

        # Recompute cached values that include progressive toolset/instructions.
        self._invalidate_progressive_cache()

        return tuple(passthrough) if passthrough else None

    def build_event_stream(self) -> OpenBBAIEventStream:
        """Create the event stream wrapper for this adapter run."""

        return OpenBBAIEventStream(
            run_input=self.run_input,
            widget_registry=self._registry,
            pending_results=self._pending_results,
            mcp_tools=self._mcp_tool_lookup or None,
            enable_local_tool_history_capsule=self.enable_local_tool_history_capsule,
        )

    @cached_property
    def messages(self) -> list[ModelMessage]:
        """Build message history with context prompts."""
        return self._transformer.transform_batch(self._base_messages)

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

        if self.enable_progressive_tool_discovery and self._toolsets:
            progressive_instructions = self._progressive_toolset.render_instructions()
            if progressive_instructions:
                lines.append("")
                lines.append(progressive_instructions)

        return "\n".join(lines)

    def _widget_line(self, name: str, widget: Widget) -> str:
        """Format a single widget as a prompt line item."""
        params_str = self._format_widget_params(widget)
        return f"- {name}: {params_str}" if params_str else f"- {name}"

    def _dashboard_context_lines(self, dashboard: DashboardInfo) -> list[str]:
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
                        name = widget_ref.name or widget.name or widget.widget_id
                        lines.append(self._widget_line(name, widget))
                    else:
                        lines.append(f"- {widget_ref.name}")
                lines.append("")

        # Handle widgets not in dashboard
        orphan_widgets = [
            w for w in self.deps.iter_widgets() if str(w.uuid) not in processed_uuids
        ]

        if orphan_widgets:
            lines.append("Other Available Widgets:")
            for widget in orphan_widgets:
                name = widget.name or widget.widget_id
                lines.append(self._widget_line(name, widget))

        lines.append("</dashboard_info>")
        return lines

    def _widget_default_lines(self) -> list[str]:
        """Build prompt lines summarizing current or default widget parameter values."""
        if not self.deps.widgets:
            return []

        return [
            self._widget_line(w.name or w.widget_id, w)
            for w in self.deps.iter_widgets()
            if self._format_widget_params(w)
        ]

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
            if source == "default":
                entry += " (default)"
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
        if self.enable_progressive_tool_discovery:
            return cast(AbstractToolset[OpenBBDeps], self._progressive_toolset)
        if len(self._toolsets) == 1:
            return self._toolsets[0]
        combined = CombinedToolset(
            cast(Sequence[AbstractToolset[None]], self._toolsets)
        )
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
        metadata: AgentMetadata[OpenBBDeps] | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[OpenBBDeps]] | None = None,
        builtin_tools: Sequence[AbstractBuiltinTool] | None = None,
        enable_progressive_tool_discovery: bool = True,
        enable_local_tool_history_capsule: bool = True,
        on_complete: OnCompleteFunc[SSE] | None = None,
    ) -> "Response":
        """Handle a request and return a streaming response.

        Parameters
        ----------
        request : Request
            Incoming Starlette/FastAPI request.
        agent : AbstractAgent[OpenBBDeps, Any]
            Agent to run for this request.
        message_history : Sequence[ModelMessage] | None, default None
            Existing model message history.
        deferred_tool_results : DeferredToolResults | None, default None
            Deferred tool results for continuation.
        model : Model | KnownModelName | str | None, default None
            Optional model override.
        instructions : Instructions[OpenBBDeps], default None
            Additional run instructions.
        deps : OpenBBDeps | None, default None
            Optional dependency overrides.
        output_type : OutputSpec[Any] | None, default None
            Optional output type override.
        model_settings : ModelSettings | None, default None
            Optional model settings override.
        usage_limits : UsageLimits | None, default None
            Optional token/request limits.
        usage : RunUsage | None, default None
            Optional usage accumulator.
        metadata : AgentMetadata[OpenBBDeps] | None, default None
            Optional run metadata.
        infer_name : bool, default True
            Whether to infer the agent name.
        toolsets : Sequence[AbstractToolset[OpenBBDeps]] | None, default None
            Additional runtime toolsets.
        builtin_tools : Sequence[AbstractBuiltinTool] | None, default None
            Additional builtin tools.
        enable_progressive_tool_discovery : bool, default True
            Whether to enable progressive tool discovery for this request.
        enable_local_tool_history_capsule : bool, default True
            Whether to capture and rehydrate local-tool history capsules.
        on_complete : OnCompleteFunc[SSE] | None, default None
            Optional completion callback.

        Returns
        -------
        Response
            Streaming response for the selected protocol.
        """
        try:
            from starlette.responses import Response
        except ImportError as e:  # pragma: no cover
            package_hint = (
                "Please install the `starlette` package to use "
                "`dispatch_request()` method, "
                "you can use the `ui` optional group — "
                '`pip install "pydantic-ai-slim[ui]"`'
            )
            raise ImportError(
                package_hint,
            ) from e

        try:
            adapter = await cls.from_request(
                request,
                agent=agent,
                enable_progressive_tool_discovery=enable_progressive_tool_discovery,
                enable_local_tool_history_capsule=enable_local_tool_history_capsule,
            )
        except ValidationError as e:  # pragma: no cover
            return Response(
                content=e.json(),
                media_type="application/json",
                status_code=HTTPStatus.UNPROCESSABLE_ENTITY,
            )

        deps_to_forward = cast(OpenBBDeps, deps)
        return adapter.streaming_response(
            adapter.run_stream(
                message_history=message_history,
                deferred_tool_results=deferred_tool_results,
                deps=deps_to_forward,
                output_type=output_type,
                model=model,
                instructions=instructions,
                model_settings=model_settings,
                usage_limits=usage_limits,
                usage=usage,
                metadata=metadata,
                infer_name=infer_name,
                toolsets=toolsets,
                builtin_tools=builtin_tools,
                on_complete=on_complete,
            ),
        )

    def _resolve_stream_defaults(
        self,
        *,
        deps: OpenBBDeps | None,
        toolsets: Sequence[AbstractToolset[OpenBBDeps]] | None,
        instructions: Instructions[OpenBBDeps],
    ) -> tuple[
        OpenBBDeps,
        Sequence[AbstractToolset[OpenBBDeps]] | None,
        Instructions[OpenBBDeps],
    ]:
        """Resolve defaults shared by stream entrypoints."""
        resolved_deps: OpenBBDeps = deps or self.deps
        resolved_toolsets = self._apply_runtime_progressive_toolsets(toolsets)
        combined_instructions = self._merge_instructions(instructions)
        return resolved_deps, resolved_toolsets, combined_instructions

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
        metadata: AgentMetadata[OpenBBDeps] | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[OpenBBDeps]] | None = None,
        builtin_tools: Sequence[AbstractBuiltinTool] | None = None,
    ):
        """
        Run the agent with OpenBB-specific defaults for
        deps, messages, and deferred results.
        """
        resolved_deps, toolsets, combined_instructions = self._resolve_stream_defaults(
            deps=deps,
            toolsets=toolsets,
            instructions=instructions,
        )

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
            metadata=metadata,
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
        metadata: AgentMetadata[OpenBBDeps] | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[OpenBBDeps]] | None = None,
        builtin_tools: Sequence[AbstractBuiltinTool] | None = None,
        on_complete: OnCompleteFunc[SSE] | None = None,
    ):
        """Run the agent and stream protocol-specific events with OpenBB defaults."""
        resolved_deps, toolsets, combined_instructions = self._resolve_stream_defaults(
            deps=deps,
            toolsets=toolsets,
            instructions=instructions,
        )

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
            metadata=metadata,
            infer_name=infer_name,
            toolsets=toolsets,
            builtin_tools=builtin_tools,
            on_complete=on_complete,
        )
