"""Message transformation utilities for OpenBB Pydantic AI adapter."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from typing import Any

from openbb_ai.models import (
    LlmClientFunctionCall,
    LlmClientFunctionCallResultMessage,
    LlmClientMessage,
    LlmMessage,
    RoleEnum,
)
from pydantic_ai.messages import (
    ModelMessage,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.ui import MessagesBuilder

from openbb_pydantic_ai._config import (
    EXECUTE_MCP_TOOL_NAME,
    GET_WIDGET_DATA_TOOL_NAME,
    LOCAL_TOOL_CAPSULE_REHYDRATED_KEY,
    LOCAL_TOOL_CAPSULE_RESULT_KEY,
)
from openbb_pydantic_ai._serializers import serialize_result
from openbb_pydantic_ai._utils import extract_tool_call_id

# UI protocol function names that should be rewritten to `call_tools`
_REWRITABLE_FUNCTIONS = frozenset({GET_WIDGET_DATA_TOOL_NAME, EXECUTE_MCP_TOOL_NAME})

_CALL_TOOLS = "call_tools"


class MessageTransformer:
    """Transforms OpenBB messages to Pydantic AI messages."""

    def __init__(self, *, rewrite_deferred_tool_names: bool = False) -> None:
        self._rewrite = rewrite_deferred_tool_names

    def transform_batch(self, messages: Sequence[LlmMessage]) -> list[ModelMessage]:
        """Transform a batch of OpenBB messages to Pydantic AI messages.

        Parameters
        ----------
        messages : Sequence[LlmMessage]
            List of OpenBB messages to transform

        Returns
        -------
        list[ModelMessage]
            List of Pydantic AI messages
        """
        tool_call_id_map = self._build_tool_call_id_map(messages)
        tool_name_map = self._build_tool_name_map(messages) if self._rewrite else {}
        call_counters: dict[str, int] = {}

        builder = MessagesBuilder()
        for message in messages:
            if isinstance(message, LlmClientMessage):
                self._add_client_message(
                    builder, message, tool_call_id_map, call_counters, tool_name_map
                )
            elif isinstance(message, LlmClientFunctionCallResultMessage):
                self._add_result_message(builder, message, tool_name_map)
        return builder.messages

    def _build_tool_call_id_map(
        self, messages: Sequence[LlmMessage]
    ) -> dict[str, list[str]]:
        """Build a lookup of function_name -> all tool_call_ids in order.

        Since the same function (e.g., get_widget_data) can be called multiple times,
        this accumulates ALL tool_call_ids for each function across all result messages,
        preserving order. These IDs are consumed sequentially via a counter when
        processing deferred tool calls.

        Parameters
        ----------
        messages : Sequence[LlmMessage]
            All messages to extract tool_call_ids from

        Returns
        -------
        dict[str, list[str]]
            Map of function names to ordered lists of tool_call_ids
        """
        id_map: defaultdict[str, list[str]] = defaultdict(list)

        for msg in messages:
            if not isinstance(msg, LlmClientFunctionCallResultMessage):
                continue

            if (extra_state := msg.extra_state) and (
                tool_calls := extra_state.get("tool_calls")
            ):
                # Extract tool_call_ids preserving order
                ids = [
                    tc["tool_call_id"]
                    for tc in tool_calls
                    if isinstance(tc, dict) and "tool_call_id" in tc
                ]
                if ids:
                    id_map[msg.function].extend(ids)

        return dict(id_map)

    @staticmethod
    def _build_tool_name_map(messages: Sequence[LlmMessage]) -> dict[str, str]:
        """Build tool_call_id -> pydantic-ai tool_name from extra_state.

        Used when rewriting UI protocol names (get_widget_data, execute_agent_tool)
        back to `call_tools`.
        """
        name_map: dict[str, str] = {}
        for msg in messages:
            if not isinstance(msg, LlmClientFunctionCallResultMessage):
                continue
            extra_state = msg.extra_state or {}
            tool_calls = extra_state.get("tool_calls", [])
            if not isinstance(tool_calls, list):
                continue
            for tc in tool_calls:
                if not isinstance(tc, dict):
                    continue
                tc_id = tc.get("tool_call_id")
                tc_name = tc.get("tool_name")
                if isinstance(tc_id, str) and isinstance(tc_name, str):
                    name_map[tc_id] = tc_name
        return name_map

    def _rewrite_tool_call(
        self,
        function_name: str,
        tool_call_id: str,
        args: dict[str, Any] | None,
        tool_name_map: dict[str, str],
    ) -> tuple[str, dict[str, Any] | None]:
        """Rewrite a UI protocol tool call to `call_tools` if applicable."""
        if not self._should_rewrite(function_name, tool_call_id, tool_name_map):
            return function_name, args

        pydantic_tool_name = tool_name_map.get(tool_call_id)
        assert pydantic_tool_name is not None

        normalized_args = self._normalize_rewritten_call_args(function_name, args)

        # Wrap original args inside call_tools' expected shape
        return _CALL_TOOLS, {
            "calls": [{"tool_name": pydantic_tool_name, "arguments": normalized_args}],
        }

    @staticmethod
    def _normalize_rewritten_call_args(
        function_name: str,
        args: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Normalize rewritten call args for protocol wrapper functions.

        UI protocol wrappers (like execute_agent_tool) include transport-level
        fields that do not belong in the underlying tool schema. When rewriting
        these calls back to `call_tools`, keep only the nested tool arguments.
        """
        if not isinstance(args, dict):
            return {}

        if function_name == EXECUTE_MCP_TOOL_NAME:
            nested = args.get("parameters")
            if isinstance(nested, dict):
                return nested

        return args

    def _should_rewrite(
        self,
        function_name: str,
        tool_call_id: str,
        tool_name_map: dict[str, str] | None,
    ) -> bool:
        """Check whether a specific tool call can be rewritten to `call_tools`."""
        return bool(
            self._rewrite
            and function_name in _REWRITABLE_FUNCTIONS
            and tool_name_map
            and tool_call_id in tool_name_map
        )

    def _add_client_message(
        self,
        builder: MessagesBuilder,
        message: LlmClientMessage,
        tool_call_id_map: dict[str, list[str]],
        call_counters: dict[str, int],
        tool_name_map: dict[str, str],
    ) -> None:
        """Add a client message to the builder.

        Parameters
        ----------
        builder : MessagesBuilder
            The message builder to add to
        message : LlmClientMessage
            The client message to add
        tool_call_id_map : dict[str, list[str]]
            Prebuilt map of function names to tool_call_ids
        call_counters : dict[str, int]
            Counter tracking which tool_call_id index to use per function
        tool_name_map : dict[str, str]
            Map of tool_call_id to pydantic-ai tool name for rewriting
        """
        content = message.content

        if isinstance(content, LlmClientFunctionCall):
            function_name = content.function
            tool_call_ids = tool_call_id_map.get(function_name, [])

            input_args = content.input_arguments or {}
            data_sources = input_args.get("data_sources", [])

            if isinstance(data_sources, list) and len(data_sources) > 1:
                counter = call_counters.get(function_name, 0)
                for i, data_source in enumerate(data_sources):
                    idx = counter + i
                    if idx >= len(tool_call_ids):
                        raise ValueError(
                            f"""
                            Not enough tool_call_ids for batched call to {function_name}
                            """.strip()
                        )

                    tc_id = tool_call_ids[idx]
                    source_args: dict[str, Any] = {"data_sources": [data_source]}
                    rewritten_name, rewritten_args = self._rewrite_tool_call(
                        function_name, tc_id, source_args, tool_name_map
                    )

                    builder.add(
                        ToolCallPart(
                            tool_name=rewritten_name,
                            tool_call_id=tc_id,
                            args=rewritten_args,
                        )
                    )

                call_counters[function_name] = counter + len(data_sources)
                return

            counter = call_counters.get(function_name, 0)
            if counter >= len(tool_call_ids):
                raise ValueError(
                    """
                    `tool_call_id` is required for deferred tool calls
                    but not enough IDs were found in prior result messages
                    """.strip()
                )
            tool_call_id = tool_call_ids[counter]
            call_counters[function_name] = counter + 1

            rewritten_name, rewritten_args = self._rewrite_tool_call(
                content.function, tool_call_id, content.input_arguments, tool_name_map
            )

            builder.add(
                ToolCallPart(
                    tool_name=rewritten_name,
                    tool_call_id=tool_call_id,
                    args=rewritten_args,
                )
            )
            return

        if isinstance(content, str):
            if message.role == RoleEnum.human:
                builder.add(UserPromptPart(content=content))
            elif message.role in (RoleEnum.ai, RoleEnum.tool):
                builder.add(TextPart(content=content))
            else:
                builder.add(TextPart(content=content))

    def _add_result_message(
        self,
        builder: MessagesBuilder,
        message: LlmClientFunctionCallResultMessage,
        tool_name_map: dict[str, str] | None = None,
    ) -> None:
        """Add a function call result message to the builder.

        For batched results (detected by extra_state.tool_calls array), unbatch
        them into individual ToolReturnPart messages, one per tool call.

        Parameters
        ----------
        builder : MessagesBuilder
            The message builder to add to
        message : LlmClientFunctionCallResultMessage
            The result message to add
        tool_name_map : dict[str, str] | None
            Map of tool_call_id to pydantic-ai tool name for rewriting
        """
        extra_state = message.extra_state or {}
        tool_calls = extra_state.get("tool_calls", [])

        if isinstance(tool_calls, list) and len(tool_calls) > 1:
            self._add_unbatched_results(builder, message, tool_name_map)
            return

        tool_call_id = extract_tool_call_id(message)
        result_tool_name = message.function

        if self._should_rewrite(result_tool_name, tool_call_id, tool_name_map):
            result_tool_name = _CALL_TOOLS

        result_content = self._result_content(message)
        builder.add(
            ToolReturnPart(
                tool_name=result_tool_name,
                tool_call_id=tool_call_id,
                content=result_content,
            )
        )

    def _add_unbatched_results(
        self,
        builder: MessagesBuilder,
        message: LlmClientFunctionCallResultMessage,
        tool_name_map: dict[str, str] | None = None,
    ) -> None:
        """Unbatch a result with multiple tool_calls into individual ToolReturnParts.

        Matches data[i] with tool_calls[i] to preserve tool_call_id association.
        This handles batched widget calls (get_widget_data) or any other
        batched tool types.

        Parameters
        ----------
        builder : MessagesBuilder
            The message builder to add to
        message : LlmClientFunctionCallResultMessage
            The batched result message
        tool_name_map : dict[str, str] | None
            Map of tool_call_id to pydantic-ai tool name for rewriting
        """
        extra_state = message.extra_state or {}
        tool_calls = extra_state.get("tool_calls", [])
        data = message.data or []

        for idx, tool_call_info in enumerate(tool_calls):
            if not isinstance(tool_call_info, dict):
                continue

            tool_call_id = tool_call_info.get("tool_call_id")
            if not isinstance(tool_call_id, str):
                continue

            result_data = data[idx] if idx < len(data) else None

            result_tool_name = message.function
            if self._should_rewrite(result_tool_name, tool_call_id, tool_name_map):
                result_tool_name = _CALL_TOOLS

            builder.add(
                ToolReturnPart(
                    tool_name=result_tool_name,
                    tool_call_id=tool_call_id,
                    content=result_data,
                )
            )

    @staticmethod
    def _result_content(message: LlmClientFunctionCallResultMessage) -> Any:
        extra_state = message.extra_state or {}
        if (
            isinstance(extra_state, dict)
            and extra_state.get(LOCAL_TOOL_CAPSULE_REHYDRATED_KEY) is True
            and LOCAL_TOOL_CAPSULE_RESULT_KEY in extra_state
        ):
            return extra_state[LOCAL_TOOL_CAPSULE_RESULT_KEY]

        return serialize_result(message)
