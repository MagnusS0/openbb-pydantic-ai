"""Message transformation utilities for OpenBB Pydantic AI adapter."""

from __future__ import annotations

import logging
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

# UI protocol function names that should be rewritten to `call_tools`
_REWRITABLE_FUNCTIONS = frozenset({GET_WIDGET_DATA_TOOL_NAME, EXECUTE_MCP_TOOL_NAME})

_CALL_TOOLS = "call_tools"
_DERIVED_ID_PREFIX = "openbb-pair"

logger = logging.getLogger(__name__)


class MessageTransformer:
    """Transforms OpenBB messages to Pydantic AI messages."""

    def __init__(self, *, rewrite_deferred_tool_names: bool = False) -> None:
        self._rewrite = rewrite_deferred_tool_names

    def transform_batch(self, messages: Sequence[LlmMessage]) -> list[ModelMessage]:
        """Transform a batch of OpenBB messages to Pydantic AI messages.

        Function calls and results are paired by adjacency: the Workspace always
        stores a call message immediately followed by its result message, and
        history may contain other agents' pairs or pairs with a deleted half.
        Unpaired halves are skipped so no orphan tool part reaches the model.

        Parameters
        ----------
        messages : Sequence[LlmMessage]
            List of OpenBB messages to transform

        Returns
        -------
        list[ModelMessage]
            List of Pydantic AI messages
        """
        tool_name_map = self._build_tool_name_map(messages)
        paired_ids = self._pair_function_calls(messages)

        builder = MessagesBuilder()
        for index, message in enumerate(messages):
            if isinstance(message, LlmClientMessage):
                if isinstance(message.content, LlmClientFunctionCall):
                    ids = paired_ids.get(index)
                    if ids is None:
                        logger.warning(
                            "Skipping unpaired function call '%s' in history",
                            message.content.function,
                        )
                        continue
                    self._add_function_call(
                        builder, message.content, ids, tool_name_map
                    )
                    continue
                self._add_text_message(builder, message)
            elif isinstance(message, LlmClientFunctionCallResultMessage):
                ids = paired_ids.get(index)
                if ids is None:
                    logger.warning(
                        "Skipping unpaired result message for '%s' in history",
                        message.function,
                    )
                    continue
                self._add_result_message(builder, message, ids, tool_name_map)
        return builder.messages

    @staticmethod
    def _pair_function_calls(
        messages: Sequence[LlmMessage],
    ) -> dict[int, tuple[str, ...]]:
        """Map the indices of each adjacent call/result pair to its tool_call_ids.

        IDs come from the result's `extra_state["tool_calls"]` (always attached by
        this adapter). Pairs without them get a deterministic id derived from the
        call's position, so both halves agree. Counting calls per function name
        instead would shift ids whenever history holds other agents' pairs.
        """
        paired: dict[int, tuple[str, ...]] = {}
        for index in range(len(messages) - 1):
            call_message, result = messages[index], messages[index + 1]
            if not (
                isinstance(call_message, LlmClientMessage)
                and isinstance(call_message.content, LlmClientFunctionCall)
                and isinstance(result, LlmClientFunctionCallResultMessage)
                and result.function == call_message.content.function
            ):
                continue

            function_name = call_message.content.function
            data_sources = (call_message.content.input_arguments or {}).get(
                "data_sources", []
            )
            required = (
                len(data_sources)
                if isinstance(data_sources, list) and len(data_sources) > 1
                else 1
            )

            extra_state = result.extra_state or {}
            tool_calls = extra_state.get("tool_calls")
            recorded_ids = [
                tc["tool_call_id"]
                for tc in (tool_calls if isinstance(tool_calls, list) else [])
                if isinstance(tc, dict) and isinstance(tc.get("tool_call_id"), str)
            ]
            if not recorded_ids and isinstance(extra_state.get("tool_call_id"), str):
                recorded_ids = [extra_state["tool_call_id"]]

            if required > 1:
                if len(recorded_ids) < required:
                    logger.warning(
                        "Not enough tool_call_ids for batched call to %s",
                        function_name,
                    )
                    continue
                ids = tuple(recorded_ids[:required])
            elif recorded_ids:
                ids = tuple(recorded_ids)
            else:
                ids = (f"{_DERIVED_ID_PREFIX}-{function_name}-{index}",)

            paired[index] = paired[index + 1] = ids
        return paired

    @staticmethod
    def _build_tool_name_map(messages: Sequence[LlmMessage]) -> dict[str, str]:
        """Build tool_call_id -> pydantic-ai tool_name from extra_state.

        Used when rewriting UI protocol names (get_widget_data, execute_agent_tool)
        back to `call_tools` or direct MCP tool names.
        """
        name_map: dict[str, str] = {}
        for msg in messages:
            if not isinstance(msg, LlmClientFunctionCallResultMessage):
                continue
            extra_state = msg.extra_state or {}
            tool_calls = extra_state.get("tool_calls", [])
            if not isinstance(tool_calls, list):
                continue
            has_single_tool_call = len(tool_calls) == 1
            for tc in tool_calls:
                if not isinstance(tc, dict):
                    continue
                tc_id = tc.get("tool_call_id")
                tc_name = tc.get("tool_name")
                if isinstance(tc_id, str) and isinstance(tc_name, str):
                    name_map[tc_id] = tc_name
                    continue

                if (
                    isinstance(tc_id, str)
                    and has_single_tool_call
                    and msg.function == EXECUTE_MCP_TOOL_NAME
                ):
                    fallback_name = msg.input_arguments.get("tool_name")
                    if isinstance(fallback_name, str) and fallback_name:
                        name_map[tc_id] = fallback_name
        return name_map

    def _rewrite_tool_call(
        self,
        function_name: str,
        tool_call_id: str,
        args: dict[str, Any] | None,
        tool_name_map: dict[str, str],
    ) -> tuple[str, dict[str, Any] | None]:
        """Rewrite UI protocol wrapper calls to model-visible tool calls."""
        if not self._should_rewrite(function_name, tool_call_id, tool_name_map):
            if function_name == EXECUTE_MCP_TOOL_NAME:
                direct_tool_name = self._direct_mcp_tool_name(
                    tool_call_id,
                    args,
                    tool_name_map,
                )
                if direct_tool_name is not None:
                    return direct_tool_name, self._normalize_rewritten_call_args(
                        function_name, args
                    )
            return function_name, args

        pydantic_tool_name = tool_name_map.get(tool_call_id)
        assert pydantic_tool_name is not None

        normalized_args = self._normalize_rewritten_call_args(function_name, args)

        # Wrap original args inside call_tools' expected shape
        return _CALL_TOOLS, {
            "calls": [{"tool_name": pydantic_tool_name, "arguments": normalized_args}],
        }

    @staticmethod
    def _direct_mcp_tool_name(
        tool_call_id: str,
        args: dict[str, Any] | None,
        tool_name_map: dict[str, str],
    ) -> str | None:
        mapped_name = tool_name_map.get(tool_call_id)
        if mapped_name:
            return mapped_name

        if not isinstance(args, dict):
            return None

        tool_name = args.get("tool_name")
        return tool_name if isinstance(tool_name, str) and tool_name else None

    def _result_tool_name(
        self,
        message: LlmClientFunctionCallResultMessage,
        tool_call_id: str,
        tool_name_map: dict[str, str] | None,
    ) -> str:
        result_tool_name = message.function
        if self._should_rewrite(result_tool_name, tool_call_id, tool_name_map):
            return _CALL_TOOLS

        if result_tool_name != EXECUTE_MCP_TOOL_NAME:
            return result_tool_name

        direct_tool_name = self._direct_mcp_tool_name(
            tool_call_id,
            message.input_arguments,
            tool_name_map or {},
        )
        return direct_tool_name or result_tool_name

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

    @staticmethod
    def _add_text_message(builder: MessagesBuilder, message: LlmClientMessage) -> None:
        content = message.content
        if not isinstance(content, str):
            return
        if message.role == RoleEnum.human:
            builder.add(UserPromptPart(content=content))
        else:
            builder.add(TextPart(content=content))

    def _add_function_call(
        self,
        builder: MessagesBuilder,
        content: LlmClientFunctionCall,
        tool_call_ids: tuple[str, ...],
        tool_name_map: dict[str, str],
    ) -> None:
        """Add the call half of a pair; batched widget calls expand per source."""
        function_name = content.function
        input_args = content.input_arguments or {}
        data_sources = input_args.get("data_sources", [])

        if isinstance(data_sources, list) and len(data_sources) > 1:
            for tc_id, data_source in zip(tool_call_ids, data_sources, strict=True):
                rewritten_name, rewritten_args = self._rewrite_tool_call(
                    function_name,
                    tc_id,
                    {"data_sources": [data_source]},
                    tool_name_map,
                )
                builder.add(
                    ToolCallPart(
                        tool_name=rewritten_name,
                        tool_call_id=tc_id,
                        args=rewritten_args,
                    )
                )
            return

        tool_call_id = tool_call_ids[0]
        rewritten_name, rewritten_args = self._rewrite_tool_call(
            function_name, tool_call_id, content.input_arguments, tool_name_map
        )
        builder.add(
            ToolCallPart(
                tool_name=rewritten_name,
                tool_call_id=tool_call_id,
                args=rewritten_args,
            )
        )

    def _add_result_message(
        self,
        builder: MessagesBuilder,
        message: LlmClientFunctionCallResultMessage,
        tool_call_ids: tuple[str, ...],
        tool_name_map: dict[str, str] | None = None,
    ) -> None:
        """Add the result half of a pair, unbatching `data[i]` per tool_call_id."""
        if len(tool_call_ids) > 1:
            data = message.data or []
            for idx, tool_call_id in enumerate(tool_call_ids):
                builder.add(
                    ToolReturnPart(
                        tool_name=self._result_tool_name(
                            message, tool_call_id, tool_name_map
                        ),
                        tool_call_id=tool_call_id,
                        content=data[idx] if idx < len(data) else None,
                    )
                )
            return

        tool_call_id = tool_call_ids[0]
        builder.add(
            ToolReturnPart(
                tool_name=self._result_tool_name(message, tool_call_id, tool_name_map),
                tool_call_id=tool_call_id,
                content=self._result_content(message),
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
