"""Message transformation utilities for OpenBB Pydantic AI adapter."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

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

from openbb_pydantic_ai._serializers import serialize_result
from openbb_pydantic_ai._utils import extract_tool_call_id


class MessageTransformer:
    """Transforms OpenBB messages to Pydantic AI messages."""

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
        call_counters: dict[str, int] = {}

        builder = MessagesBuilder()
        for message in messages:
            if isinstance(message, LlmClientMessage):
                self._add_client_message(
                    builder, message, tool_call_id_map, call_counters
                )
            elif isinstance(message, LlmClientFunctionCallResultMessage):
                self._add_result_message(builder, message)
        return builder.messages

    def _build_tool_call_id_map(
        self, messages: Sequence[LlmMessage]
    ) -> dict[str, list[str]]:
        """Build a lookup of function_name → all tool_call_ids in order.

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

    def _add_client_message(
        self,
        builder: MessagesBuilder,
        message: LlmClientMessage,
        tool_call_id_map: dict[str, list[str]],
        call_counters: dict[str, int],
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

                    builder.add(
                        ToolCallPart(
                            tool_name=function_name,
                            tool_call_id=tool_call_ids[idx],
                            args={"data_sources": [data_source]},
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

            builder.add(
                ToolCallPart(
                    tool_name=content.function,
                    tool_call_id=tool_call_id,
                    args=content.input_arguments,
                )
            )
            return

        if isinstance(content, str):
            if message.role == RoleEnum.human:
                builder.add(UserPromptPart(content=content))
            elif message.role == RoleEnum.ai:
                builder.add(TextPart(content=content))
            else:
                builder.add(TextPart(content=content))

    def _add_result_message(
        self,
        builder: MessagesBuilder,
        message: LlmClientFunctionCallResultMessage,
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
        """
        extra_state = message.extra_state or {}
        tool_calls = extra_state.get("tool_calls", [])

        if isinstance(tool_calls, list) and len(tool_calls) > 1:
            self._add_unbatched_results(builder, message)
            return

        tool_call_id = extract_tool_call_id(message)
        builder.add(
            ToolReturnPart(
                tool_name=message.function,
                tool_call_id=tool_call_id,
                content=serialize_result(message),
            )
        )

    def _add_unbatched_results(
        self,
        builder: MessagesBuilder,
        message: LlmClientFunctionCallResultMessage,
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

            builder.add(
                ToolReturnPart(
                    tool_name=message.function,
                    tool_call_id=tool_call_id,
                    content=result_data,
                )
            )
