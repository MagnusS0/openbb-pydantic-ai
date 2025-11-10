"""Message transformation utilities for OpenBB Pydantic AI adapter."""

from __future__ import annotations

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

from ._serializers import ContentSerializer


class MessageTransformer:
    """Transforms OpenBB messages to Pydantic AI messages.

    Manages tool call ID consistency across message history.
    """

    def __init__(self, tool_call_id_overrides: dict[str, str] | None = None):
        """Initialize transformer with optional tool call ID overrides.

        Parameters
        ----------
        tool_call_id_overrides : dict[str, str] | None
            Mapping from hash-based IDs to actual tool call IDs for consistency
        """
        self._overrides = tool_call_id_overrides or {}

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
        builder = MessagesBuilder()
        for message in messages:
            if isinstance(message, LlmClientMessage):
                self._add_client_message(builder, message)
            elif isinstance(message, LlmClientFunctionCallResultMessage):
                self._add_result_message(builder, message)
        return builder.messages

    def _add_client_message(
        self, builder: MessagesBuilder, message: LlmClientMessage
    ) -> None:
        """Add a client message to the builder.

        Parameters
        ----------
        builder : MessagesBuilder
            The message builder to add to
        message : LlmClientMessage
            The client message to add
        """
        content = message.content

        if isinstance(content, LlmClientFunctionCall):
            # Use override if available, otherwise use base ID
            from ._utils import hash_tool_call

            base_id = hash_tool_call(content.function, content.input_arguments)
            tool_call_id = self._overrides.get(base_id, base_id)

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

        Parameters
        ----------
        builder : MessagesBuilder
            The message builder to add to
        message : LlmClientFunctionCallResultMessage
            The result message to add
        """
        from ._utils import hash_tool_call

        # Generate base ID and use override if available
        base_id = hash_tool_call(message.function, message.input_arguments)
        tool_call_id = self._overrides.get(base_id, base_id)

        builder.add(
            ToolReturnPart(
                tool_name=message.function,
                tool_call_id=tool_call_id,
                content=ContentSerializer.serialize_result(message),
            )
        )
