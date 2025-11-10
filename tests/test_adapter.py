from __future__ import annotations

from unittest.mock import MagicMock

from openbb_ai.models import (
    ClientCommandResult,
    LlmClientFunctionCall,
    LlmClientFunctionCallResultMessage,
    LlmClientMessage,
    RoleEnum,
)
from pydantic_ai.messages import SystemPromptPart, ToolCallPart, ToolReturnPart

from openbb_pydantic_ai import OpenBBAIAdapter, build_widget_tool_name


def test_adapter_injects_system_prompt(sample_context, make_request, human_message):
    request = make_request(
        [human_message],
        context=[sample_context],
        urls=["https://example.com"],
    )

    adapter = OpenBBAIAdapter(agent=MagicMock(), run_input=request)

    system_parts = [
        part
        for message in adapter.messages
        for part in getattr(message, "parts", [])
        if isinstance(part, SystemPromptPart)
    ]

    assert system_parts, "Adapter should inject a context system prompt"
    assert "Test Context" in system_parts[0].content
    assert "https://example.com" in system_parts[0].content


def test_adapter_preserves_tool_call_ids(widget_collection, make_request):
    widget = widget_collection.primary[0]
    tool_name = build_widget_tool_name(widget)

    call_message = LlmClientMessage(
        role=RoleEnum.ai,
        content=LlmClientFunctionCall(
            function=tool_name,
            input_arguments={"symbol": "AAPL"},
        ),
    )
    result_message = LlmClientFunctionCallResultMessage(
        function=tool_name,
        input_arguments={"symbol": "AAPL"},
        data=[ClientCommandResult(status="success", message=None)],
        extra_state={"tool_call_id": "tool-123"},
    )
    request = make_request(
        [call_message, result_message],
        widgets=widget_collection,
    )

    adapter = OpenBBAIAdapter(agent=MagicMock(), run_input=request)

    tool_parts = [
        part
        for message in adapter.messages
        for part in getattr(message, "parts", [])
        if isinstance(part, ToolCallPart)
    ]
    assert tool_parts and tool_parts[0].tool_call_id == "tool-123"

    return_parts = [
        part
        for message in adapter.messages
        for part in getattr(message, "parts", [])
        if isinstance(part, ToolReturnPart)
    ]
    assert return_parts and return_parts[0].tool_call_id == "tool-123"

    assert adapter.deferred_tool_results is None
