from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from openbb_ai.models import (
    ClientCommandResult,
    LlmClientFunctionCall,
    LlmClientFunctionCallResultMessage,
    LlmClientMessage,
    RoleEnum,
)
from pydantic_ai.messages import ToolReturnPart

from openbb_pydantic_ai import OpenBBAIAdapter
from tests.unit.adapter._assertions import visible_turn_text

pytestmark = pytest.mark.regression_contract


def test_adapter_preserves_turn_boundaries_without_duplication(make_request):
    """Ensure messages are not re-grouped or duplicated across turns."""

    first_user = LlmClientMessage(
        role=RoleEnum.human,
        content="Hey can you get ticker info on AAPL",
    )
    assistant_reply = LlmClientMessage(
        role=RoleEnum.ai,
        content="Here's the latest ticker information for AAPL.",
    )
    follow_up = LlmClientMessage(
        role=RoleEnum.human,
        content="How many times have you shown that ticker info?",
    )

    request = make_request([first_user, assistant_reply, follow_up])
    adapter = OpenBBAIAdapter(agent=MagicMock(), run_input=request)

    turns = visible_turn_text(adapter)

    assert len(turns) == 3
    assert turns[0] == "Hey can you get ticker info on AAPL"
    assert turns[1] == "Here's the latest ticker information for AAPL."
    assert turns[2] == "How many times have you shown that ticker info?"


def test_adapter_ignores_late_tool_result_after_final_answer(make_request, caplog):
    """A delayed tool result after assistant text must not restart the turn."""

    call_message = LlmClientMessage(
        role=RoleEnum.ai,
        content=LlmClientFunctionCall(
            function="get_widget_data",
            input_arguments={"symbol": "AAPL"},
        ),
    )
    final_answer = LlmClientMessage(
        role=RoleEnum.ai,
        content="AAPL is trading higher today.",
    )
    late_result = LlmClientFunctionCallResultMessage(
        function="get_widget_data",
        input_arguments={"symbol": "AAPL"},
        data=[ClientCommandResult(status="success", message=None)],
        extra_state={"tool_calls": [{"tool_call_id": "late-call"}]},
    )

    request = make_request([call_message, final_answer, late_result])

    with caplog.at_level("WARNING"):
        adapter = OpenBBAIAdapter(agent=MagicMock(), run_input=request)
        parts = [part for message in adapter.messages for part in message.parts]

    assert adapter._pending_results == []
    assert late_result not in adapter._base_messages
    assert not any(isinstance(part, ToolReturnPart) for part in parts)
    assert "Ignoring 1 trailing deferred tool result" in caplog.text
