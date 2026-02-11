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

from openbb_pydantic_ai import OpenBBAIAdapter
from openbb_pydantic_ai._widget_toolsets import build_widget_tool_name
from tests.unit.adapter._assertions import tool_call_parts, tool_return_parts

pytestmark = pytest.mark.regression_contract


@pytest.mark.parametrize(
    ("symbol", "tool_call_id", "include_tool_name"),
    [
        ("AAPL", "tool-123", False),
        ("NVDA", "call-456", True),
    ],
    ids=["simple_tool_call_id", "extra_state_tool_calls_regression"],
)
def test_adapter_preserves_tool_call_ids_from_extra_state(
    widget_collection,
    make_request,
    symbol: str,
    tool_call_id: str,
    include_tool_name: bool,
) -> None:
    """Deferred results must keep tool_call_id from extra_state.tool_calls.

    Regression contract: when tool_call_id is only present in
    extra_state['tool_calls'][0]['tool_call_id'], IDs must not be re-hashed.
    """

    widget = widget_collection.primary[0]
    tool_name = build_widget_tool_name(widget)

    call_message = LlmClientMessage(
        role=RoleEnum.ai,
        content=LlmClientFunctionCall(
            function=tool_name,
            input_arguments={"symbol": symbol},
        ),
    )

    tool_call_entry: dict[str, str] = {"tool_call_id": tool_call_id}
    if include_tool_name:
        tool_call_entry["tool_name"] = tool_name

    result_message = LlmClientFunctionCallResultMessage(
        function=tool_name,
        input_arguments={"symbol": symbol},
        data=[ClientCommandResult(status="success", message=None)],
        extra_state={"tool_calls": [tool_call_entry]},
    )

    request = make_request([call_message, result_message], widgets=widget_collection)

    adapter = OpenBBAIAdapter(agent=MagicMock(), run_input=request)

    calls = tool_call_parts(adapter)
    returns = tool_return_parts(adapter)

    assert calls and calls[0].tool_call_id == tool_call_id
    assert returns and returns[0].tool_call_id == tool_call_id
    assert adapter._pending_results == [result_message]
