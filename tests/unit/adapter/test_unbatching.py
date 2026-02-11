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
from tests.unit.adapter._assertions import tool_call_parts, tool_return_parts

pytestmark = pytest.mark.regression_contract


def test_adapter_unbatches_multiple_widget_calls(widget_collection, make_request):
    """Batched widget calls should be unbatched on both sides for proper matching.

    When the UI returns a batched get_widget_data call with multiple data_sources,
    both the call and result should be unbatched so each ToolCallPart has a matching
    ToolReturnPart with the same tool_call_id.
    """
    widget = widget_collection.primary[0]

    batched_call = LlmClientMessage(
        role=RoleEnum.ai,
        content=LlmClientFunctionCall(
            function="get_widget_data",
            input_arguments={
                "data_sources": [
                    {
                        "widget_uuid": str(widget.uuid),
                        "origin": widget.origin,
                        "id": widget.widget_id,
                        "input_args": {
                            "symbol": "AAPL",
                            "selectedGroup": "balance_sheet",
                        },
                    },
                    {
                        "widget_uuid": str(widget.uuid),
                        "origin": widget.origin,
                        "id": widget.widget_id,
                        "input_args": {
                            "symbol": "AAPL",
                            "selectedGroup": "income_statement",
                        },
                    },
                ]
            },
        ),
    )

    batched_result = LlmClientFunctionCallResultMessage(
        function="get_widget_data",
        input_arguments={
            "data_sources": [
                {
                    "widget_uuid": str(widget.uuid),
                    "origin": widget.origin,
                    "id": widget.widget_id,
                    "input_args": {"symbol": "AAPL", "selectedGroup": "balance_sheet"},
                },
                {
                    "widget_uuid": str(widget.uuid),
                    "origin": widget.origin,
                    "id": widget.widget_id,
                    "input_args": {
                        "symbol": "AAPL",
                        "selectedGroup": "income_statement",
                    },
                },
            ]
        },
        data=[
            ClientCommandResult(status="success", message="Balance sheet data"),
            ClientCommandResult(status="success", message="Income statement data"),
        ],
        extra_state={
            "tool_calls": [
                {"tool_call_id": "call_123", "widget_uuid": str(widget.uuid)},
                {"tool_call_id": "call_456", "widget_uuid": str(widget.uuid)},
            ]
        },
    )

    request = make_request(
        [batched_call, batched_result],
        widgets=widget_collection,
    )

    adapter = OpenBBAIAdapter(agent=MagicMock(), run_input=request)

    calls = tool_call_parts(adapter)
    returns = tool_return_parts(adapter)

    assert len(calls) == 2
    assert len(returns) == 2

    call_ids = {part.tool_call_id for part in calls}
    return_ids = {part.tool_call_id for part in returns}
    assert call_ids == {"call_123", "call_456"}
    assert return_ids == {"call_123", "call_456"}

    for part in calls:
        data_sources = (
            part.args.get("data_sources", []) if isinstance(part.args, dict) else []
        )
        assert len(data_sources) == 1
