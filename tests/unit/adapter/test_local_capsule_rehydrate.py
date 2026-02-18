from __future__ import annotations

from typing import Any, cast
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
from openbb_pydantic_ai._config import (
    GET_WIDGET_DATA_TOOL_NAME,
    LOCAL_TOOL_CAPSULE_EXTRA_STATE_KEY,
)
from openbb_pydantic_ai._local_tool_capsule import (
    LocalToolEntry,
    pack_tool_history,
)
from openbb_pydantic_ai._widget_toolsets import build_widget_tool_name
from tests.unit.adapter._assertions import tool_call_parts, tool_return_parts

pytestmark = pytest.mark.regression_contract


class _RequestStub:
    headers = {"accept": "text/event-stream"}

    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    async def body(self) -> bytes:
        return self._payload


async def test_rehydrate_injects_rehydrated_local_tool_history(
    make_request, widget_collection
) -> None:
    widget = widget_collection.primary[0]
    widget_tool_name = build_widget_tool_name(widget)

    local_entry = LocalToolEntry(
        tool_call_id="local-1",
        tool_name="list_tools",
        args={"group": "openbb_viz_tools"},
        result="# openbb_viz_tools\ncount: 1\n- openbb_create_chart",
    )
    capsule = pack_tool_history([local_entry])

    deferred_args = {
        "data_sources": [
            {
                "widget_uuid": str(widget.uuid),
                "input_args": {"symbol": "AAPL"},
            }
        ]
    }

    user_message = LlmClientMessage(role=RoleEnum.human, content="Show me AAPL")
    deferred_call = LlmClientMessage(
        role=RoleEnum.ai,
        content=LlmClientFunctionCall(
            function=GET_WIDGET_DATA_TOOL_NAME,
            input_arguments=deferred_args,
        ),
    )
    deferred_result = LlmClientFunctionCallResultMessage(
        function=GET_WIDGET_DATA_TOOL_NAME,
        input_arguments=deferred_args,
        data=[ClientCommandResult(status="success", message=None)],
        extra_state={
            "tool_calls": [
                {
                    "tool_call_id": "deferred-1",
                    "tool_name": widget_tool_name,
                }
            ],
            LOCAL_TOOL_CAPSULE_EXTRA_STATE_KEY: capsule,
        },
    )
    request = make_request(
        [user_message, deferred_call, deferred_result],
        widgets=widget_collection,
    )

    adapter = await OpenBBAIAdapter.from_request(
        cast(Any, _RequestStub(request.model_dump_json().encode())),
        agent=MagicMock(),
    )

    calls = tool_call_parts(adapter)
    returns = tool_return_parts(adapter)

    assert len(calls) >= 2
    assert len(returns) >= 2
    assert calls[0].tool_name == "list_tools"
    assert calls[0].tool_call_id == "local-1"
    assert returns[0].tool_name == "list_tools"
    assert returns[0].tool_call_id == "local-1"
    assert returns[0].content == local_entry.result

    assert calls[1].tool_name == "call_tools"
    assert calls[1].tool_call_id == "deferred-1"
    assert returns[1].tool_name == "call_tools"
    assert returns[1].tool_call_id == "deferred-1"
