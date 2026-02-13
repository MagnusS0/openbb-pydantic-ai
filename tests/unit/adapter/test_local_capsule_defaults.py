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
from tests.unit.adapter._assertions import tool_call_parts

pytestmark = pytest.mark.regression_contract


class _RequestStub:
    headers = {"accept": "text/event-stream"}

    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    async def body(self) -> bytes:
        return self._payload


def test_local_capsule_defaults_enabled(make_request) -> None:
    adapter = OpenBBAIAdapter(
        agent=MagicMock(),
        run_input=make_request([LlmClientMessage(role=RoleEnum.human, content="Hi")]),
    )
    assert adapter.enable_local_tool_history_capsule is True


async def test_disabling_capsule_bypasses_rehydration(
    make_request, widget_collection
) -> None:
    widget = widget_collection.primary[0]
    widget_tool_name = build_widget_tool_name(widget)

    entry = LocalToolEntry(
        tool_call_id="local-off",
        tool_name="list_tools",
        args={},
        result="# tools",
    )
    capsule = pack_tool_history([entry])

    deferred_args = {
        "data_sources": [
            {
                "widget_uuid": str(widget.uuid),
                "input_args": {"symbol": "AAPL"},
            }
        ]
    }
    call_message = LlmClientMessage(
        role=RoleEnum.ai,
        content=LlmClientFunctionCall(
            function=GET_WIDGET_DATA_TOOL_NAME,
            input_arguments=deferred_args,
        ),
    )
    result_message = LlmClientFunctionCallResultMessage(
        function=GET_WIDGET_DATA_TOOL_NAME,
        input_arguments=deferred_args,
        data=[ClientCommandResult(status="success", message=None)],
        extra_state={
            "tool_calls": [
                {
                    "tool_call_id": "deferred-off",
                    "tool_name": widget_tool_name,
                }
            ],
            LOCAL_TOOL_CAPSULE_EXTRA_STATE_KEY: capsule,
        },
    )

    request = make_request(
        [
            LlmClientMessage(role=RoleEnum.human, content="Hi"),
            call_message,
            result_message,
        ],
        widgets=widget_collection,
    )

    adapter = await OpenBBAIAdapter.from_request(
        cast(Any, _RequestStub(request.model_dump_json().encode())),
        agent=MagicMock(),
        enable_local_tool_history_capsule=False,
    )

    calls = tool_call_parts(adapter)
    local_calls = [
        call_part for call_part in calls if call_part.tool_call_id == "local-off"
    ]
    assert local_calls == []
    stream = adapter.build_event_stream()
    assert stream.enable_local_tool_history_capsule is False
