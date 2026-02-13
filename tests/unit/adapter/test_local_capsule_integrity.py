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


def _deferred_pair(
    *,
    widget_uuid: str,
    widget_tool_name: str,
    symbol: str,
    deferred_id: str,
    capsule_payload: Any | None,
) -> tuple[LlmClientMessage, LlmClientFunctionCallResultMessage]:
    deferred_args = {
        "data_sources": [
            {
                "widget_uuid": widget_uuid,
                "input_args": {"symbol": symbol},
            }
        ]
    }
    call = LlmClientMessage(
        role=RoleEnum.ai,
        content=LlmClientFunctionCall(
            function=GET_WIDGET_DATA_TOOL_NAME,
            input_arguments=deferred_args,
        ),
    )
    extra_state: dict = {
        "tool_calls": [
            {
                "tool_call_id": deferred_id,
                "tool_name": widget_tool_name,
            }
        ]
    }
    if capsule_payload is not None:
        extra_state[LOCAL_TOOL_CAPSULE_EXTRA_STATE_KEY] = capsule_payload
    result = LlmClientFunctionCallResultMessage(
        function=GET_WIDGET_DATA_TOOL_NAME,
        input_arguments=deferred_args,
        data=[ClientCommandResult(status="success", message=None)],
        extra_state=extra_state,
    )
    return call, result


async def test_invalid_capsule_is_ignored_without_crashing(
    make_request, widget_collection
) -> None:
    widget = widget_collection.primary[0]
    widget_tool_name = build_widget_tool_name(widget)
    entry = LocalToolEntry(
        tool_call_id="local-1",
        tool_name="list_tools",
        args={},
        result="# x",
    )
    invalid_payload = {"bad": pack_tool_history([entry])}

    call, result = _deferred_pair(
        widget_uuid=str(widget.uuid),
        widget_tool_name=widget_tool_name,
        symbol="AAPL",
        deferred_id="deferred-1",
        capsule_payload=invalid_payload,
    )
    request = make_request(
        [LlmClientMessage(role=RoleEnum.human, content="Hi"), call, result],
        widgets=widget_collection,
    )

    adapter = await OpenBBAIAdapter.from_request(
        cast(Any, _RequestStub(request.model_dump_json().encode())),
        agent=MagicMock(),
    )

    calls = tool_call_parts(adapter)
    local_calls = [
        call_part for call_part in calls if call_part.tool_name == "list_tools"
    ]
    assert local_calls == []


async def test_duplicate_capsule_payload_is_rehydrated_only_once(
    make_request, widget_collection
) -> None:
    widget = widget_collection.primary[0]
    widget_tool_name = build_widget_tool_name(widget)
    entry = LocalToolEntry(
        tool_call_id="local-dup",
        tool_name="list_tools",
        args={},
        result="# openbb_viz_tools\ncount: 1\n- openbb_create_chart",
    )
    payload = pack_tool_history([entry])

    call_1, result_1 = _deferred_pair(
        widget_uuid=str(widget.uuid),
        widget_tool_name=widget_tool_name,
        symbol="AAPL",
        deferred_id="deferred-1",
        capsule_payload=payload,
    )
    call_2, result_2 = _deferred_pair(
        widget_uuid=str(widget.uuid),
        widget_tool_name=widget_tool_name,
        symbol="MSFT",
        deferred_id="deferred-2",
        capsule_payload=payload,
    )
    request = make_request(
        [
            LlmClientMessage(role=RoleEnum.human, content="Hi"),
            call_1,
            result_1,
            call_2,
            result_2,
        ],
        widgets=widget_collection,
    )

    adapter = await OpenBBAIAdapter.from_request(
        cast(Any, _RequestStub(request.model_dump_json().encode())),
        agent=MagicMock(),
    )

    calls = tool_call_parts(adapter)
    local_calls = [
        call_part for call_part in calls if call_part.tool_call_id == "local-dup"
    ]
    assert len(local_calls) == 1
