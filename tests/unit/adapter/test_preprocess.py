from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from openbb_ai.models import LlmClientMessage, RoleEnum

from openbb_pydantic_ai import OpenBBAIAdapter
from openbb_pydantic_ai import _adapter as adapter_module

pytestmark = pytest.mark.regression_contract


async def test_from_request_preprocesses_messages(mocker, make_request) -> None:
    """Adapter should preprocess PDF-bearing messages before transform/build."""

    original_message = LlmClientMessage(role=RoleEnum.human, content="original")
    processed_message = LlmClientMessage(role=RoleEnum.human, content="processed")
    run_input = make_request([original_message])

    preprocess_mock = mocker.patch.object(
        adapter_module,
        "preprocess_pdf_in_messages",
        new_callable=AsyncMock,
        return_value=[processed_message],
    )

    class _RequestStub:
        headers = {"accept": "text/event-stream"}

        def __init__(self, payload: bytes):
            self._payload = payload

        async def body(self) -> bytes:
            return self._payload

    request = _RequestStub(run_input.model_dump_json().encode())
    adapter = await OpenBBAIAdapter.from_request(request, agent=MagicMock())  # type: ignore[arg-type]

    preprocess_mock.assert_awaited_once()
    assert adapter._base_messages == [processed_message]


async def test_dispatch_request_uses_from_request(mocker) -> None:
    request = MagicMock()
    agent = MagicMock()
    stream = object()
    response = object()

    adapter = MagicMock()
    adapter.run_stream.return_value = stream
    adapter.streaming_response.return_value = response

    from_request_mock = mocker.patch.object(
        OpenBBAIAdapter,
        "from_request",
        new_callable=AsyncMock,
        return_value=adapter,
    )

    result = await OpenBBAIAdapter.dispatch_request(
        request,
        agent=agent,
        enable_progressive_tool_discovery=False,
    )

    from_request_mock.assert_awaited_once_with(
        request,
        agent=agent,
        enable_progressive_tool_discovery=False,
        enable_local_tool_history_capsule=True,
    )
    adapter.run_stream.assert_called_once()
    adapter.streaming_response.assert_called_once_with(stream)
    assert result is response
