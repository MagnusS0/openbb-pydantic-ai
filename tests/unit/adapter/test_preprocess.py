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
