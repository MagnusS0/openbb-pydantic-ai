from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from openbb_ai.models import LlmClientMessage, RoleEnum
from pydantic_ai.ui import DEFAULT_ALLOWED_CONTENT_TYPES

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
        headers = {
            "accept": "text/event-stream",
            "content-type": "application/json",
        }

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
        manage_system_prompt="server",
        allowed_file_url_schemes=frozenset({"http", "https"}),
        allowed_file_url_force_download=frozenset(),
        allow_uploaded_files=False,
        strip_workspace_refs=True,
        allowed_content_types=DEFAULT_ALLOWED_CONTENT_TYPES,
    )
    adapter.run_stream.assert_called_once()
    adapter.streaming_response.assert_called_once_with(stream)
    assert result is response


async def test_dispatch_request_forwards_on_cancel(mocker) -> None:
    adapter = MagicMock()
    mocker.patch.object(
        OpenBBAIAdapter,
        "from_request",
        new_callable=AsyncMock,
        return_value=adapter,
    )
    on_cancel = MagicMock()

    await OpenBBAIAdapter.dispatch_request(
        MagicMock(), agent=MagicMock(), on_cancel=on_cancel
    )

    assert adapter.run_stream.call_args.kwargs["on_cancel"] is on_cancel


def test_build_run_input_drops_unsupported_widget_param_types() -> None:
    body = json.dumps(
        {
            "messages": [{"role": "human", "content": "hi"}],
            "widgets": {
                "secondary": [
                    {
                        "origin": "o",
                        "widget_id": "w",
                        "name": "W",
                        "description": "d",
                        "params": [
                            {"name": "run", "type": "button", "description": "go"},
                            {"name": "symbol", "type": "text", "description": "s"},
                        ],
                    }
                ]
            },
        }
    ).encode()

    run_input = OpenBBAIAdapter.build_run_input(body)

    assert run_input.widgets is not None
    params = run_input.widgets.secondary[0].params
    assert [p.name for p in params] == ["symbol"]
