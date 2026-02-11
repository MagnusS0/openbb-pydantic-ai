from __future__ import annotations

import pytest
from openbb_ai.models import LlmClientMessage, RoleEnum

from openbb_pydantic_ai import OpenBBAIAdapter

pytestmark = pytest.mark.regression_contract


@pytest.mark.parametrize(
    ("stream_method", "content"),
    [
        ("run_stream_native", "Hi"),
        ("run_stream", "Hello again"),
    ],
    ids=["native", "wrapper"],
)
async def test_run_stream_methods_do_not_double_messages(
    make_request,
    agent_stream_stub,
    stream_method: str,
    content: str,
) -> None:
    """Both stream entrypoints should pass history once without duplication."""

    user_msg = LlmClientMessage(role=RoleEnum.human, content=content)
    request = make_request([user_msg])

    adapter = OpenBBAIAdapter(agent=agent_stream_stub, run_input=request)

    stream = getattr(adapter, stream_method)()
    async for _ in stream:
        pass

    assert len(agent_stream_stub.calls) == 1
    _, call_kwargs = agent_stream_stub.calls[0]
    history = call_kwargs["message_history"]

    assert history == adapter.messages
    assert len(history) == 1
