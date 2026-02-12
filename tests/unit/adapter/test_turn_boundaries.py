from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from openbb_ai.models import LlmClientMessage, RoleEnum

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
