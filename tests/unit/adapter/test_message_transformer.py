from __future__ import annotations

import pytest
from openbb_ai.models import LlmClientMessage, RoleEnum
from pydantic_ai.messages import TextPart, UserPromptPart

from openbb_pydantic_ai._message_transformer import MessageTransformer

pytestmark = pytest.mark.regression_contract


@pytest.mark.parametrize(
    ("role", "expected_part_type"),
    [
        (RoleEnum.human, UserPromptPart),
        (RoleEnum.ai, TextPart),
        (RoleEnum.tool, TextPart),
    ],
    ids=["human", "ai", "tool"],
)
def test_transform_batch_handles_all_supported_text_roles(
    role: RoleEnum,
    expected_part_type: type[UserPromptPart] | type[TextPart],
) -> None:
    transformed = MessageTransformer().transform_batch(
        [LlmClientMessage(role=role, content="hello")]
    )

    parts = [part for message in transformed for part in message.parts]
    assert len(parts) == 1
    assert isinstance(parts[0], expected_part_type)
