from __future__ import annotations

import pytest
from openbb_ai.models import (
    ClientCommandResult,
    LlmClientFunctionCall,
    LlmClientFunctionCallResultMessage,
    LlmClientMessage,
    RoleEnum,
)
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


def test_transform_batch_skips_result_without_tool_call_id(caplog) -> None:
    call_message = LlmClientMessage(
        role=RoleEnum.ai,
        content=LlmClientFunctionCall(
            function="get_widget_data",
            input_arguments={"data_sources": []},
        ),
    )
    result_message = LlmClientFunctionCallResultMessage(
        function="get_widget_data",
        input_arguments={"data_sources": []},
        data=[ClientCommandResult(status="success", message=None)],
        extra_state={},
    )

    with caplog.at_level("WARNING"):
        transformed = MessageTransformer().transform_batch(
            [call_message, result_message]
        )

    assert transformed == []
    assert "Skipping result message for 'get_widget_data'" in caplog.text
