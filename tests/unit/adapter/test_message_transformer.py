from __future__ import annotations

import pytest
from openbb_ai.models import (
    ClientCommandResult,
    LlmClientFunctionCall,
    LlmClientFunctionCallResultMessage,
    LlmClientMessage,
    RoleEnum,
)
from pydantic_ai.messages import TextPart, ToolCallPart, ToolReturnPart, UserPromptPart

from openbb_pydantic_ai._message_transformer import MessageTransformer

pytestmark = pytest.mark.regression_contract


def _derived_id(function: str, index: int) -> str:
    return f"openbb-pair-{function}-{index}"


def _call(function: str, args: dict | None = None) -> LlmClientMessage:
    return LlmClientMessage(
        role=RoleEnum.ai,
        content=LlmClientFunctionCall(function=function, input_arguments=args or {}),
    )


def _result(
    function: str, tool_call_id: str | None, args: dict | None = None
) -> LlmClientFunctionCallResultMessage:
    extra_state = (
        {"tool_calls": [{"tool_call_id": tool_call_id}]} if tool_call_id else {}
    )
    return LlmClientFunctionCallResultMessage(
        function=function,
        input_arguments=args or {},
        data=[ClientCommandResult(status="success", message=None)],
        extra_state=extra_state,
    )


def _tool_parts(messages) -> list[ToolCallPart | ToolReturnPart]:
    return [
        part
        for message in messages
        for part in message.parts
        if isinstance(part, (ToolCallPart, ToolReturnPart))
    ]


def test_other_agent_pairs_do_not_shift_our_tool_call_ids() -> None:
    history = [
        _call("get_widget_data"),
        _result("get_widget_data", "other-agent-1"),
        _call("get_widget_data"),
        _result("get_widget_data", "other-agent-2"),
        _call("get_widget_data"),
        _result("get_widget_data", "ours"),
    ]

    parts = _tool_parts(MessageTransformer().transform_batch(history))

    assert [(type(p), p.tool_call_id) for p in parts] == [
        (ToolCallPart, "other-agent-1"),
        (ToolReturnPart, "other-agent-1"),
        (ToolCallPart, "other-agent-2"),
        (ToolReturnPart, "other-agent-2"),
        (ToolCallPart, "ours"),
        (ToolReturnPart, "ours"),
    ]


def test_orphan_call_from_deleted_pair_is_skipped() -> None:
    history = [
        _call("get_widget_data"),
        LlmClientMessage(role=RoleEnum.human, content="retry"),
        _call("get_widget_data"),
        _result("get_widget_data", "kept"),
    ]

    parts = _tool_parts(MessageTransformer().transform_batch(history))

    assert [(type(p), p.tool_call_id) for p in parts] == [
        (ToolCallPart, "kept"),
        (ToolReturnPart, "kept"),
    ]


def test_orphan_result_from_deleted_pair_is_skipped() -> None:
    history = [
        _result("get_widget_data", "dangling"),
        LlmClientMessage(role=RoleEnum.human, content="retry"),
        _call("get_widget_data"),
        _result("get_widget_data", "kept"),
    ]

    parts = _tool_parts(MessageTransformer().transform_batch(history))

    assert {p.tool_call_id for p in parts} == {"kept"}
    assert len(parts) == 2


def test_result_for_different_function_does_not_pair() -> None:
    history = [_call("get_widget_data"), _result("execute_agent_tool", "x")]

    assert _tool_parts(MessageTransformer().transform_batch(history)) == []


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


def test_transform_batch_derives_deterministic_id_when_result_lacks_tool_calls() -> (
    None
):
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

    transformed = MessageTransformer().transform_batch([call_message, result_message])

    parts = [part for message in transformed for part in message.parts]
    call_part, return_part = parts
    assert isinstance(call_part, ToolCallPart)
    assert isinstance(return_part, ToolReturnPart)
    assert call_part.tool_call_id == return_part.tool_call_id
    assert call_part.tool_call_id == _derived_id("get_widget_data", 0)


def test_transform_batch_skips_partial_batched_call_when_ids_missing(caplog) -> None:
    batched_call = LlmClientMessage(
        role=RoleEnum.ai,
        content=LlmClientFunctionCall(
            function="get_widget_data",
            input_arguments={
                "data_sources": [
                    {"widget_uuid": "w1", "input_args": {"symbol": "AAPL"}},
                    {"widget_uuid": "w1", "input_args": {"symbol": "MSFT"}},
                    {"widget_uuid": "w1", "input_args": {"symbol": "NVDA"}},
                ]
            },
        ),
    )
    batched_result = LlmClientFunctionCallResultMessage(
        function="get_widget_data",
        input_arguments={"data_sources": []},
        data=[
            ClientCommandResult(status="success", message="AAPL"),
            ClientCommandResult(status="success", message="MSFT"),
        ],
        extra_state={
            "tool_calls": [
                {"tool_call_id": "id-a"},
                {"tool_call_id": "id-b"},
            ]
        },
    )
    followup_call = LlmClientMessage(
        role=RoleEnum.ai,
        content=LlmClientFunctionCall(
            function="get_widget_data",
            input_arguments={
                "data_sources": [
                    {"widget_uuid": "w1", "input_args": {"symbol": "TSLA"}},
                ]
            },
        ),
    )

    with caplog.at_level("WARNING"):
        transformed = MessageTransformer().transform_batch(
            [batched_call, batched_result, followup_call]
        )

    calls = [
        part
        for message in transformed
        for part in message.parts
        if isinstance(part, ToolCallPart)
    ]

    assert "Not enough tool_call_ids for batched call to get_widget_data" in caplog.text
    returns = [
        part
        for message in transformed
        for part in message.parts
        if isinstance(part, ToolReturnPart)
    ]
    assert calls == []
    assert returns == []
