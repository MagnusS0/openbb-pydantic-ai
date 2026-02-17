from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from openbb_ai.models import (
    ClientCommandResult,
    LlmClientFunctionCall,
    LlmClientFunctionCallResultMessage,
    LlmClientMessage,
    RoleEnum,
)
from pydantic_ai.tools import RunContext
from pydantic_ai.toolsets import CombinedToolset, FunctionToolset

from openbb_pydantic_ai import OpenBBAIAdapter, OpenBBDeps
from openbb_pydantic_ai._config import (
    EXECUTE_MCP_TOOL_NAME,
    GET_WIDGET_DATA_TOOL_NAME,
    PDF_QUERY_TOOL_NAME,
)
from openbb_pydantic_ai._widget_toolsets import build_widget_tool_name
from openbb_pydantic_ai.tool_discovery import (
    ToolDiscoveryToolset,
    add_to_progressive,
    progressive,
)
from tests.unit.adapter._assertions import tool_call_parts, tool_return_parts

pytestmark = pytest.mark.regression_contract


def _contains_discovery_toolset(toolset: object) -> bool:
    if isinstance(toolset, ToolDiscoveryToolset):
        return True
    if isinstance(toolset, CombinedToolset):
        return any(
            isinstance(inner_toolset, ToolDiscoveryToolset)
            for inner_toolset in toolset.toolsets
        )
    return False


def _build_call_and_result(
    *,
    function: str,
    input_arguments: dict,
    tool_calls: list[dict[str, str]],
) -> tuple[LlmClientMessage, LlmClientFunctionCallResultMessage]:
    call_message = LlmClientMessage(
        role=RoleEnum.ai,
        content=LlmClientFunctionCall(
            function=function,
            input_arguments=input_arguments,
        ),
    )
    result_message = LlmClientFunctionCallResultMessage(
        function=function,
        input_arguments=input_arguments,
        data=[ClientCommandResult(status="success", message=None)],
        extra_state={"tool_calls": tool_calls},
    )
    return call_message, result_message


@pytest.mark.parametrize(
    "enable_progressive",
    [True, False],
    ids=["progressive_enabled", "progressive_disabled"],
)
def test_progressive_toggle_controls_toolset_instructions_and_rewrite(
    widget_collection,
    make_request,
    enable_progressive: bool,
) -> None:
    widget = widget_collection.primary[0]
    tool_name = build_widget_tool_name(widget)

    args = {
        "data_sources": [
            {
                "widget_uuid": str(widget.uuid),
                "input_args": {"symbol": "AAPL"},
            }
        ]
    }

    call_message, result_message = _build_call_and_result(
        function=GET_WIDGET_DATA_TOOL_NAME,
        input_arguments=args,
        tool_calls=[
            {
                "tool_call_id": "call-1",
                "tool_name": tool_name,
            }
        ],
    )

    request = make_request([call_message, result_message], widgets=widget_collection)

    adapter = OpenBBAIAdapter(
        agent=MagicMock(),
        run_input=request,
        enable_progressive_tool_discovery=enable_progressive,
    )

    calls = tool_call_parts(adapter)
    returns = tool_return_parts(adapter)
    assert calls
    assert returns

    if enable_progressive:
        assert _contains_discovery_toolset(adapter.toolset)
        assert "progressive disclosure" in adapter.instructions
        assert calls[0].tool_name == "call_tools"
        assert returns[0].tool_name == "call_tools"
    else:
        assert adapter.toolset is not None
        assert not isinstance(adapter.toolset, ToolDiscoveryToolset)
        assert "progressive disclosure" not in adapter.instructions
        assert calls[0].tool_name == GET_WIDGET_DATA_TOOL_NAME
        assert returns[0].tool_name == GET_WIDGET_DATA_TOOL_NAME


@pytest.mark.parametrize(
    ("tag_mode", "expected_group", "expect_merged"),
    [
        ("direct", "custom_agent_tools", True),
        ("decorator", "decorated_tools", True),
        ("none", None, False),
    ],
    ids=["add_to_progressive", "decorator", "untagged_passthrough"],
)
async def test_runtime_toolset_routing(
    make_request,
    agent_stream_stub,
    tag_mode: str,
    expected_group: str | None,
    expect_merged: bool,
) -> None:
    request = make_request([LlmClientMessage(role=RoleEnum.human, content="Hi")])
    custom_tools = FunctionToolset[OpenBBDeps]()

    if tag_mode == "direct":

        @custom_tools.tool
        def earnings_note(ctx: RunContext[OpenBBDeps], symbol: str) -> str:
            _ = ctx
            return f"Note for {symbol}"

        add_to_progressive(
            custom_tools,
            group="custom_agent_tools",
            description="Custom user toolset",
        )

    elif tag_mode == "decorator":

        @progressive(
            toolset=custom_tools,
            group="decorated_tools",
            description="Tools marked via @progressive",
        )
        @custom_tools.tool
        def earnings_note(ctx: RunContext[OpenBBDeps], symbol: str) -> str:
            _ = ctx
            return f"Note for {symbol}"

    else:

        @custom_tools.tool
        def earnings_note(ctx: RunContext[OpenBBDeps], symbol: str) -> str:
            _ = ctx
            return f"Note for {symbol}"

    adapter = OpenBBAIAdapter(agent=agent_stream_stub, run_input=request)
    stream = adapter.run_stream(toolsets=[custom_tools])
    async for _ in stream:
        pass

    group_ids = {group_id for group_id, _ in adapter._progressive_named_toolsets}
    _, kwargs = agent_stream_stub.calls[0]
    forwarded = kwargs.get("toolsets")
    assert forwarded is not None

    if expect_merged:
        assert expected_group is not None
        assert expected_group in group_ids
        assert len(forwarded) == 1
        assert _contains_discovery_toolset(forwarded[0])
        assert all(toolset is not custom_tools for toolset in forwarded)
    else:
        assert expected_group is None
        assert all(
            group not in group_ids
            for group in {"custom_agent_tools", "decorated_tools"}
        )
        assert any(toolset is custom_tools for toolset in forwarded)


def test_execute_agent_tool_rewrite_uses_inner_parameters(make_request) -> None:
    mcp_tool_name = "equity_fundamental_income"
    wrapped_args = {
        "server_id": "server-1",
        "tool_name": mcp_tool_name,
        "parameters": {
            "provider": "fmp",
            "symbol": "AG",
            "period": "annual",
        },
    }

    call_message, result_message = _build_call_and_result(
        function=EXECUTE_MCP_TOOL_NAME,
        input_arguments=wrapped_args,
        tool_calls=[
            {
                "tool_call_id": "call-mcp-1",
                "tool_name": mcp_tool_name,
            }
        ],
    )

    adapter = OpenBBAIAdapter(
        agent=MagicMock(),
        run_input=make_request([call_message, result_message]),
    )

    calls = tool_call_parts(adapter)
    returns = tool_return_parts(adapter)

    assert len(calls) == 1
    assert calls[0].tool_name == "call_tools"
    assert isinstance(calls[0].args, dict)
    assert calls[0].args["calls"][0]["tool_name"] == mcp_tool_name
    assert calls[0].args["calls"][0]["arguments"] == wrapped_args["parameters"]

    assert len(returns) == 1
    assert returns[0].tool_name == "call_tools"


def test_rewrite_is_gated_by_tool_call_id_mapping(
    widget_collection, make_request
) -> None:
    widget = widget_collection.primary[0]
    tool_name = build_widget_tool_name(widget)

    args_a = {
        "data_sources": [
            {"widget_uuid": str(widget.uuid), "input_args": {"symbol": "AAPL"}}
        ]
    }
    args_b = {
        "data_sources": [
            {"widget_uuid": str(widget.uuid), "input_args": {"symbol": "MSFT"}}
        ]
    }

    call_a, result_a = _build_call_and_result(
        function=GET_WIDGET_DATA_TOOL_NAME,
        input_arguments=args_a,
        tool_calls=[{"tool_call_id": "id-a", "tool_name": tool_name}],
    )
    call_b, result_b = _build_call_and_result(
        function=GET_WIDGET_DATA_TOOL_NAME,
        input_arguments=args_b,
        tool_calls=[{"tool_call_id": "id-b"}],
    )

    adapter = OpenBBAIAdapter(
        agent=MagicMock(),
        run_input=make_request(
            [call_a, result_a, call_b, result_b], widgets=widget_collection
        ),
    )

    call_names = {
        part.tool_call_id: part.tool_name for part in tool_call_parts(adapter)
    }
    return_names = {
        part.tool_call_id: part.tool_name for part in tool_return_parts(adapter)
    }

    assert call_names["id-a"] == "call_tools"
    assert return_names["id-a"] == "call_tools"
    assert call_names["id-b"] == GET_WIDGET_DATA_TOOL_NAME
    assert return_names["id-b"] == GET_WIDGET_DATA_TOOL_NAME


async def test_progressive_toolset_keeps_pdf_query_direct(
    make_request,
    agent_stream_stub,
) -> None:
    pytest.importorskip("openbb_pydantic_ai.pdf")
    request = make_request([LlmClientMessage(role=RoleEnum.human, content="Hi")])
    adapter = OpenBBAIAdapter(agent=agent_stream_stub, run_input=request)

    group_ids = {group_id for group_id, _ in adapter._progressive_named_toolsets}
    assert all("pdf" not in group_id for group_id in group_ids)

    stream = adapter.run_stream()
    async for _ in stream:
        pass

    _, kwargs = agent_stream_stub.calls[0]
    forwarded = kwargs.get("toolsets")
    assert forwarded is not None
    assert len(forwarded) == 1

    combined = forwarded[0]
    assert isinstance(combined, CombinedToolset)
    inner_toolsets = combined.toolsets

    assert any(isinstance(toolset, ToolDiscoveryToolset) for toolset in inner_toolsets)
    assert any(
        isinstance(toolset, FunctionToolset) and PDF_QUERY_TOOL_NAME in toolset.tools
        for toolset in inner_toolsets
    )
